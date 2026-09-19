"""Smallest possible end-to-end test-time-training (TTT) check, run BEFORE trusting this on
real unseen test grids: fine-tune a per-task LoRA adapter on augmented copies of a training
task's demonstration pairs, then check the loss actually drops and the model can complete a
held-out augmented view. This is a mechanism sanity check, not an accuracy measurement.
"""
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.augment import make_augmented_views
from src.data import load_split
from src.serialize import format_completion, text_to_grid

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
N_AUGMENTED_VIEWS = 12
N_STEPS = 60
LR = 1e-4


def build_example(tokenizer, gin, gout, device):
    prompt, target = format_completion(gin, gout)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    target_ids = tokenizer(target + tokenizer.eos_token, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([prompt_ids, target_ids]).unsqueeze(0).to(device)
    labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids]).unsqueeze(0).to(device)
    return input_ids, labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tasks = load_split("training")
    tid = next(iter(tasks))
    task = tasks[tid]
    train_pairs = task["train"]
    test_gin, test_gout = task["test"][0]
    print(f"task: {tid}, {len(train_pairs)} train pairs")

    rng = random.Random(0)
    views = make_augmented_views(train_pairs, test_gin, N_AUGMENTED_VIEWS, rng=rng)
    # Held out for the mechanism check: one augmented view's train pair, never trained on.
    held_out_aug, held_out_train, _ = views[-1]
    held_out_pair = held_out_train[0]
    train_views = views[:-1]

    examples = [(gin, gout) for _, aug_train, _ in train_views for gin, gout in aug_train]
    print(f"{len(examples)} training examples from {len(train_views)} augmented views")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    losses = []
    for step in range(N_STEPS):
        gin, gout = examples[rng.randrange(len(examples))]
        input_ids, labels = build_example(tokenizer, gin, gout, device)
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(out.loss.item())
        if (step + 1) % 10 == 0:
            print(f"step {step + 1}/{N_STEPS} loss {out.loss.item():.4f}")

    print(f"\nloss first-10 avg: {sum(losses[:10]) / 10:.4f}  "
          f"last-10 avg: {sum(losses[-10:]) / 10:.4f}")
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10, \
        "mechanism check failed: loss did not decrease"

    model.eval()
    held_gin, held_gout = held_out_pair
    prompt, _ = format_completion(held_gin, held_gout)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=200, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
    completion = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nheld-out augmented-view generation (raw):\n{completion!r}")
    try:
        pred_grid = text_to_grid(completion)
        print(f"parsed OK, matches held-out target: {pred_grid == held_gout}")
    except ValueError as e:
        print(f"did not parse as a valid grid: {e}")

    print("\nmechanism check: loss decreased, generation produced -- pipeline is not silently broken.")


if __name__ == "__main__":
    main()
