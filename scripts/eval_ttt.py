"""Real TTT solve-rate measurement (known-ground-truth harness, same pattern as
scripts/eval_baseline.py). Per task: fresh LoRA adapter, fine-tune on augmented views of that
task's train pairs only, then predict the real test output via test-time augmentation +
majority vote (several random augmented views of the test input, same trained adapter, de-augment
each prediction, vote) instead of retraining per view.

Tried loading the base model once and cycling LoRA adapters (delete_adapter/add_adapter)
between tasks to avoid reloading ~1GB per task. That leaked real GPU memory that gc.collect()
+ torch.cuda.empty_cache() could not reclaim (13.84 GiB "allocated" on a 6GB card after 5
tasks, then 16.25 GiB after the same 5 tasks on a second attempt with the cleanup added -
something in peft keeps a live reference to deleted adapters). Reverted to a full per-task
model reload, which never hit a CUDA OOM in any run - simpler and it's the one that's proven
not to leak. The ~1-2s reload overhead is negligible next to per-task time (tens to hundreds
of seconds).
"""
import argparse
import gc
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.augment import make_augmented_views, random_augmentation
from src.data import load_split
from src.serialize import format_completion, text_to_grid

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
N_TRAIN_AUGS = 12
N_STEPS = 60
N_TEST_AUGS = 3
MAX_NEW_TOKENS_CAP = 700
LR = 1e-4
ADAPTER_NAME = "task"
LORA_CFG = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0, task_type="CAUSAL_LM",
)


def estimate_max_new_tokens(train_pairs, test_shape):
    """Grid text length for an R x C grid is R*(2C-1) + (R-1) chars. Generating a flat 700
    tokens regardless of actual output size was the dominant cost in the first timing run
    (617s/task avg, only ~70 tasks fit the 12h budget) - most ARC grids are much smaller than
    that. Infer expected output shape from train pairs (same-shape, or a consistent ratio),
    else fall back to the old generous cap.
    """
    th, tw = test_shape
    same_shape = all(len(gin) == len(gout) and len(gin[0]) == len(gout[0]) for gin, gout in train_pairs)
    if same_shape:
        rh, rw = th, tw
    else:
        h_ratios = {len(gout) / len(gin) for gin, gout in train_pairs}
        w_ratios = {len(gout[0]) / len(gin[0]) for gin, gout in train_pairs}
        if len(h_ratios) == 1 and len(w_ratios) == 1:
            rh = max(1, round(th * next(iter(h_ratios))))
            rw = max(1, round(tw * next(iter(w_ratios))))
        else:
            return MAX_NEW_TOKENS_CAP
    chars = rh * (2 * rw - 1) + (rh - 1)
    return min(MAX_NEW_TOKENS_CAP, chars + 30)


def build_example(tokenizer, gin, gout, device):
    prompt, target = format_completion(gin, gout)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    target_ids = tokenizer(target + tokenizer.eos_token, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([prompt_ids, target_ids]).unsqueeze(0).to(device)
    labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids]).unsqueeze(0).to(device)
    return input_ids, labels


def fresh_lora_model(tokenizer, device):
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    return get_peft_model(base, LORA_CFG, adapter_name=ADAPTER_NAME)


def train_on_task(model, tokenizer, train_pairs, test_input, device, rng):
    views = make_augmented_views(train_pairs, test_input, N_TRAIN_AUGS, rng=rng)
    examples = [(gin, gout) for _, aug_train, _ in views for gin, gout in aug_train]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(N_STEPS):
        gin, gout = examples[rng.randrange(len(examples))]
        input_ids, labels = build_example(tokenizer, gin, gout, device)
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def predict(model, tokenizer, train_pairs, test_input, device, rng):
    model.eval()
    preds = []
    for _ in range(N_TEST_AUGS):
        aug = random_augmentation(rng)
        aug_input = aug.apply(test_input)
        max_new = estimate_max_new_tokens(train_pairs, (len(aug_input), len(aug_input[0])))
        prompt, _ = format_completion(aug_input, aug_input)  # target side unused for generation
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
        try:
            grid = aug.invert(text_to_grid(text))
            preds.append(grid)
        except (ValueError, IndexError):
            continue
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = load_split(args.split)
    task_ids = list(tasks)[: args.limit]

    solved = 0
    t0_all = time.time()
    for tid in task_ids:
        t0 = time.time()
        task = tasks[tid]
        train_pairs = task["train"]
        # zlib.crc32, not the builtin hash(): Python's str hash is randomized per-process by
        # default, so "deterministic" seeding via hash(tid) silently changed every run.
        rng_seed = zlib.crc32(tid.encode())
        import random
        rng = random.Random(rng_seed)

        model = fresh_lora_model(tokenizer, device)
        print(f"  [after model load] cuda allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB")
        task_solved = True
        train_time = predict_time = 0.0
        for test_gin, test_gout in task["test"]:
            tt0 = time.time()
            train_on_task(model, tokenizer, train_pairs, test_gin, device, rng)
            train_time += time.time() - tt0
            print(f"  [after train] cuda allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB "
                  f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB")
            tp0 = time.time()
            preds = predict(model, tokenizer, train_pairs, test_gin, device, rng)
            predict_time += time.time() - tp0
            print(f"  [after predict] cuda allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB "
                  f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB")
            counts = Counter(preds)
            top2 = [g for g, _ in counts.most_common(2)]
            attempt1 = top2[0] if top2 else test_gin
            attempt2 = top2[1] if len(top2) > 1 else attempt1
            if attempt1 != test_gout and attempt2 != test_gout:
                task_solved = False
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [after cleanup] cuda allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB")

        if task_solved:
            solved += 1
        elapsed = time.time() - t0
        print(f"{tid}: solved={task_solved} time={elapsed:.1f}s "
              f"(train={train_time:.1f}s predict={predict_time:.1f}s)")

    n = len(task_ids)
    total = time.time() - t0_all
    print(f"\nsplit={args.split} n={n} solved={solved} solve_rate={solved / n:.1%} "
          f"total={total:.1f}s avg={total / n:.1f}s/task")
    budget_tasks = 12 * 3600 / (total / n)
    print(f"at this rate, 12h Kaggle budget covers ~{budget_tasks:.0f} tasks")


if __name__ == "__main__":
    main()
