"""Diagnostic: print every TTT candidate prediction against the real answer for one task,
instead of just pass/fail, to see whether failures are near-misses, malformed generations,
or something structurally wrong.
"""
import random
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.augment import random_augmentation
from src.data import load_split
from src.serialize import format_completion, text_to_grid, grid_to_text
from scripts.eval_ttt import build_model, train_on_task, estimate_max_new_tokens
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "00576224"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = load_split("training")
    task = tasks[task_id]
    train_pairs = task["train"]
    test_gin, test_gout = task["test"][0]
    print(f"task {task_id}: {len(train_pairs)} train pairs, test input {len(test_gin)}x{len(test_gin[0])}, "
          f"true output {len(test_gout)}x{len(test_gout[0])}")
    print(f"true output:\n{grid_to_text(test_gout)}\n")

    rng = random.Random(zlib.crc32(task_id.encode()))
    model = build_model(tokenizer, device)
    train_on_task(model, tokenizer, train_pairs, test_gin, device, rng)

    model.eval()
    for i in range(5):
        aug = random_augmentation(rng)
        aug_input = aug.apply(test_gin)
        max_new = estimate_max_new_tokens(train_pairs, (len(aug_input), len(aug_input[0])))
        prompt, _ = format_completion(aug_input, aug_input)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        raw = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"--- view {i} (aug={aug.geom_name}) max_new_tokens={max_new} ---")
        print(f"raw generation:\n{raw!r}\n")
        try:
            grid = aug.invert(text_to_grid(raw))
            print(f"parsed+inverted:\n{grid_to_text(grid)}")
            print(f"matches true output: {grid == test_gout}\n")
        except (ValueError, IndexError) as e:
            print(f"parse/invert failed: {e}\n")


if __name__ == "__main__":
    main()
