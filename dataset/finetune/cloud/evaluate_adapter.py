"""Sanity-check a trained cloud (peft/QLoRA) adapter on held-out prompts.

Counterpart to ../mlx/evaluate_samples.py -- same held-out test.jsonl
prompts, for an apples-to-apples comparison between the two backends.
"""

from __future__ import annotations

import argparse
import json

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

DEFAULT_ADAPTER_PATH = "dataset/finetune/cloud/adapters"
DEFAULT_TEST_FILE = "dataset/finetune/data/test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sample completions from a trained cloud QLoRA adapter."
    )
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def load_test_examples(path: str, limit: int) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= limit:
                break
    return examples


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    examples = load_test_examples(args.test_file, args.num_samples)
    for i, example in enumerate(examples):
        question = example["messages"][0]["content"]
        reference = example["messages"][1]["content"]

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        completion = tokenizer.decode(
            output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True
        )

        print(f"--- sample {i} ---")
        print(f"question:   {question}")
        print(f"generated:  {completion}")
        print(f"reference:  {reference}")
        print()


if __name__ == "__main__":
    main()
