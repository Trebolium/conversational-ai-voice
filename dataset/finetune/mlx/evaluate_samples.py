"""Sanity-check a trained MLX LoRA adapter on held-out prompts.

Loads the base model with the adapter fused in at load time and generates
completions for a handful of held-out test-set prompts, printing each
alongside its reference (original) lyrics for a qualitative comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm import generate, load

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_ADAPTER_PATH = "dataset/finetune/mlx/adapters"
DEFAULT_TEST_FILE = "dataset/finetune/data/test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sample completions from a trained MLX LoRA adapter."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
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

    if not Path(args.adapter_path).exists():
        raise SystemExit(
            f"Adapter path {args.adapter_path!r} not found -- train first with "
            "`mlx_lm.lora -c dataset/finetune/mlx/lora_config.yaml`."
        )

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    examples = load_test_examples(args.test_file, args.num_samples)
    for i, example in enumerate(examples):
        question = example["messages"][0]["content"]
        reference = example["messages"][1]["content"]

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True
        )
        completion = generate(
            model, tokenizer, prompt=prompt, max_tokens=args.max_tokens
        )

        print(f"--- sample {i} ---")
        print(f"question:   {question}")
        print(f"generated:  {completion}")
        print(f"reference:  {reference}")
        print()


if __name__ == "__main__":
    main()
