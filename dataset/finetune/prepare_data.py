"""Turn song_lyrics_50_reformatted.csv into chat-format JSONL training data.

For each CSV row, the "question 1" column becomes a single (instruction,
response) example: the question is the user turn, and the row's `lyrics`
is the assistant turn. The "question 2" and "question 3" columns (if
present) are leftover from an earlier version of the generation script and
are ignored. Rows are split into train/valid/test. Output is written as
JSONL in the {"messages": [...]} chat format consumed directly by both the
mlx_lm and trl.SFTTrainer training backends.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

DEFAULT_CSV = "dataset/csvs/song_lyrics_50_reformatted.csv"
DEFAULT_OUTPUT_DIR = "dataset/finetune/data"
QUESTION_COLUMN = "question 1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/valid/test JSONL chat data from the song "
        "lyrics CSV's question 1/2/3 and lyrics columns."
    )
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N rows of the CSV (for quick testing).",
    )
    return parser.parse_args()


def load_rows(csv_path: str, limit: int | None) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows if limit is None else rows[:limit]


def row_to_examples(row: dict) -> list[dict]:
    lyrics = (row.get("lyrics") or "").strip()
    question = (row.get(QUESTION_COLUMN) or "").strip()
    if not question or not lyrics:
        return []
    return [
        {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": lyrics},
            ]
        }
    ]


def split_rows(
    rows: list[dict], train_frac: float, valid_frac: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    train_rows = shuffled[:n_train]
    valid_rows = shuffled[n_train : n_train + n_valid]
    test_rows = shuffled[n_train + n_valid :]
    return train_rows, valid_rows, test_rows


def write_jsonl(rows: list[dict], path: Path) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            for example in row_to_examples(row):
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    args = parse_args()

    rows = load_rows(args.csv_path, args.limit)
    train_rows, valid_rows, test_rows = split_rows(
        rows, args.train_frac, args.valid_frac, args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.jsonl": train_rows,
        "valid.jsonl": valid_rows,
        "test.jsonl": test_rows,
    }
    for filename, split_rows_ in splits.items():
        n_examples = write_jsonl(split_rows_, output_dir / filename)
        print(
            f"{filename}: {len(split_rows_)} rows -> {n_examples} examples"
        )


if __name__ == "__main__":
    main()
