"""Randomly keep only X rows from a CSV, discarding the rest.

Streams the source CSV in a single pass and reservoir-samples X rows, so
the whole file never has to be held in memory. Unlike
sample_lyrics_by_language.py, this does not filter on any column -- it
just thins an already-prepared CSV (e.g. a segmented lyrics file) down to
a smaller random subset.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly keep only X rows from a CSV and write the result to dataset/."
    )
    parser.add_argument("source", help="Path to the source CSV to subsample.")
    parser.add_argument("count", type=int, help="Number of rows to randomly keep.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the subsampled CSV to. Default: "
        "dataset/<source stem>_sample<count>.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: nondeterministic).",
    )
    return parser.parse_args()


def reservoir_sample(source: str, count: int) -> tuple[list[str], list[list[str]]]:
    """Single-pass reservoir sample of `count` rows from the source CSV."""
    csv.field_size_limit(sys.maxsize)

    reservoir: list[list[str]] = []
    seen = 0

    with open(source, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            seen += 1
            if len(reservoir) < count:
                reservoir.append(row)
            else:
                j = random.randint(0, seen - 1)
                if j < count:
                    reservoir[j] = row

    print(f"Total rows seen: {seen}")
    print(f"Kept: {len(reservoir)}")
    return header, reservoir


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output = args.output
    if output is None:
        output = f"dataset/{Path(args.source).stem}_sample{args.count}.csv"

    header, rows = reservoir_sample(args.source, args.count)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
