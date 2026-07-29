"""Randomly sample rows from a song-lyrics CSV, filtered by language.

Streams the source CSV in a single pass and reservoir-samples rows whose
`language` column matches a given value, so the whole file never has to be
held in memory. Useful for the multi-gigabyte song_lyrics.csv dataset.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a song-lyrics CSV by language and write out a random sample."
    )
    parser.add_argument(
        "--source",
        default="dataset/csvs/song_lyrics.csv",
        help="Path to the source CSV to sample from.",
    )
    parser.add_argument(
        "--output",
        default="dataset/csvs/song_lyrics_sample.csv",
        help="Path to write the sampled CSV to.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Value to match against the --language-column (e.g. 'en').",
    )
    parser.add_argument(
        "--language-column",
        default="language",
        help="Name of the column to filter on.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of rows to randomly sample from the matching rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: nondeterministic).",
    )
    return parser.parse_args()


def reservoir_sample(
    source: str, language: str, language_column: str, sample_size: int
) -> tuple[list[str], list[list[str]]]:
    """Single-pass reservoir sample of rows matching language_column == language."""
    csv.field_size_limit(sys.maxsize)

    reservoir: list[list[str]] = []
    seen = 0

    with open(source, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_idx = header.index(language_column)

        for row in reader:
            if len(row) <= col_idx or row[col_idx] != language:
                continue
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = random.randint(0, seen - 1)
                if j < sample_size:
                    reservoir[j] = row

    print(f"Total matching rows seen: {seen}")
    print(f"Sampled: {len(reservoir)}")
    return header, reservoir


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    header, rows = reservoir_sample(
        args.source, args.language, args.language_column, args.sample_size
    )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
