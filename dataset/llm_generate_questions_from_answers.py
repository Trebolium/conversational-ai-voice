"""Generate one plausible question for each lyric segment in a CSV.

For every row, the `lyrics` text is treated as an answer one person gave to
another, and a single-turn OpenAI call invents a question that answer would
be appropriate for. The model itself randomly picks which of three personas
-- a friend, a student, or a stern employer -- is asking, so the style of
question varies row to row without us having to choose client-side. Each
row is its own independent API session (no back-and-forth). The result is
written back into the "question 1" column on the same CSV, in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT_TEMPLATE = (
    "Make up a question that one of two people might ask, where the "
    "information within the other person's response (disregard the "
    "line-breaking, rhythmic and rhyming structures) would be an "
    "appropriate answer: {lyrics}. First, privately and randomly choose "
    "one of three askers: a friend, a student, or a stern employer -- do "
    "not reveal which one you picked. Then write that single question in "
    "a voice and tone appropriate to that asker, focused on one specific "
    "element or theme within the response."
)

SYSTEM_PROMPT = (
    "Respond with a JSON object with exactly one key, \"question\", "
    "mapped to the single question as a plain string. Output only the "
    "JSON object -- no labels, no markdown formatting, no surrounding "
    "text, no mention of which asker you picked, and no example answers "
    "or responses within the question text."
)

QUESTION_COLUMN = "question 1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one character-specific question (friend, "
        "student, or stern employer, chosen at random by the model) for "
        "each lyric segment in a CSV, via a single independent OpenAI API "
        "call per row."
    )
    parser.add_argument("csv_path", help="Path to the CSV file to process (overwritten in place).")
    parser.add_argument("--lyrics-column", default="lyrics")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the API base URL to use an OpenAI-compatible provider "
        "other than OpenAI, e.g. https://api.deepseek.com, "
        "https://generativelanguage.googleapis.com/v1beta/openai/ (Gemini), "
        "or http://localhost:11434/v1 (Ollama). Default: OpenAI's own endpoint.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Name of the environment variable holding the API key for "
        "--base-url's provider (default: OPENAI_API_KEY). For Ollama, this "
        "can point at any variable set to a placeholder value, since Ollama "
        "doesn't check it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows starting at --start_entry (for testing). Default: all remaining rows.",
    )
    parser.add_argument(
        "--start_entry",
        type=int,
        default=0,
        help="Row index (0-based, as printed in [i] progress lines) to resume from, skipping everything before it.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of rows to send to the API in parallel (these are I/O-bound "
        "requests, not CPU work). Lower this if you start hitting rate limits.",
    )
    return parser.parse_args()


def build_prompt(lyrics: str) -> str:
    return PROMPT_TEMPLATE.format(lyrics=lyrics)


_LABEL_PREFIX_RE = re.compile(r"^\**question\**\s*:\s*", flags=re.IGNORECASE)


def clean_question(text: str) -> str:
    """Strip common formatting artifacts models add despite instructions
    (a leading 'Question:' label, wrapping quotes, or a trailing invented
    answer/response section)."""
    text = text.strip()

    # Drop everything from an invented answer/response section onward.
    text = re.split(r"\n\s*\**(?:response|answer)\**\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]

    text = _LABEL_PREFIX_RE.sub("", text.strip())

    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1]

    return text.strip()


def generate_question(
    client, model: str, lyrics: str, max_retries: int = 5, initial_backoff: float = 5.0
) -> str:
    """Return a single question, in a persona voice the model picked itself.

    Retries with exponential backoff on rate-limit (429) errors -- free-tier
    APIs enforce low requests-per-minute caps, and burning through retries
    immediately (rather than waiting out the window) just wastes the quota
    on rows that will fail anyway.
    """
    delay = initial_backoff
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(lyrics)},
                ],
            )
            data = json.loads(response.choices[0].message.content)
            return clean_question(str(data["question"]))
        except Exception as exc:
            is_rate_limit = getattr(exc, "status_code", None) == 429
            if is_rate_limit and attempt < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise


def write_csv(csv_path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Write atomically: a kill/crash mid-write must never leave csv_path
    truncated or partially overwritten -- write to a temp file in the same
    directory, flush it to disk, then atomically swap it into place."""
    tmp_path = f"{csv_path}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, csv_path)


def main() -> None:
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set in the environment.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if QUESTION_COLUMN not in fieldnames:
        fieldnames.append(QUESTION_COLUMN)
    for row in rows:
        row.setdefault(QUESTION_COLUMN, "")

    to_process = rows[args.start_entry :]
    if args.limit is not None:
        to_process = to_process[: args.limit]

    # Filter out rows that don't need an API call before scheduling any work.
    tasks: list[tuple[int, dict]] = []
    for i, row in enumerate(to_process, start=args.start_entry):
        lyrics = (row.get(args.lyrics_column) or "").strip()
        already_done = bool((row.get(QUESTION_COLUMN) or "").strip())

        if not lyrics:
            print(f"[{i}] skipping (empty {args.lyrics_column!r})")
            continue
        if already_done:
            print(f"[{i}] skipping (already has a question)")
            continue
        tasks.append((i, row))

    # Requests are I/O-bound (waiting on the API), so a thread pool gives real
    # speedup here. The OpenAI client is safe to share across threads. Writes
    # are serialized with a lock -- progress is still saved after every row
    # completes (not just at the end), same as the sequential version.
    write_lock = threading.Lock()

    def save() -> None:
        with write_lock:
            write_csv(args.csv_path, fieldnames, rows)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                generate_question,
                client,
                args.model,
                (row.get(args.lyrics_column) or "").strip(),
            ): (i, row)
            for i, row in tasks
        }

        for future in as_completed(futures):
            i, row = futures[future]
            title = row.get("title", "")
            try:
                question = future.result()
            except Exception as exc:
                print(f"[{i}] {title!r}: API error, leaving blank ({exc})")
                row[QUESTION_COLUMN] = ""
                save()
                continue

            row[QUESTION_COLUMN] = question
            print(f"[{i}] {title!r}: {question}")
            save()


if __name__ == "__main__":
    main()
