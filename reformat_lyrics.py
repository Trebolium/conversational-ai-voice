import argparse
import csv
import math
import os
import re
import sys

csv.field_size_limit(sys.maxsize)

TAG_RE = re.compile(r"^\[([^\]]+)\]:?\s*$")
PAREN_WORD_RE = re.compile(r"^\(([^()\s]+)\)\s*$")
COLON_LINE_RE = re.compile(r"^[\w'&.-]+(?:\s+[\w'&.-]+){0,2}:\s*$")
PAREN_STRIP_RE = re.compile(r"\([^()]*\)")
WORD_TOKEN_RE = re.compile(r"[A-Za-z']+")

MAX_SEGMENT_LINES = 16
MIN_SEGMENT_LINES = 4

KNOWN_TYPES = [
    "pre-chorus",
    "post-chorus",
    "chorus",
    "verse",
    "intro",
    "outro",
    "bridge",
    "hook",
    "refrain",
    "interlude",
    "breakdown",
    "instrumental",
    "spoken",
]


def normalize_segment_name(tag_content: str) -> str:
    text = tag_content.strip().lower()
    text = text.split(":", 1)[0].strip()
    for t in KNOWN_TYPES:
        if text.startswith(t):
            return t
    return "unk"


MIN_VOCAB_WORD_LEN = 4
MIN_VOCAB_SONG_COUNT = 2


def collect_vocabulary(rows, lyrics_idx):
    song_counts = {}
    for row in rows:
        words_in_song = set()
        for line in row[lyrics_idx].split("\n"):
            m = TAG_RE.match(line.strip())
            if m:
                for w in WORD_TOKEN_RE.findall(m.group(1)):
                    words_in_song.add(w.lower())
        for w in words_in_song:
            song_counts[w] = song_counts.get(w, 0) + 1

    return {
        w
        for w, count in song_counts.items()
        if len(w) >= MIN_VOCAB_WORD_LEN and count >= MIN_VOCAB_SONG_COUNT
    }


def load_existing_vocabulary(path):
    """Read a previously written vocabulary file. Missing file means an empty set."""
    try:
        with open(path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def write_vocabulary(path, words):
    """Write the vocabulary via a temp file so a crash can't truncate the accumulated list."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for word in sorted(words):
            f.write(word + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def is_no_alnum_line(stripped: str) -> bool:
    return stripped != "" and not re.search(r"[A-Za-z0-9]", stripped)


def raw_segments(lyrics: str, vocab: set):
    lines = lyrics.split("\n")
    segments = []
    current_name = None
    current_lines = []

    def flush(name):
        nonlocal current_lines
        if current_lines:
            segments.append([name if name is not None else "unk", current_lines])
        current_lines = []

    for line in lines:
        stripped = line.strip()

        m = TAG_RE.match(stripped)
        if m:
            flush(current_name)
            current_name = normalize_segment_name(m.group(1))
            continue

        if stripped == "":
            if current_lines:
                flush(current_name)
                current_name = None
            continue

        m2 = PAREN_WORD_RE.match(stripped)
        if m2:
            flush(current_name)
            current_name = normalize_segment_name(m2.group(1))
            continue

        if COLON_LINE_RE.match(stripped):
            flush(current_name)
            current_name = None
            continue

        if is_no_alnum_line(stripped):
            flush(current_name)
            current_name = None
            continue

        without_parens = PAREN_STRIP_RE.sub(" ", stripped)
        tokens = without_parens.split()
        if 1 <= len(tokens) <= 3:
            matched = None
            for t in tokens:
                tw = re.sub(r"[^A-Za-z']", "", t).lower()
                if tw and tw in vocab:
                    matched = tw
                    break
            if matched:
                flush(current_name)
                mapped = normalize_segment_name(matched)
                current_name = mapped if mapped != "unk" else matched
                continue

        current_lines.append(line)

    flush(current_name)

    result = []
    for name, seg_lines in segments:
        while seg_lines and seg_lines[0].strip() == "":
            seg_lines = seg_lines[1:]
        while seg_lines and seg_lines[-1].strip() == "":
            seg_lines = seg_lines[:-1]
        if seg_lines:
            result.append((name, seg_lines))
    return result


def subdivide_long_segments(segments):
    result = []
    for name, lines in segments:
        n = len(lines)
        if n <= MAX_SEGMENT_LINES:
            result.append([name, lines])
            continue
        num_chunks = math.ceil(n / MAX_SEGMENT_LINES)
        for i in range(num_chunks):
            chunk = lines[i * MAX_SEGMENT_LINES : (i + 1) * MAX_SEGMENT_LINES]
            result.append([name, chunk])
    return result


def merge_name(name_a, name_b):
    parts = []
    for n in (name_a, name_b):
        if n not in parts:
            parts.append(n)
    return "+".join(parts)


def merge_short_segments(segments):
    segs = [list(s) for s in segments]

    while len(segs) > 1:
        lengths = [len(s[1]) for s in segs]
        min_len = min(lengths)
        if min_len >= MIN_SEGMENT_LINES:
            break

        idx = lengths.index(min_len)
        if idx == 0:
            neighbor = 1
        elif idx == len(segs) - 1:
            neighbor = idx - 1
        else:
            left_len = len(segs[idx - 1][1])
            right_len = len(segs[idx + 1][1])
            neighbor = idx - 1 if left_len <= right_len else idx + 1

        i, j = sorted([idx, neighbor])
        merged_name = merge_name(segs[i][0], segs[j][0])
        merged_lines = segs[i][1] + segs[j][1]
        segs[i] = [merged_name, merged_lines]
        del segs[j]

    return [(name, lines) for name, lines in segs]


def split_lyrics(lyrics: str, vocab: set):
    segments = raw_segments(lyrics, vocab)
    segments = subdivide_long_segments(segments)
    segments = merge_short_segments(segments)

    result = []
    for name, lines in segments:
        text = "\n".join(lines).strip("\n").strip()
        if text:
            result.append((name, text))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Split a song-lyrics CSV's 'lyrics' column into one row per structural segment."
    )
    parser.add_argument("input_csv", help="Path to the input CSV (must have a 'lyrics' column)")
    parser.add_argument("output_csv", help="Path to write the reformatted CSV")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N data rows (for testing)",
    )
    parser.add_argument(
        "--vocab-out",
        default="/dataset/finetune/data/collected_segment_names.txt",
        help=(
            "Path to the collected bracket-tag vocabulary. Words found in this run are "
            "merged into whatever the file already contains, and the merged result is "
            "what segmentation uses"
        ),
    )
    parser.add_argument(
        "--reset-vocab",
        action="store_true",
        help=(
            "Discard the existing vocabulary file instead of merging into it, so "
            "segmentation uses only the words collected from this input"
        ),
    )
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        lyrics_idx = header.index("lyrics")
        rows = list(reader)

    if args.limit is not None:
        rows = rows[: args.limit]

    vocab = collect_vocabulary(rows, lyrics_idx)

    # Segmentation uses the accumulated vocabulary, so tags learned from previously
    # processed corpora are recognised here too. This makes the output depend on the
    # vocabulary file's history as well as on the input; --reset-vocab restricts
    # segmentation to this run's own words.
    previous_vocab = set() if args.reset_vocab else load_existing_vocabulary(args.vocab_out)
    merged_vocab = previous_vocab | vocab
    write_vocabulary(args.vocab_out, merged_vocab)

    out_header = (
        header[: lyrics_idx + 1]
        + ["segment index", "segment name"]
        + header[lyrics_idx + 1 :]
    )

    out_rows = []
    for row in rows:
        lyrics = row[lyrics_idx]
        segments = split_lyrics(lyrics, merged_vocab)
        for i, (name, text) in enumerate(segments, start=1):
            new_row = (
                row[:lyrics_idx]
                + [text]
                + [str(i), name]
                + row[lyrics_idx + 1 :]
            )
            out_rows.append(new_row)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(out_rows)

    print(f"input rows: {len(rows)}")
    print(f"output rows: {len(out_rows)}")
    print(f"vocabulary collected from this input: {len(vocab)}")
    print(
        f"vocabulary used for segmentation: {len(merged_vocab)} "
        f"(+{len(merged_vocab) - len(previous_vocab)} new in {args.vocab_out})"
    )


if __name__ == "__main__":
    main()
