"""Score a trained adapter on the two things that actually matter for this
task: did it adopt the lyric *style*, and does the content genuinely answer
the question -- plus a leakage check that it isn't just regurgitating songs
it memorized during training.

Style is scored reference-free, against the distribution of the real lyric
targets in train.jsonl, because for style transfer there is no single correct
output. Content is scored by an LLM judge that never sees the reference.

Run the baselines first, otherwise the numbers mean nothing:

    # floor: base model, no instruction
    python dataset/finetune/mlx/score_samples.py --no-adapter
    # the bar to beat: base model, style asked for in the prompt
    python dataset/finetune/mlx/score_samples.py --no-adapter --style-prompt
    # the fine-tune
    python dataset/finetune/mlx/score_samples.py

If the adapter doesn't beat the --style-prompt baseline, the fine-tune isn't
earning its keep.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_ADAPTER_PATH = "dataset/finetune/mlx/adapters"
DEFAULT_TEST_FILE = "dataset/finetune/data/test.jsonl"
DEFAULT_TRAIN_FILE = "dataset/finetune/data/train.jsonl"

STYLE_PROMPT = "Answer the user's question in the style of song lyrics."

PROSE_TELLS = re.compile(
    r"(as an ai|i'm an ai|i am an ai|i don't have|i do not have|language model"
    r"|^\s*[-*]\s|^\s*\d+\.\s|```|\*\*)",
    re.I | re.M,
)


# --------------------------------------------------------------------------
# A. Style features
# --------------------------------------------------------------------------
def _rhymes(a: str, b: str) -> bool:
    """Crude rhyme test: shared 3-char (or 2-char vowel-containing) suffix."""
    a, b = re.sub(r"[^a-z]", "", a.lower()), re.sub(r"[^a-z]", "", b.lower())
    if not a or not b or a == b:
        return False
    for n in (3, 2):
        if len(a) >= n and len(b) >= n and a[-n:] == b[-n:]:
            return bool(re.search(r"[aeiou]", a[-n:]))
    return False


def style_features(text: str) -> dict[str, float]:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return dict(n_lines=0, words_per_line=0, end_punct=1.0, cap_start=0.0,
                    rhyme_rate=0.0, prose_tells=1.0)

    finals = [l.split()[-1] for l in lines if l.split()]
    pairs = list(zip(finals, finals[1:])) + list(zip(finals, finals[2:]))

    return dict(
        n_lines=len(lines),
        words_per_line=statistics.mean(len(l.split()) for l in lines),
        end_punct=sum(l[-1] in ".,;:" for l in lines) / len(lines),
        cap_start=sum(l[0].isupper() for l in lines) / len(lines),
        rhyme_rate=(sum(_rhymes(a, b) for a, b in pairs) / len(pairs)) if pairs else 0.0,
        prose_tells=1.0 if PROSE_TELLS.search(text) else 0.0,
    )


def reference_bands(targets: list[str]) -> dict[str, tuple[float, float]]:
    """[p5, p95] band per feature over the real lyric targets."""
    feats = [style_features(t) for t in targets]
    return {
        k: (float(np.percentile([f[k] for f in feats], 5)),
            float(np.percentile([f[k] for f in feats], 95)))
        for k in feats[0]
    }


def style_score(text: str, bands: dict[str, tuple[float, float]]) -> tuple[float, dict]:
    feats = style_features(text)
    inside = {k: bands[k][0] <= v <= bands[k][1] for k, v in feats.items()}
    return sum(inside.values()) / len(inside), feats


# --------------------------------------------------------------------------
# C. Leakage against the training targets
# --------------------------------------------------------------------------
def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z']+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def leakage(text: str, train_ngrams: set[tuple[str, ...]], n: int = 5) -> float:
    """Share of the generation's n-grams that appear verbatim in training data."""
    g = _ngrams(text, n)
    return len(g & train_ngrams) / len(g) if g else 0.0


# --------------------------------------------------------------------------
# B. Content judge
# --------------------------------------------------------------------------
JUDGE_TEMPLATE = """You are grading whether a response genuinely answers a question.

The response is DELIBERATELY written as song lyrics. Do not penalise it for
being lyrical, fragmentary, non-literal, unpunctuated, or unlike normal prose.
Judge ONLY whether a reader could extract a real, on-topic answer to what was
asked.

Question: {question}

Response:
{response}

Reply with JSON only: {{"score": <1-5>, "why": "<one short sentence>"}}
where 5 = fully and specifically answers the question, 3 = on-topic but vague
or partial, 1 = does not engage with the question at all."""


def judge_content(question: str, response: str) -> tuple[int | None, str]:
    from llm import Message, generate_response  # your existing Claude/Ollama backend

    prompt = JUDGE_TEMPLATE.format(question=question, response=response)
    raw = generate_response(
        [Message(role="user", content=prompt)],
        system="You are a strict, concise grading assistant. Reply with JSON only.",
    )
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, f"unparseable: {raw[:80]}"
    try:
        out = json.loads(m.group())
        return int(out["score"]), str(out.get("why", ""))
    except Exception as exc:  # noqa: BLE001
        return None, f"unparseable ({exc}): {raw[:80]}"


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    p.add_argument("--no-adapter", action="store_true", help="Baseline: base model only.")
    p.add_argument("--style-prompt", action="store_true",
                   help="Add the style instruction as a system prompt.")
    p.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    p.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--judge", action="store_true", help="Run the LLM content judge.")
    p.add_argument("--dump", default=None, help="Write per-sample JSONL here.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from mlx_lm import generate, load

    train = [json.loads(l) for l in open(args.train_file, encoding="utf-8")]
    targets = sorted({r["messages"][1]["content"] for r in train})
    bands = reference_bands(targets)
    train_ngrams = set().union(*(_ngrams(t, 5) for t in targets))

    adapter = None if args.no_adapter else args.adapter_path
    if adapter and not Path(adapter).exists():
        raise SystemExit(f"Adapter path {adapter!r} not found -- train first.")
    model, tok = load(args.model, adapter_path=adapter)

    test = [json.loads(l) for l in open(args.test_file, encoding="utf-8")][:args.n]
    rows = []
    for ex in test:
        question = ex["messages"][0]["content"]
        msgs = ([{"role": "system", "content": STYLE_PROMPT}] if args.style_prompt else [])
        msgs.append({"role": "user", "content": question})
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        out = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, verbose=False)

        s, feats = style_score(out, bands)
        row = dict(question=question, generation=out, style=s,
                   leakage=leakage(out, train_ngrams), **feats)
        if args.judge:
            row["content"], row["why"] = judge_content(question, out)
        rows.append(row)

    # Reference ceiling: score the real lyrics with the same battery.
    ceiling = statistics.mean(style_score(t, bands)[0] for t in targets)

    label = ("base" if args.no_adapter else "adapter") + ("+styleprompt" if args.style_prompt else "")
    print(f"\n=== {label}  (n={len(rows)}) ===")
    print(f"style score      {statistics.mean(r['style'] for r in rows):.2f}"
          f"   (real lyrics score {ceiling:.2f})")
    print(f"leakage (5-gram) {statistics.mean(r['leakage'] for r in rows):.3f}"
          f"   max {max(r['leakage'] for r in rows):.3f}")
    if args.judge:
        got = [r["content"] for r in rows if r["content"] is not None]
        print(f"content 1-5      {statistics.mean(got):.2f}   (judged {len(got)}/{len(rows)})")
    print("\nper-feature means:")
    for k in ("n_lines", "words_per_line", "end_punct", "cap_start", "rhyme_rate", "prose_tells"):
        lo, hi = bands[k]
        print(f"  {k:16s} {statistics.mean(r[k] for r in rows):7.2f}   target band [{lo:.2f}, {hi:.2f}]")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(rows)} samples to {args.dump}")


if __name__ == "__main__":
    main()
