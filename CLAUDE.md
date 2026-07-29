# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Two unrelated pieces of work sharing one repo, **not wired together**:

1. **Voice assistant** (repo root: `main.py`, `asr.py`, `llm.py`, `tts.py`, `rec.py`, `metrics.py`) — a
   push-to-talk loop: record mic audio → transcribe locally with `faster-whisper` → send conversation to
   Claude or Ollama → stream-synthesize the reply with `pocket_tts`. Every stage is latency/resource
   instrumented.
2. **Dataset + fine-tuning pipeline** (`dataset/`) — turns a large song-lyrics CSV into chat-format JSONL
   and LoRA fine-tunes Qwen2.5-3B-Instruct via two interchangeable backends (MLX for Apple Silicon,
   QLoRA for CUDA). The resulting adapter is **not** loaded by the voice assistant.

`README.md` is the authoritative, exhaustively detailed reference for both halves (every CLI flag,
default, and gotcha is documented there) — consult it before guessing at behavior instead of re-deriving
it from source. This file only covers what's needed to start working productively and things that are
easy to get wrong.

## Commands

This is a **flat script layout, not an installable package** (`[tool.uv] package = false`). There is no
`src/`, no console-script entrypoints, no test suite, and no configured linter. Run everything with
`uv run <script.py>` from the repository root.

```bash
uv sync                                    # install deps (uv-managed, uv.lock present)

# Voice assistant
export ANTHROPIC_API_KEY=sk-ant-...        # omit to fall back to Ollama
uv run main.py

# Dataset pipeline, in order (run from repo root — paths are relative to it)
uv run dataset/sample_lyrics_by_language.py --source ... --output ... --seed 0
uv run reformat_lyrics.py <input_csv> <output_csv>
export OPENAI_API_KEY=...
uv run dataset/llm_generate_questions_from_answers.py <csv_path> --limit N
uv run dataset/finetune/prepare_data.py

# Fine-tuning
mlx_lm.lora -c dataset/finetune/mlx/lora_config.yaml              # Apple Silicon only
uv run dataset/finetune/mlx/evaluate_samples.py

pip install -r dataset/finetune/cloud/requirements-cloud.txt      # CUDA only, separate from pyproject.toml
python dataset/finetune/cloud/train_qlora.py
python dataset/finetune/cloud/evaluate_adapter.py
```

## Architecture notes (voice assistant)

- **Backend selection is environment-driven, not a CLI flag.** `llm.py:active_backend()` picks Claude if
  `ANTHROPIC_API_KEY` is set, otherwise Ollama. `CLAUDE_MODEL`, `OLLAMA_MODEL`, `OLLAMA_HOST` override
  defaults. This is easy to miss when reading `main.py`'s argument parser, which has no backend flag at
  all.
- **Pipeline shape**: `main.py:run_turn()` wires `rec.record_audio` → `asr.ASR.transcribe` →
  `llm.generate_response` → `tts.TextToSpeech.synthesize_to_file`, all under one
  `metrics.measure("pipeline.turn_total")` block that deliberately starts *after* recording so user
  think-time isn't counted as pipeline cost.
- **`ASR` and `TextToSpeech` are constructed once** in `main.py:main()` and reused across turns — don't
  reconstruct them per turn, that defeats the point of avoiding model reload cost.
- **Conversation state is in-memory only** (`list[Message]` in `main.py`), no persistence across
  processes. `audio_test.wav`, `transcript.txt`, `response.txt`, `output.wav` are overwritten (not
  appended) every turn and are **not gitignored** — leftover copies exist at repo root from prior runs.
- **`metrics.measure()`** is the shared instrumentation primitive used across all stages (ASR, LLM, TTS,
  turn total): a context manager that samples CPU/mem/GPU on a background thread every 50ms and appends
  a `StageMetrics` to the module-level `default_recorder`. GPU probing order is NVML → `torch.cuda` →
  `torch.mps` → none, and is purely additive — its absence never raises. Follow this pattern (`with
  measure("stage.name"): ...`) if adding new pipeline stages.

## Architecture notes (dataset pipeline)

- **Four-stage pipeline, each stage's output CSV feeds the next**: `sample_lyrics_by_language.py` →
  `reformat_lyrics.py` → `llm_generate_questions_from_answers.py` (mutates its input CSV **in place**,
  no `--output` flag) → `dataset/finetune/prepare_data.py` (splits at the row level so one song can't
  straddle train/test).
- **`reformat_lyrics.py` has a stateful vocabulary file** (`collected_segment_names.txt` by default) that
  *accumulates* across runs and *drives segmentation* on every subsequent run — output is a function of
  input **and** vocabulary history, not input alone. Use `--reset-vocab` for a reproducible, input-only
  run.
- **The two fine-tuning backends are intentionally separate dependency stacks.** MLX deps live in the
  main `pyproject.toml` (Apple Silicon only); CUDA deps live in
  `dataset/finetune/cloud/requirements-cloud.txt` and are installed separately, since `bitsandbytes` has
  no CPU/MPS support and would break the main env on non-CUDA machines. Both backends train against the
  same `dataset/finetune/data/{train,valid,test}.jsonl` and both eval scripts read the same
  `test.jsonl` for apples-to-apples comparison.
- **Generated artifacts are gitignored and don't exist on a fresh clone**: `dataset/finetune/data/`,
  `dataset/finetune/mlx/adapters/`, `dataset/finetune/cloud/adapters/`, and everything under
  `dataset/csvs/`. Run `prepare_data.py` before training, and train before evaluating — both eval
  scripts fail (one with a friendly message, one with a raw stack trace) if the adapter doesn't exist
  yet.
