# conversational-ai-voice

This repository contains two related pieces of work.

The first is a **local voice-conversation assistant**: a push-to-talk turn loop that records from the
microphone, transcribes locally with Whisper (via `faster-whisper`), sends the running conversation to
an LLM (Anthropic Claude, or a local Ollama model), and speaks the reply back with streaming
text-to-speech so playback begins before synthesis finishes. Every stage is wrapped in latency and
resource instrumentation, and an aggregated per-stage table is printed when the session ends.

The second is a **dataset and fine-tuning pipeline**. It samples from a large song-lyrics CSV, splits
each song into structural segments (verse, chorus, bridge, …), uses an LLM to invent plausible
questions that each lyric segment could be an answer to, and emits chat-format JSONL splits. Those
splits are used to LoRA fine-tune Qwen2.5-3B-Instruct through two interchangeable backends: a local
MLX run on Apple Silicon (a cheap sanity-check of the data and pipeline) and a cloud QLoRA run on a
CUDA GPU (intended to produce the final-quality adapter). Both backends consume the same `test.jsonl`
prompts so their outputs can be compared directly.

The two halves share a repository but are not wired together: the fine-tuned adapter is not currently
loaded by the voice assistant.

---

## Requirements

- **Python 3.11+** (`.python-version` pins `3.11`; `pyproject.toml` sets `requires-python = ">=3.11"`).
- **[uv](https://docs.astral.sh/uv/)** — the project is uv-managed (`uv_build` build backend, with a
  `uv.lock` present).
- **Working audio hardware and PortAudio** for the voice assistant. `sounddevice` needs a real
  microphone and output device; the assistant is not headless-server friendly.
- **An LLM backend for the assistant**: either an `ANTHROPIC_API_KEY` plus network access, or a local
  Ollama daemon with the target model pulled.
- **An OpenAI-compatible API key** for the question-generation step of the dataset pipeline.

Platform split for fine-tuning:

| Path | Hardware | Stack |
|---|---|---|
| `dataset/finetune/mlx/` | Apple Silicon (M-series) Mac | `mlx-lm` |
| `dataset/finetune/cloud/` | NVIDIA CUDA GPU | `torch` + `transformers` + `peft` + `bitsandbytes` + `trl` |

MLX will not run on non-Apple-Silicon hardware, and `bitsandbytes` has no CPU or MPS support — that is
precisely why there are two paths.

---

## Installation

```bash
uv sync
```

This installs everything declared in `pyproject.toml`: `anthropic`, `faster-whisper`, `mlx-lm`,
`ollama`, `openai`, `pocket-tts`, `psutil`, `sounddevice`, `soundfile`.

Commands below are written as `uv run <script>`. Plain `python <script>` works equivalently inside an
activated virtualenv.

The **cloud QLoRA path has its own dependency file**, deliberately kept out of `pyproject.toml` so the
CUDA stack does not pollute the Apple-Silicon-oriented main environment. Install it only on the CUDA
machine:

```bash
pip install -r dataset/finetune/cloud/requirements-cloud.txt
```

Its contents (lower bounds only): `torch>=2.4`, `transformers>=5.14`, `trl>=1.9`, `peft>=0.19`,
`accelerate>=1.14`, `bitsandbytes>=0.49`, `datasets>=5.0`.

---

## Quick start

```bash
# Option A: Claude backend
export ANTHROPIC_API_KEY=sk-ant-...
uv run main.py

# Option B: local Ollama backend (no ANTHROPIC_API_KEY set)
ollama pull qwen2.5:3b
uv run main.py
```

The first run downloads the Whisper weights (CTranslate2 format) and loads the `pocket_tts` model, so
startup is slower than subsequent runs.

Once loaded, the assistant prints `Loaded. Let's talk.` and then repeatedly prompts:

```
Press Enter to record your message (or type 'q' to quit):
```

Press Enter to start recording, press Enter again to stop — there is no voice-activity detection, so
recording is fully manual push-to-talk. Type `q`, `quit`, or `exit` to end the session and print the
metrics summary.

If no microphone is detected (for example on a headless server or cloud VM with no audio hardware), the
assistant automatically falls back to a typed-text prompt (`You (type 'q' to quit):`) instead of
recording, and skips loading the ASR model entirely. Likewise, if no audio output device is detected,
synthesized speech is still written to `--tts-output` but playback is skipped. See
[Audio device fallback](#audio-device-fallback) below.

---

## The voice assistant

### Usage

```bash
uv run main.py [--audio-file PATH] [--transcript-file PATH] [--response-file PATH] \
               [--tts-output PATH] [--model MODEL] [--device DEVICE] \
               [--compute-type TYPE] [--batch-size N] [--voice VOICE]
```

| Flag | Default | Purpose |
|---|---|---|
| `--audio-file` | `audio_test.wav` | Path the recorded mic input is written to |
| `--transcript-file` | `transcript.txt` | Path the ASR transcript is written to |
| `--response-file` | `response.txt` | Path the LLM text reply is written to |
| `--tts-output` | `output.wav` | Path the synthesized speech is written to |
| `--model` | `small` | Whisper model size for `faster_whisper.WhisperModel` |
| `--device` | `cpu` | Whisper inference device (`cpu`, `cuda`, …) |
| `--compute-type` | `int8` | Whisper compute precision (CTranslate2 quantization) |
| `--batch-size` | `16` | Batch size for `BatchedInferencePipeline.transcribe` |
| `--voice` | `alba` | TTS voice / audio-prompt name for `pocket_tts` |

### LLM backend selection (environment variables only)

There is **no CLI flag for the LLM backend or model** — this is entirely environment-driven and easy
to miss. `llm.py` reads:

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | If present, the Claude backend is used. If absent, the assistant falls back to Ollama. |
| `CLAUDE_MODEL` | `claude-sonnet-5` | Overrides the Claude model id |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Overrides the local Ollama model |
| `OLLAMA_HOST` | `http://localhost:11434` | Overrides the Ollama server URL |

The system prompt is a module constant in `llm.py` instructing the model to answer in short, natural
sentences suitable for text-to-speech. The Claude call hardcodes `max_tokens=300`; it is not
configurable.

### Modules

| File | Role |
|---|---|
| `main.py` | Entrypoint. Argument parsing, model construction, the turn loop, metrics summary. |
| `rec.py` | Microphone capture via `sounddevice`. `DEFAULT_SAMPLERATE = 44100`, `DEFAULT_CHANNELS = 1`. No VAD or silence auto-stop. Also provides `has_input_device()`. |
| `asr.py` | Whisper wrapper (`faster-whisper`). Defaults `model="small"`, `device="cpu"`, `compute_type="int8"`, `transcribe(batch_size=16)`. Also provides `save_transcript`. |
| `llm.py` | Backend selection (`active_backend()`), the `Message` type, and `generate_response()` for Claude and Ollama. |
| `tts.py` | `pocket_tts` wrapper. `synthesize_to_file(..., play=False, prebuffer_seconds=1.0)`; `main.py` calls it with `play=True` only when an output device is available. Also provides `has_output_device()`. |
| `metrics.py` | `measure()` context manager, a resource sampler polling every 50 ms, and `default_recorder` with `summary()` / `to_json()`. |

`ASR` and `TextToSpeech` are constructed **once** and reused across turns, so models are not reloaded
per turn. TTS playback is streamed: a background producer thread feeds a queue and a 1.0 s prebuffer is
filled before the `sounddevice` output stream starts, so audio begins before synthesis completes.

### Audio device fallback

`main.py` probes `rec.has_input_device()` and `tts.has_output_device()` once at startup (both call
`sounddevice.query_devices()` and catch the `PortAudioError` raised when no matching device exists —
e.g. on a headless server or cloud VM with no audio hardware):

- **No input device**: the `ASR` model is never constructed (nothing would ever call it), and each turn
  prompts `You (type 'q' to quit):` for typed text instead of recording + transcribing. The typed text
  is still written to `--transcript-file`, same as a real transcript.
- **No output device**: `TextToSpeech.synthesize_to_file` is still called every turn and still writes
  `--tts-output`, just with `play=False`, so no `sounddevice.OutputStream` is opened.

The two are detected independently, so e.g. an input-only or output-only device is handled correctly.

### Files written per turn

All four files are **overwritten**, not appended, on every turn:

- `audio_test.wav` — the mic recording (then immediately read back by the ASR stage)
- `transcript.txt` — the ASR transcript
- `response.txt` — the LLM reply text
- `output.wav` — the synthesized speech

Conversation history is held in memory only (a `list[Message]`); there is no config file and no
persisted session state.

If the transcript comes back empty, the assistant prints `No speech detected; try again.` and returns
without calling the LLM or TTS, so the turn is not consumed.

### Instrumentation

`metrics.measure()` wraps the ASR stage, the LLM call (tagged with the active backend, for example
`llm.generate:claude`), the TTS stage, and `pipeline.turn_total`. `pipeline.turn_total` deliberately
starts *after* recording finishes, so user think-time is excluded from the measured pipeline cost.

Each measurement records wall-clock duration, process CPU percent, whole-machine CPU percent, and peak
process RSS. GPU statistics are added when a backend is detected, probed in priority order: NVIDIA via
`pynvml`, then `torch.cuda`, then Apple `torch.backends.mps`, then none. GPU metrics are purely
informational — their absence never raises.

At the end of the session `default_recorder.summary()` prints a per-stage count / average / max table.
`default_recorder.to_json(path)` exists for dumping the raw records, though `main.py` does not call it.

---

## The dataset pipeline

The pipeline turns a very large raw lyrics CSV into chat-format JSONL training splits, in four steps.

```
song_lyrics.csv  (~9 GB, not in this repo)
        |
        |  1. dataset/sample_lyrics_by_language.py    reservoir-sample N rows of one language
        v
song_lyrics_sample.csv
        |
        |  2. reformat_lyrics.py                      one row per lyric segment
        v
song_lyrics_*_reformatted.csv
        |
        |  3. dataset/llm_generate_questions_from_answers.py   adds question 1 IN PLACE
        v
song_lyrics_*_reformatted.csv   (same file, now with a question column)
        |
        |  4. dataset/finetune/prepare_data.py        row-level split into chat JSONL
        v
dataset/finetune/data/{train,valid,test}.jsonl
```

Run all of these **from the repository root** — the default paths are relative to it.

### 1. Sample by language

Streaming Algorithm-R reservoir sampler. Memory is bounded by the sample size, not the source file
size, which is what makes it viable against the ~9 GB `song_lyrics.csv`.

```bash
uv run dataset/sample_lyrics_by_language.py \
  --source dataset/csvs/song_lyrics.csv \
  --output dataset/csvs/song_lyrics_sample.csv \
  --language en --language-column language \
  --sample-size 50 --seed 0
```

| Flag | Default | Notes |
|---|---|---|
| `--source` | `dataset/csvs/song_lyrics.csv` | |
| `--output` | `dataset/csvs/song_lyrics_sample.csv` | The output directory is not created for you |
| `--language` | `en` | Exact string equality on the language column |
| `--language-column` | `language` | Raises `ValueError` if not found in the header |
| `--sample-size` | `50` | Reservoir capacity |
| `--seed` | none | **Output is nondeterministic unless you pass this** |

Prints the total matching rows seen, the number sampled, and the output path. Stdlib only. Rows shorter
than the language-column index are silently skipped.

### 2. Reformat into segments

Splits each song into structural segments and emits one output row per segment. Boundaries and names
are inferred from bracket tags (`[Chorus]`), lone parenthesized single-word cues, or a learned
vocabulary of recurring bracket-tag words. Segment names are normalized against an ordered list of
known types (pre-chorus, post-chorus, chorus, verse, intro, outro, bridge, hook, refrain, interlude,
breakdown, instrumental, spoken); anything unmatched becomes `unk`.

```bash
uv run reformat_lyrics.py <input_csv> <output_csv> [--limit N] [--vocab-out PATH] [--reset-vocab]
```

| Argument | Default | Notes |
|---|---|---|
| `input_csv` (positional) | required | Must have a `lyrics` column |
| `output_csv` (positional) | required | A separate file; the input is never modified |
| `--limit` | none | Process only the first N data rows (for testing) |
| `--vocab-out` | `collected_segment_names.txt` | Collected bracket-tag vocabulary, one lowercase word per line. Merged with its existing contents, and the merged result drives segmentation |
| `--reset-vocab` | off | Ignore and overwrite the stored vocabulary, so segmentation uses only this input's words |

Length normalization is governed by module constants: segments longer than `MAX_SEGMENT_LINES = 16`
are subdivided into roughly equal chunks that keep the parent name; segments shorter than
`MIN_SEGMENT_LINES = 4` are iteratively merged into their shorter neighbour, producing composite names
joined with `+` (for example `chorus+verse`). A bracket-tag word enters the vocabulary only if it is at
least `MIN_VOCAB_WORD_LEN = 4` characters and appears in at least `MIN_VOCAB_SONG_COUNT = 2` distinct
songs.

The vocabulary file **accumulates across runs**: words found in this run are merged into whatever the
file already contains, so processing several corpora in sequence builds up a combined catalogue rather
than leaving only the last run's words. It is written through a temporary file and renamed into place,
so an interrupted run cannot truncate the accumulated list.

**Segmentation uses the accumulated vocabulary**, not just the words collected from the current input.
A tag learned from an earlier corpus is recognised as a segment boundary in every later run. This makes
the script better at segmenting small inputs, which on their own would not meet the
`MIN_VOCAB_SONG_COUNT = 2` threshold for any tag.

Two consequences follow, and both matter if you are generating training data:

- **Output depends on the vocabulary file's history, not on the input alone.** Re-running the same
  input after the vocabulary has grown can produce different segments. To reproduce a specific output
  you need the same input *and* the same vocabulary file.
- **Accumulated words become segment names.** The vocabulary collects any recurring bracket-tag word,
  including artist names and ad-libs, and unrecognised names pass through verbatim rather than
  normalizing to a known type. Expect names such as `chorus+<word>` alongside the canonical ones.

Pass `--reset-vocab` to ignore and overwrite the stored vocabulary, restricting segmentation to the
current input. That is the reproducible mode: same input, same output, regardless of run history.

The output keeps the input columns and inserts `segment index` (1-based, per song) and `segment name`
immediately after `lyrics`; the `lyrics` value becomes just that segment's text.

Stdlib only, but note that this script reads the **entire input CSV into memory** — use `--limit`, or
run it on an already-sampled file rather than on the full 9 GB source.

### 3. Generate questions

For each row, makes one OpenAI chat-completion call that treats the lyric segment as an answer someone
gave. The model is instructed to privately and randomly pick one of three askers — a friend, a student,
or a stern employer — without revealing which, and to write a single question in that asker's voice,
focused on one specific element of the lyrics. Responses are forced to JSON via
`response_format={"type": "json_object"}`.

```bash
export OPENAI_API_KEY=...
uv run dataset/llm_generate_questions_from_answers.py \
  dataset/csvs/song_lyrics_50_reformatted.csv --limit 50 --concurrency 8
```

| Flag | Default | Notes |
|---|---|---|
| `csv_path` (positional) | required | **Overwritten in place — there is no `--output`** |
| `--lyrics-column` | `lyrics` | Source text column |
| `--model` | `gpt-4o-mini` | |
| `--base-url` | none | For OpenAI-compatible providers, e.g. Ollama at `http://localhost:11434/v1` |
| `--api-key-env` | `OPENAI_API_KEY` | The *name* of the env var holding the key; hard-fails at startup if unset |
| `--limit` | none | Only the first N rows from `--start_entry`; the main cost-control lever |
| `--start_entry` | `0` | 0-based row index to resume from |
| `--concurrency` | `8` | Thread pool size (the OpenAI client is shared across threads) |

Behaviour worth knowing:

- Adds and populates a single `question 1` column with a persona-styled question (the model chooses
  friend / student / employer itself, not the script). Failed rows are left blank and logged; a partial
  failure never aborts the run.
- Safe to re-run. A row is skipped with no API call if `question 1` is already filled or the lyrics cell
  is empty; previously failed (blank) rows are retried.
- Writes are atomic: the whole CSV is rewritten to `<csv_path>.tmp`, flushed and fsynced, then
  `os.replace()`d after every completed row, so progress is durable and a crash mid-write cannot
  corrupt the target.
- Retries only on HTTP 429, up to 5 times, with exponential backoff from 5 s doubling to a 60 s cap.
  Under free-tier rate limits this can add substantial wall-clock time.
- No sampling parameters are set — no `temperature`, `top_p`, or `max_tokens`; provider defaults apply,
  so cost scales linearly with unfinished rows and is not capped per call.

### 4. Prepare JSONL splits

Converts the reformatted CSV into chat-format JSONL. Each row's `question 1` becomes one training
example pairing that question (user turn) with the row's `lyrics` (assistant turn), so a row yields at
most one example. `question 2` / `question 3`, if present from an older run of step 3, are ignored.
Splitting happens at the **row** level so lyrics from the same song can't end up split across train and
test.

```bash
uv run dataset/finetune/prepare_data.py
```

or with explicit arguments:

```bash
uv run dataset/finetune/prepare_data.py dataset/csvs/song_lyrics_50_reformatted.csv \
  --output-dir dataset/finetune/data --seed 0 --train-frac 0.8 --valid-frac 0.1
```

| Argument | Default | Notes |
|---|---|---|
| `csv_path` (positional, optional) | `dataset/csvs/song_lyrics_50_reformatted.csv` | |
| `--output-dir` | `dataset/finetune/data` | Created if missing |
| `--seed` | `0` | Seeds a local `random.Random` instance, not global state |
| `--train-frac` | `0.8` | |
| `--valid-frac` | `0.1` | The remainder goes to test |
| `--limit` | none | First N CSV rows, for testing |

Each output line has the shape:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

There is no system prompt in the records, and no truncation or maximum-length logic anywhere in the
script. Examples with a blank question or blank lyrics are silently dropped; only per-split counts are
printed. Runs are byte-for-byte reproducible for a fixed CSV, seed, fractions, and limit, and each run
unconditionally truncates and overwrites the three JSONL files.

Example counts now equal row counts with a non-blank `question 1` (previously up to 3x that, back when
every question column was used).

### Utility: thin a CSV down to N random rows

`dataset/subsample_csv.py` is a standalone helper, not part of the numbered pipeline above. It
reservoir-samples `count` rows from any CSV in a single streaming pass (memory bounded by `count`, not
file size) and writes them, with the header, to a new file.

```bash
uv run dataset/subsample_csv.py dataset/csvs/song_lyrics_sample300_english_segmented.csv 100 --seed 0
```

| Argument | Default | Notes |
|---|---|---|
| `source` (positional) | required | |
| `count` (positional) | required | Number of rows to keep |
| `--output` | `dataset/<source stem>_sample<count>.csv` | |
| `--seed` | none | Output is nondeterministic unless passed |

---

## Fine-tuning

Both backends train a LoRA adapter for Qwen2.5-3B-Instruct on the same `dataset/finetune/data/` splits,
and both evaluation scripts read the same `test.jsonl` prompts for an apples-to-apples comparison.
Neither evaluation script computes a quantitative metric — they are print-and-inspect tools that show
each generated completion next to the dataset's reference answer.

Generate the data splits (step 4 above) before running either backend.

### Local: MLX on Apple Silicon

Training is driven entirely by a YAML config through the `mlx_lm.lora` CLI that ships with `mlx-lm`:

```bash
mlx_lm.lora -c dataset/finetune/mlx/lora_config.yaml
```

The config comment states the intent plainly: these settings are conservative for an 8 GB
unified-memory Mac, and the run is a cheap local sanity-check of the data and pipeline rather than the
final model.

| Setting | Value |
|---|---|
| Base model | `mlx-community/Qwen2.5-3B-Instruct-4bit` (pre-quantized) |
| `fine_tune_type` / `optimizer` | `lora` / `adamw` |
| `num_layers` | `8` (LoRA on only 8 transformer layers) |
| `batch_size` / `grad_accumulation_steps` | `1` / `4` (effective batch 4) |
| `iters` | `500` |
| `learning_rate` | `1e-5` |
| `max_seq_length` | `1536` |
| `grad_checkpoint` | `true` |
| LoRA keys | `self_attn.q_proj`, `self_attn.v_proj` |
| LoRA `rank` / `scale` / `dropout` | `8` / `20.0` / `0.0` |
| `steps_per_report` / `steps_per_eval` / `save_every` | `10` / `200` / `100` |
| `adapter_path` | `dataset/finetune/mlx/adapters` |
| `seed` | `0` |

Evaluate:

```bash
uv run dataset/finetune/mlx/evaluate_samples.py \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path dataset/finetune/mlx/adapters \
  --test-file dataset/finetune/data/test.jsonl \
  --num-samples 5 --max-tokens 256
```

All flags are optional; the values shown are the defaults. The adapter is fused in at load time. If the
adapter directory is missing, the script exits with a message containing the exact training command
rather than a stack trace. Samples are the **first** N lines of the test file, not a random draw, so
repeated runs show the same prompts. There is no `--temperature` flag — no sampler argument is passed,
so `mlx_lm.generate` defaults apply.

### Cloud: QLoRA on CUDA

```bash
pip install -r dataset/finetune/cloud/requirements-cloud.txt
python dataset/finetune/cloud/train_qlora.py            # add --bf16 on A10/A100/L4
```

The script hard-exits if `torch.cuda.is_available()` is false. It is sized for a single 16 GB Colab T4;
fp16 is the default because the T4 (compute capability 7.5) lacks bf16 tensor cores, so `--bf16` should
only be passed on Ampere-or-newer hardware.

| Flag | Default |
|---|---|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` |
| `--train-file` | `dataset/finetune/data/train.jsonl` |
| `--valid-file` | `dataset/finetune/data/valid.jsonl` |
| `--output-dir` | `dataset/finetune/cloud/adapters` |
| `--bf16` | off (fp16 unless set) |
| `--max-length` | `1536` |
| `--lora-r` | `16` |
| `--lora-alpha` | `32` |
| `--lora-dropout` | `0.05` |
| `--per-device-train-batch-size` | `2` |
| `--gradient-accumulation-steps` | `8` |
| `--num-train-epochs` | `3` |
| `--learning-rate` | `2e-4` |

Quantization is 4-bit NF4 with double quantization, compute dtype bfloat16 with `--bf16` and float16
otherwise. LoRA targets all seven attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`) with `bias="none"` and `task_type="CAUSAL_LM"` — a notably wider
target set than the MLX config. Training uses `trl.SFTTrainer` with `packing=False` and
`assistant_only_loss=True`, `optim="paged_adamw_8bit"`, gradient checkpointing, `warmup_ratio=0.03`,
effective batch size 16, per-epoch evaluation and checkpointing, and `load_best_model_at_end=True`
selecting on `eval_loss`.

The base model is the full-precision Hugging Face repo quantized at load time — not the pre-quantized
`mlx-community` variant the MLX path uses. It is public and non-gated, so no Hugging Face token is
required. There is no `--resume-from-checkpoint` flag; per-epoch checkpoints are written inside the
output directory, but resuming means invoking the Trainer mechanics manually.

Evaluate:

```bash
python dataset/finetune/cloud/evaluate_adapter.py \
  --adapter-path dataset/finetune/cloud/adapters \
  --test-file dataset/finetune/data/test.jsonl \
  --num-samples 5 --max-new-tokens 256
```

Decoding is hardcoded greedy (`do_sample=False`), so results are deterministic. The base model id is
not a flag — both tokenizer and model load from the adapter path, and `AutoPeftModelForCausalLM`
resolves the base model from the adapter's `adapter_config.json`. Note that this script has **no CUDA
check** (unlike training) yet loads with `dtype=torch.float16` and `device_map="auto"`; on CPU it will
be unusably slow or fail outright. It also has no friendly missing-adapter check — it fails inside
`from_pretrained` if you have not trained yet.

---

## Project layout

```
main.py                     Voice assistant entrypoint
asr.py                      Whisper / faster-whisper wrapper
llm.py                      Claude + Ollama backends, env-driven selection
tts.py                      pocket_tts streaming synthesis wrapper
rec.py                      Microphone capture (sounddevice)
metrics.py                  measure() instrumentation + aggregated summary
reformat_lyrics.py          Dataset step 2: split songs into segments

dataset/
  sample_lyrics_by_language.py            Step 1: reservoir sample by language
  llm_generate_questions_from_answers.py  Step 3: synthesize a question (in place)
  subsample_csv.py                        Utility: keep only N random rows from any CSV
  csvs/                                   Lyrics CSVs, including the ~9 GB source (untracked)
  finetune/
    prepare_data.py                       Step 4: CSV -> chat JSONL splits
    data/                                 train/valid/test.jsonl  (gitignored)
    mlx/
      lora_config.yaml                    mlx_lm.lora hyperparameters
      evaluate_samples.py                 Qualitative eval, Apple Silicon
      adapters/                           MLX adapter output (gitignored)
    cloud/
      train_qlora.py                      CUDA QLoRA training
      evaluate_adapter.py                 Qualitative eval, CUDA
      requirements-cloud.txt              CUDA-only dependency set
      adapters/                           Cloud adapter output (gitignored)

faster-whisper/                           Vendored upstream source tree (gitignored)
pocket-tts/                               Vendored upstream source tree (gitignored)
```

---

## Notes and gotchas

- **This is a flat script layout, not an installable package.** There is no `src/` and no console-script
  entrypoint; `pyproject.toml` sets `[tool.uv] package = false` so `uv sync` just builds the dependency
  environment. Run everything via `uv run <script.py>` from the repository root, starting with `main.py`
  for the voice assistant.
- **`llm_generate_questions_from_answers.py` no longer fills `question 2` / `question 3`.** Those
  columns are only vestigial if present from an older run; `prepare_data.py` ignores them and reads
  `question 1` only.
- **Generated data and adapters are gitignored and must be regenerated.** `.gitignore` excludes
  `dataset/finetune/data/`, `dataset/finetune/mlx/adapters/`, and `dataset/finetune/cloud/adapters/`.
  A fresh clone has no JSONL splits and no trained adapters: run `prepare_data.py` before training, and
  train before evaluating.
- **The ~9 GB source CSV is not in the repository.** `dataset/csvs/song_lyrics.csv` is untracked and
  must be supplied separately. Smaller sampled and reformatted CSVs are present in the working tree
  under `dataset/csvs/`.
- **`llm_generate_questions_from_answers.py` overwrites its input CSV in place.** There is no `--output`
  flag. Back the file up, or keep it under version control, before running it.
- **`reformat_lyrics.py` accumulates its vocabulary, and that vocabulary drives segmentation.**
  `--vocab-out` (default `collected_segment_names.txt`) is merged with its existing contents on every
  run, and the merged set is what the segmenter matches against. A small `--limit` run no longer
  discards a vocabulary built from a larger corpus, but the output is no longer a pure function of the
  input either — reproducing a given output requires the same vocabulary file. Use `--reset-vocab` for
  a self-contained, reproducible run. This script does not create `.bak` files — the `.bak` files
  present in `dataset/csvs/` came from some other process.
- **Voice assistant run artifacts are not gitignored.** `audio_test.wav`, `output.wav`,
  `transcript.txt`, and `response.txt` are rewritten on every turn and are not covered by `.gitignore`
  (nor is `.DS_Store`). Leftover copies of all four already exist at the repository root.
- **The MLX run is not a production fine-tune.** Its config is explicit that it is a low-memory local
  sanity-check; the cloud QLoRA path is the one intended to produce a final-quality adapter.
- **Segment naming is heuristic.** It can produce composite names such as `chorus+verse`, or `unk` when
  no known type matches. Downstream steps do not depend on the segment name, but be aware of it if you
  intend to filter on it.
- **The dataset is song lyrics and contains explicit language.** Output from either evaluation script
  will reflect that.
- **The fine-tuned adapter is not used by the voice assistant.** `main.py` talks to Claude or Ollama and
  has no code path for loading a local LoRA adapter. Wiring the two halves together is not implemented.
- **Sampling is nondeterministic by default.** `sample_lyrics_by_language.py` seeds the RNG only when
  `--seed` is passed explicitly.
