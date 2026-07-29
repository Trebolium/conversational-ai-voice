import logging
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

from metrics import measure

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class ASR:
    """Wraps a loaded Whisper model so it can be reused across many
    transcribe() calls instead of reloading from disk every time (as is
    needed in a multi-turn conversation loop)."""

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        logger.info(
            "Loading model %s on %s with compute type %s",
            model,
            device,
            compute_type,
        )
        whisper_model = WhisperModel(model, device=device, compute_type=compute_type)
        self.batched_model = BatchedInferencePipeline(model=whisper_model)

    def transcribe(
        self,
        input_file: str | Path,
        *,
        batch_size: int = 16,
    ) -> tuple[str, list[TranscriptSegment]]:
        """Transcribe an audio file and return full text plus per-segment details."""
        input_file = Path(input_file)

        # faster-whisper returns `segments` as a lazy generator -- the actual
        # transcription work happens while iterating it, so the iteration
        # must be inside the measured block, not just the `.transcribe()`
        # call itself.
        with measure("asr.transcribe"):
            segments, info = self.batched_model.transcribe(str(input_file), batch_size=batch_size)
            transcript_segments = [
                TranscriptSegment(segment.start, segment.end, segment.text)
                for segment in segments
            ]

        full_text = " ".join(segment.text.strip() for segment in transcript_segments).strip()

        for segment in transcript_segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        logger.info("Detected language: %s (%.2f)", info.language, info.language_probability)
        return full_text, transcript_segments


def transcribe(
    input_file: str | Path,
    *,
    model: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 16,
) -> tuple[str, list[TranscriptSegment]]:
    """One-shot convenience wrapper: loads a model and transcribes a single
    file. For a conversation loop with multiple turns, use ASR directly so
    the model is loaded only once."""
    return ASR(model=model, device=device, compute_type=compute_type).transcribe(
        input_file, batch_size=batch_size
    )


def save_transcript(text: str, output_file: str | Path) -> Path:
    output_file = Path(output_file)
    output_file.write_text(text, encoding="utf-8")
    logger.info("Saved transcript to %s", output_file)
    return output_file
