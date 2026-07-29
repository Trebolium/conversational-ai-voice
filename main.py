import argparse
import logging
from pathlib import Path

from asr import ASR, save_transcript
from llm import Message, active_backend, generate_response
from metrics import default_recorder, gpu_backend, measure
from rec import has_input_device, record_audio
from tts import TextToSpeech, has_output_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUIT_WORDS = {"q", "quit", "exit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Have a spoken conversation with the LLM: record, transcribe, "
        "respond, and speak back, looping until you choose to quit."
    )
    parser.add_argument("--audio-file", default="audio_test.wav")
    parser.add_argument("--transcript-file", default="transcript.txt")
    parser.add_argument("--response-file", default="response.txt")
    parser.add_argument("--tts-output", default="output.wav")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--voice", default="alba")
    return parser.parse_args()


def _generate_and_speak(
    args: argparse.Namespace,
    tts: TextToSpeech,
    conversation: list[Message],
    transcript: str,
    *,
    audio_output_available: bool,
) -> None:
    conversation.append(Message(role="user", content=transcript))

    logger.info("Generating response with %s backend", active_backend())
    response = generate_response(conversation)
    conversation.append(Message(role="assistant", content=response))
    Path(args.response_file).write_text(response, encoding="utf-8")

    print(f"\nYou: {transcript}")
    print(f"Assistant: {response}")

    if audio_output_available:
        print("Synthesizing speech (playback starts as each sentence is ready)...")
    else:
        print(f"Synthesizing speech (no audio output device; saving to {args.tts_output})...")
    tts.synthesize_to_file(response, args.tts_output, play=audio_output_available)


def run_turn(
    args: argparse.Namespace,
    asr: ASR,
    tts: TextToSpeech,
    conversation: list[Message],
    *,
    audio_output_available: bool,
) -> bool:
    """Record one utterance, get a response, and speak it. Returns False if
    no speech was detected (so the caller can retry without consuming a turn
    of conversation history)."""
    audio_path = record_audio(args.audio_file, wait_for_start=False)

    # Wraps everything downstream of recording -- i.e. actual pipeline
    # compute cost, not time spent waiting on the user to speak.
    with measure("pipeline.turn_total"):
        transcript, _segments = asr.transcribe(audio_path, batch_size=args.batch_size)
        save_transcript(transcript, args.transcript_file)

        if not transcript:
            print("No speech detected; try again.")
            return False

        _generate_and_speak(
            args, tts, conversation, transcript, audio_output_available=audio_output_available
        )
    return True


def run_text_turn(
    args: argparse.Namespace,
    tts: TextToSpeech,
    conversation: list[Message],
    transcript: str,
    *,
    audio_output_available: bool,
) -> None:
    """Text-input equivalent of run_turn, used when no microphone is
    available: skips recording and ASR since the transcript is already
    typed text."""
    save_transcript(transcript, args.transcript_file)
    with measure("pipeline.turn_total"):
        _generate_and_speak(
            args, tts, conversation, transcript, audio_output_available=audio_output_available
        )


def main() -> None:
    args = parse_args()

    logger.info("GPU metrics backend: %s", gpu_backend() or "none (CPU-only metrics)")

    audio_input_available = has_input_device()
    audio_output_available = has_output_device()

    # Skip loading the (heavyweight) ASR model entirely when there's no mic
    # to feed it -- it would never be used.
    asr = (
        ASR(model=args.model, device=args.device, compute_type=args.compute_type)
        if audio_input_available
        else None
    )
    tts = TextToSpeech(voice=args.voice)

    conversation: list[Message] = []

    if not audio_input_available:
        print("No audio input device detected; falling back to typed text input.")
    if not audio_output_available:
        print(f"No audio output device detected; replies will be saved to {args.tts_output} instead of played.")

    print("Loaded. Let's talk.")
    while True:
        if audio_input_available:
            choice = input(
                "\nPress Enter to record your message (or type 'q' to quit): "
            ).strip().lower()
            if choice in QUIT_WORDS:
                print("Ending conversation.")
                break

            run_turn(args, asr, tts, conversation, audio_output_available=audio_output_available)
        else:
            text = input("\nYou (type 'q' to quit): ").strip()
            if text.lower() in QUIT_WORDS:
                print("Ending conversation.")
                break
            if not text:
                print("No input entered; try again.")
                continue

            run_text_turn(args, tts, conversation, text, audio_output_available=audio_output_available)

    print("\n--- Session metrics summary ---")
    print(default_recorder.summary())


if __name__ == "__main__":
    main()
