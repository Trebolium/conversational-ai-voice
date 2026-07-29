import argparse
import logging
from pathlib import Path

from asr import ASR, save_transcript
from llm import Message, active_backend, generate_response
from metrics import default_recorder, gpu_backend, measure
from rec import record_audio
from tts import TextToSpeech

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


def run_turn(
    args: argparse.Namespace,
    asr: ASR,
    tts: TextToSpeech,
    conversation: list[Message],
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

        conversation.append(Message(role="user", content=transcript))

        logger.info("Generating response with %s backend", active_backend())
        response = generate_response(conversation)
        conversation.append(Message(role="assistant", content=response))
        Path(args.response_file).write_text(response, encoding="utf-8")

        print(f"\nYou: {transcript}")
        print(f"Assistant: {response}")

        print("Synthesizing speech (playback starts as each sentence is ready)...")
        tts.synthesize_to_file(response, args.tts_output, play=True)
    return True


def main() -> None:
    args = parse_args()

    logger.info("GPU metrics backend: %s", gpu_backend() or "none (CPU-only metrics)")

    asr = ASR(model=args.model, device=args.device, compute_type=args.compute_type)
    tts = TextToSpeech(voice=args.voice)

    conversation: list[Message] = []

    print("Loaded. Let's talk.")
    while True:
        choice = input(
            "\nPress Enter to record your message (or type 'q' to quit): "
        ).strip().lower()
        if choice in QUIT_WORDS:
            print("Ending conversation.")
            break

        run_turn(args, asr, tts, conversation)

    print("\n--- Session metrics summary ---")
    print(default_recorder.summary())


if __name__ == "__main__":
    main()
