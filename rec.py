import logging
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

DEFAULT_SAMPLERATE = 44100
DEFAULT_CHANNELS = 1


def has_input_device() -> bool:
    """Return whether a usable audio input device is available. False in
    headless/audio-less environments (e.g. a cloud VM with no PortAudio
    devices), so callers can fall back to text input instead of letting
    sd.InputStream raise."""
    try:
        sd.query_devices(kind="input")
        return True
    except Exception:
        return False


def record_audio(
    output_path: str | Path = "audio_test.wav",
    samplerate: int = DEFAULT_SAMPLERATE,
    channels: int = DEFAULT_CHANNELS,
    *,
    wait_for_start: bool = True,
) -> Path:
    """Record from the microphone until the user presses Enter.

    Set wait_for_start=False when the caller has already gotten a
    start-recording confirmation from the user (e.g. a conversation loop
    that also offers a quit option at that same prompt).
    """
    output_path = Path(output_path)
    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        if status:
            logger.warning(status)
        frames.append(indata.copy())

    if wait_for_start:
        input("Press Enter to start recording...")
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        print("Recording... press Enter to stop.")
        input()

    # import pdb; pdb.set_trace()
    audio = np.concatenate(frames, axis=0)
    sf.write(output_path, audio, samplerate)
    logger.info("Saved recording to %s", output_path)
    return output_path
