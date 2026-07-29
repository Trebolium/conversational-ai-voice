import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from pocket_tts import TTSModel

from metrics import measure, record_instant

logger = logging.getLogger(__name__)

_STREAM_DONE = object()


class TextToSpeech:
    def __init__(self, voice: str = "alba"):
        self.model = TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt(voice)

    def synthesize(self, text: str):
        audio = self.model.generate_audio(self.voice_state, text)
        return audio, self.model.sample_rate

    def synthesize_stream(self, text: str):
        """Yield audio chunks as soon as each is ready (many small chunks per
        sentence, as the decoder produces them), instead of blocking until
        the whole response is synthesized."""
        for chunk in self.model.generate_audio_stream(self.voice_state, text):
            yield chunk.numpy(), self.model.sample_rate

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path = "output.wav",
        *,
        play: bool = False,
        prebuffer_seconds: float = 1.0,
    ) -> Path:
        """Synthesize speech for text, saving it to output_path. If play is
        True, generation runs in a background thread that feeds a queue,
        while this thread drains the queue into a continuous output stream --
        decoupling audio-device feeding from chunk generation so a slow
        chunk doesn't starve playback. A short prebuffer is collected before
        playback starts to absorb generation-speed jitter between chunks."""
        output_path = Path(output_path)
        sample_rate = self.model.sample_rate

        if not play:
            chunks: list[np.ndarray] = []
            start = time.perf_counter()
            first_chunk_seen = False
            with measure("tts.synthesize"):
                for chunk, _sr in self.synthesize_stream(text):
                    if not first_chunk_seen:
                        record_instant("tts.time_to_first_chunk", time.perf_counter() - start)
                        first_chunk_seen = True
                    chunks.append(chunk)
            audio = np.concatenate(chunks)
            sf.write(output_path, audio, sample_rate)
            logger.info("Saved TTS audio to %s", output_path)
            return output_path

        chunks: list[np.ndarray] = []
        chunk_queue: queue.Queue = queue.Queue()

        def produce() -> None:
            start = time.perf_counter()
            first_chunk_seen = False
            try:
                with measure("tts.synthesize_stream"):
                    for chunk_audio, _sr in self.synthesize_stream(text):
                        if not first_chunk_seen:
                            record_instant(
                                "tts.time_to_first_chunk", time.perf_counter() - start
                            )
                            first_chunk_seen = True
                        chunks.append(chunk_audio)
                        chunk_queue.put(chunk_audio)
            finally:
                chunk_queue.put(_STREAM_DONE)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()

        # Collect a small prebuffer before opening the output stream, so an
        # early slow chunk doesn't immediately starve playback.
        prebuffered: list[np.ndarray] = []
        prebuffered_samples = 0
        target_samples = int(prebuffer_seconds * sample_rate)
        producer_done = False
        while prebuffered_samples < target_samples:
            item = chunk_queue.get()
            if item is _STREAM_DONE:
                producer_done = True
                break
            prebuffered.append(item)
            prebuffered_samples += item.shape[-1]

        stream = sd.OutputStream(samplerate=sample_rate, channels=1)
        stream.start()
        try:
            for chunk_audio in prebuffered:
                stream.write(np.ascontiguousarray(chunk_audio))
            while not producer_done:
                item = chunk_queue.get()
                if item is _STREAM_DONE:
                    break
                stream.write(np.ascontiguousarray(item))
        finally:
            producer.join()
            stream.stop()
            stream.close()

        audio = np.concatenate(chunks)
        sf.write(output_path, audio, sample_rate)
        logger.info("Saved TTS audio to %s", output_path)

        return output_path


def speak(text: str, *, voice: str = "alba", output_path: str = "output.wav") -> Path:
    return TextToSpeech(voice=voice).synthesize_to_file(text, output_path, play=True)
