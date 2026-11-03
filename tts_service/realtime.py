"""Persistent local Kokoro speech output through RealtimeTTS."""

import argparse
from dataclasses import dataclass
import math
import sys
import time
from typing import Iterable, Optional


# Favor a quick first phrase without chopping the response into tiny fragments.
PLAY_OPTIONS = {
    "fast_sentence_fragment": True,
    "fast_sentence_fragment_allsentences": True,
    "fast_sentence_fragment_allsentences_multiple": False,
    "buffer_threshold_seconds": 0.25,
    "minimum_sentence_length": 18,
    "minimum_first_fragment_length": 18,
    "sentence_fragment_delimiters": ".?!;:,\n…—",
    "force_first_fragment_after_words": 12,
    "fragment_lookahead_words": 0,
}


@dataclass
class SpeechTiming:
    first_audio_seconds: Optional[float]


def validate_options(voice: str, speed: float):
    if not voice.strip():
        raise RuntimeError("--tts-voice must not be empty.")
    if not math.isfinite(speed) or speed <= 0:
        raise RuntimeError("--tts-speed must be a finite number greater than zero.")


def _load_realtime_tts():
    try:
        from RealtimeTTS import KokoroEngine, TextToAudioStream
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Could not load RealtimeTTS/Kokoro. Use Python 3.11 or 3.12 and run: "
            "python -m pip install -r requirements-tts.txt. "
            "On macOS, first run: brew install portaudio espeak-ng. "
            f"Details: {exc}"
        ) from exc
    return KokoroEngine, TextToAudioStream


class KokoroSession:
    """Own one warmed Kokoro model and reuse it for multiple conversation turns."""

    def __init__(self, voice: str = "af_heart", speed: float = 1.0, warmup: bool = True):
        validate_options(voice, speed)
        KokoroEngine, TextToAudioStream = _load_realtime_tts()

        self.engine = None
        self.stream = None
        self.closed = False
        load_started = time.perf_counter()
        try:
            print("Loading Kokoro...")
            self.engine = KokoroEngine(voice=voice, default_speed=speed)
            voices = {item.name for item in self.engine.get_voices()}
            if voice not in voices:
                raise RuntimeError(
                    f"Unknown Kokoro voice: {voice}. Try af_heart, af_bella, "
                    "am_michael, bf_emma, or bm_george."
                )
            self.stream = TextToAudioStream(self.engine)
            self.model_load_seconds = time.perf_counter() - load_started
            print(f"[Latency] Kokoro model load: {self.model_load_seconds:.2f}s")

            self.warmup_seconds = 0.0
            if warmup:
                self._warm_up()
        except BaseException:
            self.close()
            raise

    def _warm_up(self):
        """Run muted synthesis so the first real turn avoids lazy setup costs."""
        warmup_started = time.perf_counter()
        received_audio = False

        def on_audio_chunk(chunk):
            nonlocal received_audio
            if chunk:
                received_audio = True

        self.stream.feed("Ready.")
        self.stream.play(
            muted=True,
            fast_sentence_fragment=False,
            minimum_sentence_length=1,
            minimum_first_fragment_length=1,
            on_audio_chunk=on_audio_chunk,
        )
        if not received_audio:
            raise RuntimeError("Kokoro warm-up produced no audio.")
        self.warmup_seconds = time.perf_counter() - warmup_started
        print(f"[Latency] Kokoro warm-up: {self.warmup_seconds:.2f}s (startup only)")

    def speak_stream(
        self,
        text_chunks: Iterable[str],
        latency_started: Optional[float] = None,
        latency_origin: str = "Gemini request",
    ) -> SpeechTiming:
        """Consume streaming text and play phrase-sized audio as soon as it is ready."""
        if self.closed or self.stream is None:
            raise RuntimeError("Kokoro session is closed.")

        received_audio = False
        first_audio_seconds = None

        def on_audio_chunk(chunk):
            nonlocal received_audio, first_audio_seconds
            if chunk and not received_audio:
                received_audio = True
                if latency_started is not None:
                    first_audio_seconds = time.perf_counter() - latency_started
                    print(
                        f"\n[Latency] First audible speech: {first_audio_seconds:.2f}s "
                        f"after {latency_origin}"
                    )

        try:
            self.stream.feed(iter(text_chunks))
            self.stream.play(on_audio_chunk=on_audio_chunk, **PLAY_OPTIONS)
            if not received_audio:
                raise RuntimeError(
                    "Kokoro produced no playable audio. Check the model cache "
                    "and your default audio output device."
                )
            return SpeechTiming(first_audio_seconds=first_audio_seconds)
        except KeyboardInterrupt:
            self.stream.stop()
            raise
        except Exception as exc:
            self.stream.stop()
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Kokoro speech output failed: {exc}") from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.stream is not None:
                self.stream.stop()
        finally:
            if self.engine is not None:
                self.engine.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def speak(text: str, voice: str = "af_heart", speed: float = 1.0):
    """One-shot compatibility wrapper; conversation mode should use KokoroSession."""
    if not text.strip():
        raise RuntimeError("Cannot speak an empty response.")
    with KokoroSession(voice=voice, speed=speed) as session:
        return session.speak_stream(iter((text,)))


def main():
    parser = argparse.ArgumentParser(description="Test local speech without Gemini or a microphone.")
    parser.add_argument("text", nargs="?", default="Hello. Local voice output is working.")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    try:
        speak(args.text, voice=args.voice, speed=args.speed)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
