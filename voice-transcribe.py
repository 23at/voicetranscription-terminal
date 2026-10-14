import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Record audio and transcribe it with whisper-cli.")
    parser.add_argument(
        "-m",
        "--model",
        default=os.path.expanduser("~/whisper-models/ggml-small.en.bin"),
        help="Path to the whisper model file.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds.",
    )
    return parser.parse_args()


def ensure_dependencies(model_path):
    if shutil.which("whisper-cli") is None:
        raise RuntimeError(
            "whisper-cli was not found in PATH. Install whisper.cpp and ensure the binary is available."
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Update --model or set the correct path."
        )


def record_audio(duration_seconds: float, sample_rate: int = SAMPLE_RATE):
    print(f"Recording for {duration_seconds:.1f} seconds...")
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    if audio.size == 0:
        raise RuntimeError("No audio was captured.")

    return audio[:, 0]


def save_wav(audio, wav_path: str, sample_rate: int = SAMPLE_RATE):
    audio = np.asarray(audio, dtype=np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))

    pcm = np.clip(audio, -1.0, 1.0)
    int16_audio = np.round(pcm * 32767).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())


def transcribe(model_path: str, wav_path: str):
    result = subprocess.run(
        ["whisper-cli", "-m", model_path, "-f", wav_path, "-otxt"],
        capture_output=True,
        text=True,
    )

    txt_path = wav_path + ".txt"
    transcript_text = ""

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()

    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "Unknown whisper-cli error").strip()
        raise RuntimeError(f"Transcription failed: {error_text}")

    return transcript_text or (result.stdout or result.stderr or "").strip()


def main():
    args = parse_args()

    try:
        ensure_dependencies(args.model)
        audio = record_audio(args.duration)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            save_wav(audio, wav_path)
            transcript = transcribe(args.model, wav_path)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

            txt_path = wav_path + ".txt"
            if os.path.exists(txt_path):
                os.unlink(txt_path)

        print("\n--- Transcript ---")
        print(transcript)
        return 0

    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())