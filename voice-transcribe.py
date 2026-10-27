import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Optional
import urllib.error
import urllib.parse
import urllib.request
import wave
import numpy as np
import sounddevice as sd


def load_dotenv(path=None):
    env_path = os.path.abspath(path or os.path.join(os.getcwd(), ".env"))
    if not os.path.isfile(env_path):
        return False

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    return True


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Record audio and transcribe it with whisper-cli.")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to an audio file to transcribe instead of recording from the microphone.",
    )
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
    parser.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        help="Gemini model to use (default: gemini-3.8-flash or GEMINI_MODEL).",
    )
    parser.add_argument(
        "--gemini-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for Gemini before timing out (default: 180).",
    )
    parser.add_argument(
        "--speak-response",
        action="store_true",
        help="Generate and play Gemini's response using the local VoxCPM2 model.",
    )
    parser.add_argument(
        "--tts-reference-audio",
        help="Optional audio sample to clone for the spoken Gemini response.",
    )
    parser.add_argument(
        "--tts-python",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tts_service",
            ".venv312",
            "bin",
            "python",
        ),
        help="Python interpreter for MLX Audio (defaults to tts_service/.venv312).",
    )
    parser.add_argument(
        "--tts-inference-steps",
        type=int,
        default=5,
        help="VoxCPM2 inference steps for speech generation (default: 5).",
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

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create an API key in Google AI Studio and set it in your environment."
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


def transcribe(model_path: str, audio_path: str):
    with tempfile.TemporaryDirectory() as output_dir:
        output_path = os.path.join(output_dir, "transcript")
        result = subprocess.run(
            ["whisper-cli", "-m", model_path, "-f", audio_path, "-otxt", "-of", output_path],
            capture_output=True,
            text=True,
        )

        txt_path = output_path + ".txt"
        transcript_text = ""

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript_text = f.read().strip()

    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "Unknown whisper-cli error").strip()
        raise RuntimeError(f"Transcription failed: {error_text}")

    return transcript_text or (result.stdout or result.stderr or "").strip()


def ask_gemini(transcript: str, model: str, api_key: str, timeout_seconds: float):
    model_path = urllib.parse.quote(model, safe="-")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
    payload = json.dumps(
        {"contents": [{"parts": [{"text": transcript}]}]}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        try:
            error_response = json.loads(error_body)
            if isinstance(error_response, dict):
                error_body = error_response.get("error", {}).get("message", error_body)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Gemini API request failed ({exc.code}): {error_body}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RuntimeError(
                f"Gemini API request timed out after {timeout_seconds:g} seconds."
            ) from exc
        raise RuntimeError(f"Could not connect to the Gemini API: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Gemini API request timed out after {timeout_seconds:g} seconds."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini API returned an invalid JSON response.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Gemini API returned an unexpected response.")

    candidates = result.get("candidates", [])
    if not candidates:
        feedback = result.get("promptFeedback", {}).get("blockReason", "no response candidate")
        raise RuntimeError(f"Gemini did not return a response: {feedback}")

    response_text = "".join(
        part.get("text", "")
        for part in candidates[0].get("content", {}).get("parts", [])
    ).strip()
    if not response_text:
        raise RuntimeError("Gemini returned an empty response.")

    return response_text


def speak_response(
    text: str,
    python_path: str,
    inference_steps: int,
    reference_audio: Optional[str] = None,
):
    generator_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tts_service", "generate.py"
    )
    player_path = shutil.which("afplay")
    if not os.path.isfile(python_path):
        raise FileNotFoundError(
            f"MLX Audio Python interpreter not found: {python_path}. "
            "Set --tts-python to the Python executable in your MLX Audio environment."
        )
    if not os.path.isfile(generator_path):
        raise FileNotFoundError(f"Speech generator not found: {generator_path}")
    if player_path is None:
        raise RuntimeError("afplay was not found; audio playback requires macOS.")
    if inference_steps < 1:
        raise RuntimeError("--tts-inference-steps must be at least 1.")
    if reference_audio:
        reference_audio = os.path.abspath(os.path.expanduser(reference_audio))
        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(f"Reference audio file not found: {reference_audio}")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(project_dir, "generated_audio")
    os.makedirs(audio_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    audio_path = os.path.join(audio_dir, f"gemini-response-{timestamp}.wav")

    command = [
        python_path,
        generator_path,
        "--text-stdin",
        "--output",
        audio_path,
        "--inference-steps",
        str(inference_steps),
    ]
    if reference_audio:
        command.extend(["--reference-audio", reference_audio])
    result = subprocess.run(command, input=text, capture_output=True, text=True)
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "Unknown TTS error").strip()
        raise RuntimeError(f"Speech generation failed: {error_text}")
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("Speech generation completed without creating an audio file.")
    if result.stdout.strip():
        print(result.stdout.strip())

    print(f"Saved response audio: {audio_path}")
    playback = subprocess.run([player_path, audio_path], capture_output=True, text=True)
    if playback.returncode != 0:
        error_text = (playback.stderr or playback.stdout or "Unknown audio playback error").strip()
        raise RuntimeError(f"Audio playback failed: {error_text}")


def main():
    args = parse_args()

    try:
        ensure_dependencies(args.model)
        if args.input:
            audio_path = os.path.expanduser(args.input)
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            transcript = transcribe(args.model, audio_path)
        else:
            audio = record_audio(args.duration)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            try:
                save_wav(audio, wav_path)
                transcript = transcribe(args.model, wav_path)
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

        print("\n--- Transcript ---")
        print(transcript)
        if args.gemini_timeout <= 0:
            raise RuntimeError("--gemini-timeout must be greater than zero.")
        response = ask_gemini(
            transcript,
            args.gemini_model,
            os.environ["GEMINI_API_KEY"],
            args.gemini_timeout,
        )
        print("\n--- Gemini Response ---")
        print(response)
        if args.speak_response:
            speak_response(
                response,
                args.tts_python,
                args.tts_inference_steps,
                args.tts_reference_audio,
            )
        return 0

    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())