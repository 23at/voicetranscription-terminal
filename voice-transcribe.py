import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Optional
import urllib.error
import urllib.parse
import urllib.request
import wave


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


@dataclass
class RecordingResult:
    audio: object
    speech_end_time: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe it locally with Faster-Whisper."
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to an audio file to transcribe instead of recording from the microphone.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get("WHISPER_MODEL", "small.en"),
        help=(
            "Faster-Whisper model name or local model directory "
            "(default: small.en or WHISPER_MODEL)."
        ),
    )
    parser.add_argument(
        "--whisper-device",
        choices=("cpu", "cuda", "auto"),
        default=os.environ.get("WHISPER_DEVICE", "cpu"),
        help="Device used for transcription (default: cpu or WHISPER_DEVICE).",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
        help=(
            "CTranslate2 compute type (default: int8 or WHISPER_COMPUTE_TYPE). "
            "For an NVIDIA GPU, try float16."
        ),
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("WHISPER_LANGUAGE", "en"),
        help=(
            "Spoken language code (default: en or WHISPER_LANGUAGE); "
            "use 'auto' to detect it."
        ),
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=15.0,
        help="Maximum recording duration in seconds (default: 15).",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.7,
        help="Stop after this many seconds of silence following speech (default: 0.7).",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.015,
        help="RMS level considered speech; lower this for quiet microphones (default: 0.015).",
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
        help="Speak Gemini's response locally (Kokoro by default).",
    )
    parser.add_argument(
        "--conversation",
        action="store_true",
        help=(
            "Keep models loaded and record repeated conversation turns until "
            "you quit or press Ctrl+C."
        ),
    )
    parser.add_argument(
        "--tts-engine",
        choices=("kokoro", "voxcpm2"),
        default="kokoro",
        help="Local speech engine: kokoro (Windows/macOS) or voxcpm2 (Apple Silicon Mac).",
    )
    parser.add_argument(
        "--tts-voice",
        default="af_heart",
        help="Kokoro voice ID (default: af_heart, American English).",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="Kokoro speaking speed; must be positive (default: 1.0).",
    )
    parser.add_argument(
        "--tts-reference-audio",
        help="Optional voice-cloning sample for --tts-engine voxcpm2 only.",
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


def ensure_dependencies():
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create an API key in Google AI Studio and set it in your environment."
        )


def record_audio(
    duration_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    silence_duration: float = 0.7,
    silence_threshold: float = 0.015,
    block_duration: float = 0.1,
    pre_roll_duration: float = 0.3,
):
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Microphone recording requires sounddevice. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    if duration_seconds <= 0:
        raise RuntimeError("--duration must be greater than zero.")
    if silence_duration <= 0:
        raise RuntimeError("--silence-duration must be greater than zero.")
    if not math.isfinite(silence_threshold) or silence_threshold <= 0:
        raise RuntimeError("--silence-threshold must be a finite number greater than zero.")

    block_frames = max(1, int(block_duration * sample_rate))
    max_blocks = max(1, int(math.ceil(duration_seconds / block_duration)))
    silence_blocks_needed = max(1, int(math.ceil(silence_duration / block_duration)))
    pre_roll_blocks = max(1, int(math.ceil(pre_roll_duration / block_duration)))
    pre_roll = deque(maxlen=pre_roll_blocks)
    captured = []
    speech_started = False
    silent_blocks = 0
    speech_end_time = None

    print(
        f"Listening (stops after {silence_duration:g}s of silence; "
        f"{duration_seconds:g}s maximum)..."
    )
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_frames,
    ) as stream:
        for _ in range(max_blocks):
            block, overflowed = stream.read(block_frames)
            if overflowed:
                print("Warning: microphone input overflowed; some audio may be missing.")
            mono = np.asarray(block[:, 0], dtype=np.float32).copy()
            rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))

            if not speech_started:
                pre_roll.append(mono)
                if rms < silence_threshold:
                    continue
                speech_started = True
                captured.extend(pre_roll)
                pre_roll.clear()
                speech_end_time = time.perf_counter()
                continue

            captured.append(mono)
            if rms >= silence_threshold:
                speech_end_time = time.perf_counter()
                silent_blocks = 0
            else:
                silent_blocks += 1
                if silent_blocks >= silence_blocks_needed:
                    print("Speech ended; transcribing...")
                    break

    if not speech_started or not captured or speech_end_time is None:
        raise RuntimeError(
            "No speech was detected. Check the microphone or lower --silence-threshold."
        )

    return RecordingResult(audio=np.concatenate(captured), speech_end_time=speech_end_time)


def save_wav(audio, wav_path: str, sample_rate: int = SAMPLE_RATE):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Microphone recording requires numpy. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

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


class FasterWhisperSession:
    """Load Faster-Whisper once and reuse it for every conversation turn."""

    def __init__(self, model_name: str, device: str, compute_type: str):
        try:
            from faster_whisper import WhisperModel

            load_started = time.perf_counter()
            print(f"Loading Faster-Whisper model '{model_name}' on {device}...")
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            self.model_load_seconds = time.perf_counter() - load_started
            print(f"[Latency] Whisper model load: {self.model_load_seconds:.2f}s")
        except Exception as exc:
            raise RuntimeError(f"Could not load Faster-Whisper: {exc}") from exc

    def transcribe(self, audio_path: str, language: str):
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=None if language.lower() == "auto" else language,
                beam_size=5,
            )
            transcript_text = "".join(segment.text for segment in segments).strip()
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

        if not transcript_text:
            raise RuntimeError("Transcription completed but no speech was detected.")

        detected_language = getattr(info, "language", None)
        if language.lower() == "auto" and detected_language:
            print(f"Detected language: {detected_language}")
        return transcript_text


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


def _gemini_text_from_event(event):
    if not isinstance(event, dict):
        return ""
    candidates = event.get("candidates", [])
    if not candidates:
        return ""
    return "".join(
        part.get("text", "")
        for part in candidates[0].get("content", {}).get("parts", [])
        if isinstance(part, dict)
    )


def iter_gemini_response(
    transcript: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    history=None,
):
    """Yield Gemini text deltas from the REST SSE streaming endpoint."""
    model_path = urllib.parse.quote(model, safe="-")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_path}:streamGenerateContent?alt=sse"
    )
    contents = list(history or [])
    contents.append({"role": "user", "parts": [{"text": transcript}]})
    payload = json.dumps({"contents": contents}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    received_text = False
    feedback = None
    data_lines = []

    def parse_event():
        if not data_lines:
            return None
        data = "\n".join(data_lines)
        data_lines.clear()
        if data == "[DONE]":
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini API returned an invalid streaming event.") from exc

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line:
                    continue

                event = parse_event()
                if event is None:
                    continue
                feedback = event.get("promptFeedback", feedback)
                text = _gemini_text_from_event(event)
                if text:
                    received_text = True
                    yield text

            event = parse_event()
            if event is not None:
                feedback = event.get("promptFeedback", feedback)
                text = _gemini_text_from_event(event)
                if text:
                    received_text = True
                    yield text
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

    if not received_text:
        block_reason = (feedback or {}).get("blockReason", "no response candidate")
        raise RuntimeError(f"Gemini did not return a response: {block_reason}")


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


def capture_transcript(args, whisper_session):
    if args.input:
        audio_path = os.path.expanduser(args.input)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return whisper_session.transcribe(audio_path, args.language), None

    recording = record_audio(
        args.duration,
        silence_duration=args.silence_duration,
        silence_threshold=args.silence_threshold,
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        save_wav(recording.audio, wav_path)
        transcript = whisper_session.transcribe(
            wav_path,
            args.language,
        )
        return transcript, recording.speech_end_time
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def run_turn(args, history, whisper_session, kokoro_session=None):
    transcript, speech_end_time = capture_transcript(args, whisper_session)
    if speech_end_time is not None:
        transcript_seconds = time.perf_counter() - speech_end_time
        print(f"[Latency] End of speech to transcript: {transcript_seconds:.2f}s")
    print("\n--- Transcript ---")
    print(transcript)

    response_chunks = []
    request_started = time.perf_counter()
    latency_started = speech_end_time or request_started
    latency_origin = "end of speech" if speech_end_time is not None else "Gemini request"
    first_text_seen = False
    first_text_seconds = None
    chunks = iter_gemini_response(
        transcript,
        args.gemini_model,
        os.environ["GEMINI_API_KEY"],
        args.gemini_timeout,
        history=history,
    )

    print("\n--- Gemini Response (streaming) ---")

    def observed_chunks():
        nonlocal first_text_seen, first_text_seconds
        for chunk in chunks:
            if not first_text_seen:
                first_text_seen = True
                first_text_seconds = time.perf_counter() - latency_started
                print(
                    f"[Latency] {latency_origin.capitalize()} to first Gemini text: "
                    f"{first_text_seconds:.2f}s"
                )
            response_chunks.append(chunk)
            print(chunk, end="", flush=True)
            yield chunk

    if kokoro_session is not None:
        timing = kokoro_session.speak_stream(
            observed_chunks(),
            latency_started=latency_started,
            latency_origin=latency_origin,
        )
        if timing.first_audio_seconds is not None and first_text_seconds is not None:
            print(
                f"[Latency] Gemini-to-audio synthesis: "
                f"{timing.first_audio_seconds - first_text_seconds:.2f}s"
            )
    else:
        for _ in observed_chunks():
            pass

    response = "".join(response_chunks).strip()
    print()
    if not response:
        raise RuntimeError("Gemini returned an empty response.")

    if args.speak_response and args.tts_engine == "voxcpm2":
        speak_response(
            response,
            args.tts_python,
            args.tts_inference_steps,
            args.tts_reference_audio,
        )

    history.extend(
        [
            {"role": "user", "parts": [{"text": transcript}]},
            {"role": "model", "parts": [{"text": response}]},
        ]
    )


def main():
    args = parse_args()
    kokoro_session = None
    whisper_session = None

    try:
        if args.gemini_timeout <= 0:
            raise RuntimeError("--gemini-timeout must be greater than zero.")
        if args.conversation and args.input:
            raise RuntimeError("--conversation records microphone turns and cannot use --input.")
        if args.speak_response and args.tts_engine == "kokoro":
            from tts_service.realtime import KokoroSession, validate_options

            validate_options(args.tts_voice, args.tts_speed)
            if args.tts_reference_audio:
                raise RuntimeError(
                    "Kokoro uses preset voices. Use --tts-voice, or select "
                    "--tts-engine voxcpm2 on an Apple Silicon Mac for voice cloning."
                )

        ensure_dependencies()

        whisper_session = FasterWhisperSession(
            args.model,
            args.whisper_device,
            args.whisper_compute_type,
        )

        if args.speak_response and args.tts_engine == "kokoro":
            kokoro_session = KokoroSession(
                voice=args.tts_voice,
                speed=args.tts_speed,
            )

        history = []
        while True:
            run_turn(
                args,
                history,
                whisper_session,
                kokoro_session=kokoro_session,
            )
            if not args.conversation:
                break
            answer = input("\nPress Enter for another turn, or type q to quit: ").strip()
            if answer.lower() in {"q", "quit", "exit"}:
                break
        return 0

    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if kokoro_session is not None:
            kokoro_session.close()


if __name__ == "__main__":
    raise SystemExit(main())
