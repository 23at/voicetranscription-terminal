# Voice Transcription

Record audio from your microphone or transcribe an audio file locally with
Faster-Whisper, then send the transcript to Gemini and display its response.
Transcription works on Windows and macOS.

## Requirements

- Python 3.9 or newer
- The Python packages in `requirements.txt`
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

Create a virtual environment and install the dependencies:

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Windows, CTranslate2 also requires the Microsoft Visual C++ runtime, which
is already present on most systems.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model

Transcription runs locally through Faster-Whisper. The default model is
`small.en`. It is downloaded automatically on the first run and cached for
later use. Select another model name or a local Faster-Whisper model directory
with `--model`.

CPU transcription with `int8` computation is the cross-platform default. A
Windows machine with a correctly configured NVIDIA CUDA environment can use:

```powershell
python voice-transcribe.py --whisper-device cuda --whisper-compute-type float16
```

Use `--language auto` for automatic language detection, or pass a language
code such as `en`, `es`, or `fr`. A multilingual model such as `small` is
required for languages other than English.

## Run

```bash
python voice-transcribe.py --model small.en --duration 5
```

To transcribe an audio file instead, use:

```bash
python voice-transcribe.py --input /path/to/audio.wav
```

`--input` supports common formats such as WAV, MP3, M4A, and FLAC. Faster-
Whisper uses PyAV, whose wheels bundle the required FFmpeg decoding libraries,
so a separate FFmpeg installation is normally unnecessary.
Microphone recording lasts 5 seconds by default and can be changed with
`--duration`.

## Gemini response

Set your API key in the environment before running the script:

```bash
export GEMINI_API_KEY="your-api-key"
```

The script uses `gemini-3.8-flash` by default. Choose a different Gemini model
with `--gemini-model` or set the `GEMINI_MODEL` environment variable:

```bash
python voice-transcribe.py --input /path/to/audio.wav --gemini-model gemini-3.8-flash
```

Gemini requests wait up to 180 seconds by default. Increase the limit with
`--gemini-timeout`, for example `--gemini-timeout 300`.

Audio transcription happens locally. The resulting transcript is sent to
Google's Gemini API, so do not use this with sensitive speech unless you are
comfortable sharing the transcript with Google. Free-tier availability and
usage limits depend on the model and can change; check the current
[Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).

## Speak Gemini's response

On an Apple Silicon Mac with MLX Audio installed in `tts_service/.venv312`,
run the transcription script with `--speak-response`:

```bash
python voice-transcribe.py --duration 5 --speak-response
```

Speech generation uses 5 inference steps (down from VoxCPM2's default of 10)
to reduce generation time. Fewer steps may affect speech quality. Change this
with `--tts-inference-steps`; use 10 to restore VoxCPM2's default:

```bash
python voice-transcribe.py --duration 5 --speak-response --tts-inference-steps 10
```

VoxCPM2 can generate speech without a reference clip. To clone a voice for a
response, pass the reference file each run:

```bash
python voice-transcribe.py --duration 5 --speak-response \
  --tts-reference-audio "/path/to/reference.wav"
```

The reference file can be anywhere on your Mac. TTS runs in a separate process
for each response, so the model is loaded each time. Generated response WAV
files are kept in the project’s `generated_audio/` folder and played with
macOS `afplay`.
