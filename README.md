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
with `--model`. The model is loaded once when the program starts and reused for
every turn in conversation mode.

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
python voice-transcribe.py --model small.en
```

To transcribe an audio file instead, use:

```bash
python voice-transcribe.py --input /path/to/audio.wav
```

`--input` supports common formats such as WAV, MP3, M4A, and FLAC. Faster-
Whisper uses PyAV, whose wheels bundle the required FFmpeg decoding libraries,
so a separate FFmpeg installation is normally unnecessary.
Microphone recording starts listening immediately and stops after speech is
followed by 0.7 seconds of silence. `--duration` is a 15-second safety maximum,
not a fixed recording length. In a noisy room, raise `--silence-threshold`; for
a quiet microphone, lower it. For example:

```bash
python voice-transcribe.py --silence-duration 0.6 --silence-threshold 0.012
```

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

Responses use the editable `USER_STYLE_PROFILE` in `voice-transcribe.py` on
every Gemini request. The default `mine` mode keeps replies short, casual, and
natural. The other response styles remain available:

```bash
python voice-transcribe.py --response-style mine
python voice-transcribe.py --response-style concise
python voice-transcribe.py --response-style conversational
python voice-transcribe.py --response-style detailed
```

Edit `user_style_examples.json` to add pairs showing how you would respond.
Each entry must contain non-empty `user` and `response` strings. These examples
are used only as style guidance in `mine` mode: they are not added to Gemini
conversation history and are not passed directly to speech output. To use a
different file, pass `--style-examples path/to/examples.json` or set
`USER_STYLE_EXAMPLES_FILE`. If the configured file does not exist, the app
continues without examples.

Audio transcription happens locally. The resulting transcript is sent to
Google's Gemini API, so do not use this with sensitive speech unless you are
comfortable sharing the transcript with Google. Free-tier availability and
usage limits depend on the model and can change; check the current
[Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).

## Speak Gemini's response

`--speak-response` now defaults to **RealtimeTTS + Kokoro** on both Windows
and macOS. Speech is generated locally with preset voices; no speech API key
or dedicated GPU is required. Gemini still uses your existing API key.

Gemini's text is streamed directly into RealtimeTTS. Kokoro starts synthesizing
after a short phrase instead of waiting for the complete response. The phrase
settings favor punctuation boundaries and force the first phrase after roughly
12 words, balancing startup latency against choppy audio.

### Windows / ASUS setup (PowerShell)

Use Python **3.11 or 3.12** for the optional speech stack. From the project
folder, create a virtual environment with Python 3.12 installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-tts.txt
```

If you have multiple registered Python versions, use
`py -3.12 -m venv .venv` instead.

If your existing `.venv` already uses Python 3.11/3.12, activate it and run
the two pip commands instead. Use native Windows Python for speaker playback.
If PowerShell blocks activation, you can run the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tts.txt
.\.venv\Scripts\python.exe -m tts_service.realtime "Hello, local voice output is working."
```

### macOS setup

Install PortAudio for playback and eSpeak NG for pronunciation fallback:

```bash
brew install python@3.12 portaudio espeak-ng
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tts.txt
```

### Test speech, then run the complete app

Test the voice first, without a Gemini API key or microphone:

```bash
python -m tts_service.realtime "Hello, local voice output is working."
```

The first run needs internet access to download the Kokoro model, selected
voice, and language resources. Later speech runs can use the cached files.
Allow the initial downloads to finish before judging startup speed.

Then run the existing recording/transcription/Gemini flow with speech:

```bash
python voice-transcribe.py --speak-response
```

For repeated turns, use conversation mode. Faster-Whisper and Kokoro each load
once at startup; Kokoro performs one muted warm-up, and both models stay in
memory until you quit. Gemini also receives the previous text turns as
conversation history:

```powershell
.\.venv\Scripts\python.exe voice-transcribe.py --speak-response --conversation
```

After each answer, press Enter to record another turn or type `q` to quit.
Press Ctrl+C while recording, generating, or speaking to cancel and release the
audio/model resources.

The app prints one-time Whisper/Kokoro model-loading measurements, then measures
each microphone turn from the last detected speech block through transcription,
the first Gemini text chunk, and the first audio written to the output device.
This end-of-speech measurement includes the silence-detection tail and reflects
the delay you actually experience. It also reports the interval between first
text and first audio to isolate speech-synthesis latency.

On Windows, set the Gemini key in PowerShell or in the existing `.env` file:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

Choose a preset voice and speed:

```bash
python voice-transcribe.py --speak-response --tts-voice am_michael --tts-speed 1.1
```

Some English voices are `af_heart`, `af_bella`, `am_michael`, `bf_emma`, and
`bm_george`. The prefix selects the language/accent; the default `af_heart`
is American English. Kokoro uses preset voices and does not support the
`--tts-reference-audio` cloning option. Press Ctrl+C to stop playback.

If you hear nothing, check your system's default speaker/headphone device.
If an error mentions eSpeak NG or its library, install
[eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases) for your platform
and reopen the terminal. If PyAudio fails to install on Windows, check that
you are using 64-bit Python 3.11/3.12; on macOS, check `brew install portaudio`.

### Existing VoxCPM2 voice cloning (Apple Silicon only)

The previous MLX Audio path is still available. If you have MLX Audio installed
in `tts_service/.venv312`, select it explicitly:

```bash
python voice-transcribe.py --duration 5 --speak-response --tts-engine voxcpm2
```

Speech generation uses 5 inference steps (down from VoxCPM2's default of 10)
to reduce generation time. Fewer steps may affect speech quality. Change this
with `--tts-inference-steps`; use 10 to restore VoxCPM2's default:

```bash
python voice-transcribe.py --duration 5 --speak-response --tts-engine voxcpm2 --tts-inference-steps 10
```

VoxCPM2 can generate speech without a reference clip. To clone a voice for a
response, pass the reference file each run:

```bash
python voice-transcribe.py --duration 5 --speak-response --tts-engine voxcpm2 \
  --tts-reference-audio "/path/to/reference.wav"
```

The reference file can be anywhere on your Mac. TTS runs in a separate process
for each response, so the model is loaded each time. Generated response WAV
files are kept in the project’s `generated_audio/` folder and played with
macOS `afplay`. These WAV-saving and inference-step options apply only to
VoxCPM2; the Kokoro path plays speech directly through RealtimeTTS.

## Development checks

The regression tests use fake audio/model backends, so they do not download
models or require an audio device:

```bash
python -m unittest discover -s tests -v
```
