# Voice Transcription

Record audio from your microphone or transcribe an audio file locally with
`whisper-cli`, then send the transcript to Gemini and display its response.

## Requirements

- Python 3
- `numpy` and `sounddevice` (`python -m pip install numpy sounddevice`)
- `whisper-cli` from whisper.cpp
- A whisper.cpp model file
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Model

Transcription runs locally through `whisper-cli` using a Whisper model from
whisper.cpp. By default, the script uses `~/whisper-models/ggml-small.en.bin`,
the English-only Small model. Select a different model file with `--model`.

## Run

```bash
python voice-transcribe.py --model /path/to/model.bin --duration 5
```

To transcribe an audio file instead, use:

```bash
python voice-transcribe.py --input /path/to/audio.wav
```

`--input` supports the audio formats accepted by your `whisper-cli` build.
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
