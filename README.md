# Voice Transcription

Record audio from your microphone or transcribe an audio file locally with `whisper-cli`.

## Requirements

- Python 3
- `numpy` and `sounddevice` (`python -m pip install numpy sounddevice`)
- `whisper-cli` from whisper.cpp
- A whisper.cpp model file

## Run

```bash
python voice-transcribe.py --model /path/to/model.bin --duration 5
```

To transcribe an audio file instead, use:

```bash
python voice-transcribe.py --input /path/to/audio.wav
```

`--input` supports the audio formats accepted by your `whisper-cli` build. `--model` defaults to `~/whisper-models/ggml-small.en.bin`; microphone recording lasts 5 seconds by default and can be changed with `--duration`.
