# Voice Transcription

Record audio from your microphone and transcribe it locally with `whisper-cli`.

## Requirements

- Python 3
- `numpy` and `sounddevice` (`python -m pip install numpy sounddevice`)
- `whisper-cli` from whisper.cpp
- A whisper.cpp model file

## Run

```bash
python voice-transcribe.py --model /path/to/model.bin --duration 5
```

`--model` defaults to `~/whisper-models/ggml-small.en.bin`. `--duration` is the recording length in seconds and defaults to `5`.
