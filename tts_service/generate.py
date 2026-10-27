import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load


MODEL_NAME = "mlx-community/VoxCPM2-4bit"
DEFAULT_INFERENCE_STEPS = 9


def generate_voice(text, reference_audio, output_path, inference_steps):
    start = time.perf_counter()
    model = load(MODEL_NAME)
    print(f"Model load: {time.perf_counter() - start:.1f}s", flush=True)

    generation_args = {
        "text": text,
        "inference_timesteps": inference_steps,
    }
    if reference_audio:
        generation_args["ref_audio"] = reference_audio

    start = time.perf_counter()
    result = next(model.generate(**generation_args))
    generation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    audio = np.array(result.audio)
    sf.write(output_path, audio, model.sample_rate)
    save_seconds = time.perf_counter() - start

    print(f"Audio generation: {generation_seconds:.1f}s", flush=True)
    print(f"Generated: {output_path}")
    print(f"WAV save: {save_seconds:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with MLX Audio VoxCPM2."
    )
    parser.add_argument("text", nargs="?", help="Text to speak; read from stdin if omitted.")
    parser.add_argument("--text-stdin", action="store_true", help="Read text to speak from stdin.")
    parser.add_argument(
        "--reference-audio",
        help="Optional reference audio for voice cloning for this generation.",
    )
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=DEFAULT_INFERENCE_STEPS,
        help=(
            f"VoxCPM2 inference steps (default: {DEFAULT_INFERENCE_STEPS}; "
            "higher can improve quality but take longer)."
        ),
    )
    args = parser.parse_args()

    if args.text_stdin and args.text is not None:
        parser.error("provide text as a positional argument or through --text-stdin, not both")
    text = sys.stdin.read() if args.text_stdin or args.text is None else args.text
    if not text.strip():
        parser.error("text to speak must not be empty")
    if args.inference_steps < 1:
        parser.error("--inference-steps must be at least 1")

    reference_audio = args.reference_audio
    if reference_audio and not os.path.isfile(reference_audio):
        parser.error(f"reference audio file not found: {reference_audio}")

    generate_voice(text, reference_audio, args.output, args.inference_steps)


if __name__ == "__main__":
    main()
