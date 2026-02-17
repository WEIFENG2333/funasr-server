"""VAD + SenseVoiceSmall + timestamps demo.

Shows the full workflow:
1. Install runtime (first time only)
2. Start the server
3. Load SenseVoiceSmall with fsmn-vad for long audio support
4. Run inference with per-character timestamps
5. Print formatted results with performance metrics

Usage:
    python examples/asr_with_timestamps.py [audio_file]

If no audio file is provided, uses the test fixture audio.
"""

import os
import sys
import time
from pathlib import Path

from funasr_server import FunASR


def format_time(ms: int) -> str:
    """Format milliseconds as mm:ss.mmm."""
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def format_size(nbytes: int) -> str:
    """Format byte count as human-readable size."""
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f}KB"
    return f"{nbytes / 1024 / 1024:.1f}MB"


def main():
    # -- Determine audio file --------------------------------------------------

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = str(
            Path(__file__).parent.parent / "tests" / "fixtures" / "asr_example.wav"
        )

    if not Path(audio_path).exists():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    audio_size = os.path.getsize(audio_path)

    print(f"Audio : {audio_path}")
    print(f"Size  : {format_size(audio_size)}")
    print()

    # -- Setup -----------------------------------------------------------------

    with FunASR() as asr:
        t0 = time.perf_counter()
        asr.ensure_installed()
        install_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        asr.start()
        start_time = time.perf_counter() - t0

        # -- Load model --------------------------------------------------------

        t0 = time.perf_counter()
        model = asr.load_model("SenseVoiceSmall", vad_model="fsmn-vad")
        load_time = time.perf_counter() - t0

        # -- Inference ---------------------------------------------------------

        t0 = time.perf_counter()
        results = model.infer(audio=audio_path, use_itn=True, output_timestamp=True)
        infer_time = time.perf_counter() - t0

        # -- Print results -----------------------------------------------------

        for item in results:
            text = item.get("text", "")
            timestamps = item.get("timestamp", [])
            words = item.get("words", [])

            print(f"Text: {text}")
            print()

            if timestamps and words:
                print("Timestamps:")
                for word, (start, end) in zip(words, timestamps):
                    print(f"  {format_time(start)} - {format_time(end)}  {word}")
                print()

        # -- Performance summary -----------------------------------------------

        health = asr.health()

        print("-" * 40)
        print("Performance")
        print("-" * 40)
        print(f"  Install     : {format_duration(install_time)}")
        print(f"  Server start: {format_duration(start_time)}")
        print(f"  Model load  : {format_duration(load_time)}")
        print(f"  Inference   : {format_duration(infer_time)}")
        print()
        print(f"  Device      : {health.get('device', 'unknown')}")
        print(f"  CUDA        : {health.get('cuda_available', False)}")

        model.unload()


if __name__ == "__main__":
    main()
