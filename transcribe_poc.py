"""Transcribe a video clip locally and save timestamped, unreviewed speech."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import whisper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-s", type=float, default=60)
    parser.add_argument("--model", default="base.en")
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Source video not found: {args.source}")
    if shutil.which("ffmpeg") is None:
        bundled = Path.home() / "AppData/Local/Programs/Stremio/ffmpeg.exe"
        if not bundled.is_file():
            raise SystemExit("ffmpeg is required on PATH")
        os.environ["PATH"] = str(bundled.parent) + os.pathsep + os.environ["PATH"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model = whisper.load_model(args.model, download_root=str(model_dir))
    result = model.transcribe(
        str(args.source),
        language="en",
        clip_timestamps=f"0,{args.end_s}",
        verbose=False,
    )
    output = {
        "source": args.source.as_posix(),
        "model": args.model,
        "clip_start_s": 0,
        "clip_end_s": args.end_s,
        "language": result["language"],
        "review_status": "unreviewed",
        "segments": [
            {
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "text": segment["text"].strip(),
                "avg_logprob": round(segment["avg_logprob"], 4),
                "no_speech_prob": round(segment["no_speech_prob"], 4),
            }
            for segment in result["segments"]
        ],
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(output['segments'])} segments to {args.output}")


if __name__ == "__main__":
    main()
