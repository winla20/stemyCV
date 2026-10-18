"""Automatic lab-video segmentation demo using a text-prompted YOLOE model."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE


DEFAULT_CLASSES = ["pipette", "rack of test tubes", "microplate", "person"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/source/pooling.ogv"))
    parser.add_argument("--output", type=Path, default=Path("output/pooling_yoloe26s"))
    parser.add_argument("--model", default="yoloe-26s-seg.pt")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--include-audio", action="store_true")
    return parser.parse_args()


def compress_intervals(times: list[float], step: float) -> list[dict]:
    if not times:
        return []
    intervals = []
    start = last = times[0]
    for current in times[1:]:
        if current - last > step * 1.5:
            intervals.append({"start_s": round(start, 3), "end_s": round(last + step, 3)})
            start = current
        last = current
    intervals.append({"start_s": round(start, 3), "end_s": round(last + step, 3)})
    return intervals


def find_ffmpeg() -> str | None:
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    bundled = Path.home() / "AppData/Local/Programs/Stremio/ffmpeg.exe"
    return str(bundled) if bundled.is_file() else None


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Source video not found: {args.source}")
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output / "masks"
    mask_dir.mkdir(exist_ok=True)
    video = cv2.VideoCapture(str(args.source))
    if not video.isOpened():
        raise SystemExit(f"Cannot decode {args.source}")
    source_fps = video.get(cv2.CAP_PROP_FPS)
    frame_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0:
        raise SystemExit("Source frame rate is unavailable")
    stride = max(1, round(source_fps / args.sample_fps))
    actual_sample_fps = source_fps / stride
    sample_step = 1 / actual_sample_fps

    model = YOLOE(args.model)
    model.set_classes(args.classes)
    writer = cv2.VideoWriter(
        str(args.output / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_sample_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit("Cannot create overlay.mp4")

    counts: Counter[str] = Counter()
    present_at: dict[str, list[float]] = {label: [] for label in args.classes}
    sampled = 0
    masks_written = 0
    started = time.perf_counter()
    jsonl_path = args.output / "detections.jsonl"
    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            frame_index = 0
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                if frame_index % stride:
                    frame_index += 1
                    continue
                t = frame_index / source_fps
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=args.image_size,
                    conf=args.confidence,
                    device=args.device,
                    retina_masks=True,
                    verbose=False,
                )[0]
                objects = []
                ids = result.boxes.id
                masks = result.masks.data if result.masks is not None else None
                for i, box in enumerate(result.boxes):
                    label = result.names[int(box.cls.item())]
                    counts[label] += 1
                    mask_path = None
                    if masks is not None and i < len(masks):
                        mask = (masks[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
                        if mask.shape != (height, width):
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                        mask_path = f"masks/frame_{frame_index:06d}_object_{i:02d}.png"
                        if not cv2.imwrite(str(args.output / mask_path), mask):
                            raise RuntimeError(f"Failed to write {mask_path}")
                        masks_written += 1
                    objects.append(
                        {
                            "label": label,
                            "confidence": round(float(box.conf.item()), 4),
                            "track_id": int(ids[i].item()) if ids is not None else None,
                            "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                            "mask_path": mask_path,
                        }
                    )
                for label in {item["label"] for item in objects}:
                    present_at[label].append(t)
                record = {
                    "record_type": "model_frame_predictions",
                    "review_status": "unreviewed",
                    "model": args.model,
                    "source": args.source.as_posix(),
                    "frame_index": frame_index,
                    "time_s": round(t, 3),
                    "objects": objects,
                }
                jsonl.write(json.dumps(record) + "\n")
                overlay = result.plot(labels=True, boxes=True, masks=True)
                cv2.putText(
                    overlay,
                    f"{t:05.1f}s | model output, unreviewed",
                    (12, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(overlay)
                sampled += 1
                if sampled % 25 == 0:
                    print(f"Processed {sampled} sampled frames / {t:.1f}s", flush=True)
                frame_index += 1
    finally:
        video.release()
        writer.release()

    summary = {
        "source": args.source.as_posix(),
        "model": args.model,
        "classes": args.classes,
        "confidence_threshold": args.confidence,
        "source_frames": frame_total,
        "source_fps": source_fps,
        "decoded_frames": frame_index,
        "decoded_duration_s": round(frame_index / source_fps, 3),
        "sampled_frames": sampled,
        "sampled_fps": actual_sample_fps,
        "masks_written": masks_written,
        "detections_by_class": dict(counts),
        "visible_intervals_by_class": {
            label: compress_intervals(times, sample_step)
            for label, times in present_at.items()
        },
        "wall_time_s": round(time.perf_counter() - started, 2),
        "review_status": "unreviewed",
        "limitations": [
            "Labels, masks and tracks are model predictions, not verified annotations.",
            "Visible intervals mean the model detected the object; they do not prove a protocol action.",
            "Object identities may reset or switch after occlusion.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        target = args.output / (
            "overlay_with_audio.mp4" if args.include_audio else "overlay_browser.mp4"
        )
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.output / "overlay.mp4")]
        if args.include_audio:
            command += ["-i", str(args.source), "-map", "0:v:0", "-map", "1:a:0"]
        command += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]
        if args.include_audio:
            command += ["-c:a", "aac", "-shortest"]
        command += ["-movflags", "+faststart", str(target)]
        subprocess.run(command, check=True)
    else:
        print("ffmpeg unavailable: browser-compatible video was not produced", file=sys.stderr)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("build_viewer.py")), str(args.output)],
        check=True,
    )
    print(json.dumps({key: summary[key] for key in ("sampled_frames", "masks_written", "detections_by_class", "wall_time_s")}, indent=2))


if __name__ == "__main__":
    main()
