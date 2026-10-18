"""Rank protocol-specific action descriptions against short video windows.

This emits unreviewed candidates. Similarity scores are relative to the supplied
descriptions, and do not prove that a physical action occurred.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("action_labels.json"))
    parser.add_argument("--requirements", type=Path, default=Path("action_requirements.json"))
    parser.add_argument("--model", default="models/xclip-base-patch32")
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=1.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recheck-coverage", action="store_true", help="Re-evaluate saved candidates with new detections; skip model inference")
    return parser.parse_args()


def read_detection_records(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def detection_evidence(records: list[dict], start: float, end: float) -> dict:
    in_window = [r for r in records if start <= r["time_s"] < end]
    by_label: dict[str, dict] = defaultdict(
        lambda: {"frames_detected": 0, "track_ids": set(), "max_detector_score": 0.0}
    )
    for frame in in_window:
        seen_labels = set()
        for obj in frame["objects"]:
            entry = by_label[obj["label"]]
            if obj["label"] not in seen_labels:
                entry["frames_detected"] += 1
                seen_labels.add(obj["label"])
            if obj["track_id"] is not None:
                entry["track_ids"].add(obj["track_id"])
            entry["max_detector_score"] = max(entry["max_detector_score"], obj["confidence"])
    return {
        "sampled_detector_frames": len(in_window),
        "by_label": {
            label: {
                "frames_detected": row["frames_detected"],
                "track_ids": sorted(row["track_ids"]),
                "max_detector_score": round(row["max_detector_score"], 4),
            }
            for label, row in by_label.items()
        },
    }


def coverage_gate(label: str, evidence: dict, requirements: dict, has_detections: bool) -> dict:
    if not has_detections:
        return {"status": "no_detector_evidence", "missing_object_groups": []}
    missing = []
    frame_count = evidence["sampled_detector_frames"]
    for group in requirements.get(label, []):
        observed = max(
            (evidence["by_label"].get(name, {}).get("frames_detected", 0) for name in group["any_of"]),
            default=0,
        )
        fraction = observed / frame_count if frame_count else 0.0
        if fraction < group["min_frame_fraction"]:
            missing.append({
                "any_of": group["any_of"],
                "observed_frame_fraction": round(fraction, 3),
                "required_frame_fraction": group["min_frame_fraction"],
            })
    return {
        "status": "insufficient_object_coverage" if missing else "object_supported_candidate",
        "missing_object_groups": missing,
    }


def get_frames(capture: cv2.VideoCapture, fps: float, start: float, end: float) -> list[np.ndarray]:
    # X-CLIP was trained using eight frames per clip.
    times = np.linspace(start, end, num=8, endpoint=False)
    frames = []
    for time_s in times:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(float(time_s) * fps))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode video frame near {time_s:.2f}s")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def windows(start: float, end: float, width: float, stride: float):
    t = start
    while t < end:
        stop = min(t + width, end)
        if stop - t < min(1.5, width / 2):
            break
        yield round(t, 3), round(stop, 3)
        t += stride


def main() -> None:
    args = arguments()
    if not args.source.is_file():
        raise SystemExit(f"Source video not found: {args.source}")
    if args.window_s <= 0 or args.stride_s <= 0:
        raise SystemExit("Window and stride must be positive")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    requirements = json.loads(args.requirements.read_text(encoding="utf-8"))
    if len(labels) < 2 or not all(isinstance(v, str) and v for v in labels.values()):
        raise SystemExit("Labels must be a JSON object with at least two text descriptions")
    detector_records = read_detection_records(args.detections)
    if args.recheck_coverage:
        target = args.output / "actions.jsonl"
        if not target.is_file():
            raise SystemExit(f"No saved action candidates at {target}")
        rows = []
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["source"] != args.source.as_posix():
                    raise SystemExit("Saved action candidate source does not match --source")
                evidence = detection_evidence(detector_records, row["start_s"], row["end_s"])
                row["detector_evidence"] = evidence
                row["coverage_gate"] = coverage_gate(
                    row["top_label"], evidence, requirements, args.detections is not None
                )
                rows.append(row)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_path = args.output / "action_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["object_requirements"] = requirements
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Rechecked coverage for {len(rows)} saved candidates in {target}")
        return
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot decode {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end_s = min(args.end_s if args.end_s is not None else duration, duration)
    if not 0 <= args.start_s < end_s:
        raise SystemExit("Invalid start/end time")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval().to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for start, end in windows(args.start_s, end_s, args.window_s, args.stride_s):
            frames = get_frames(capture, fps, start, end)
            # transformers 4.57 XCLIPProcessor only forwards text in its combined call.
            # Call each component explicitly so pixel_values is always present.
            inputs = dict(processor.tokenizer(list(labels.values()), return_tensors="pt", padding=True))
            inputs.update(processor.image_processor(frames, return_tensors="pt"))
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(**inputs).logits_per_video[0]
                else:
                    logits = model(**inputs).logits_per_video[0]
            similarities = logits.float().cpu().tolist()
            probabilities = torch.softmax(logits.float(), dim=0).cpu().tolist()
            ranked = sorted(
                (
                    {
                        "label": label,
                        "description": description,
                        "similarity": round(score, 4),
                        "relative_score": round(relative, 4),
                    }
                    for (label, description), score, relative in zip(
                        labels.items(), similarities, probabilities
                    )
                ),
                key=lambda row: row["similarity"],
                reverse=True,
            )
            evidence = detection_evidence(detector_records, start, end)
            row = {
                "record_type": "model_action_candidate",
                "review_status": "unreviewed",
                "source": args.source.as_posix(),
                "model": args.model,
                "start_s": start,
                "end_s": end,
                "top_label": ranked[0]["label"],
                "ranked_hypotheses": ranked,
                "detector_evidence": evidence,
                "coverage_gate": coverage_gate(
                    ranked[0]["label"],
                    evidence,
                    requirements,
                    args.detections is not None,
                ),
                "score_note": "Scores rank only the supplied descriptions; they are not calibrated probabilities or confirmation of an action.",
            }
            rows.append(row)
            print(f"{start:.1f}-{end:.1f}s: {ranked[0]['label']} ({ranked[0]['relative_score']:.3f} relative)", flush=True)
    finally:
        capture.release()

    target = args.output / "actions.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": args.source.as_posix(),
        "model": args.model,
        "labels": labels,
        "object_requirements": requirements,
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "windows_scored": len(rows),
        "device": device,
        "review_status": "unreviewed",
        "interpretation": "Relative video-text similarity, not calibrated action confidence or ground truth.",
    }
    (args.output / "action_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
