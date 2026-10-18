"""Infer narrow, reviewable tube/mixer contact intervals from tracked boxes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def box_gap(a: list[float], b: list[float]) -> float:
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return math.hypot(dx, dy)


def tube_on_mixer(tube: dict, mixer: dict) -> bool:
    tx1, _, tx2, ty2 = tube["bbox_xyxy"]
    mx1, my1, mx2, my2 = mixer["bbox_xyxy"]
    tube_bottom_x = (tx1 + tx2) / 2
    top_band = max(18.0, 0.2 * (my2 - my1))
    return mx1 <= tube_bottom_x <= mx2 and abs(ty2 - my1) <= top_band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tube-label", default="test tube")
    parser.add_argument("--mixer-label", default="vortex mixer")
    parser.add_argument("--hand-label", default="gloved hand")
    parser.add_argument("--min-frames", type=int, default=3)
    args = parser.parse_args()

    with args.detections.open(encoding="utf-8") as handle:
        frames = [json.loads(line) for line in handle if line.strip()]
    if len(frames) < 2:
        raise SystemExit("Need at least two sampled frames")
    source = frames[0]["source"]
    if any(frame["source"] != source for frame in frames):
        raise SystemExit("Mixed video sources in detection records")
    sample_step = median(b["time_s"] - a["time_s"] for a, b in zip(frames, frames[1:]))
    observations: dict[tuple[int | None, int | None], list[dict]] = defaultdict(list)

    for frame in frames:
        objects = frame["objects"]
        tubes = [o for o in objects if o["label"] == args.tube_label]
        mixers = [o for o in objects if o["label"] == args.mixer_label]
        hands = [o for o in objects if o["label"] == args.hand_label]
        for tube in tubes:
            for mixer in mixers:
                if not tube_on_mixer(tube, mixer):
                    continue
                key = (tube["track_id"], mixer["track_id"])
                near_hand = any(
                    box_gap(tube["bbox_xyxy"], hand["bbox_xyxy"]) <= 20
                    for hand in hands
                )
                observations[key].append(
                    {
                        "time_s": frame["time_s"],
                        "frame_index": frame["frame_index"],
                        "hand_near_tube": near_hand,
                        "tube_detector_score": tube["confidence"],
                        "mixer_detector_score": mixer["confidence"],
                    }
                )

    events = []
    for (tube_id, mixer_id), points in observations.items():
        groups = []
        current = []
        for point in points:
            if current and point["time_s"] - current[-1]["time_s"] > sample_step * 1.5:
                groups.append(current)
                current = []
            current.append(point)
        if current:
            groups.append(current)
        for group in groups:
            if len(group) < args.min_frames:
                continue
            events.append(
                {
                    "record_type": "model_interaction_candidate",
                    "review_status": "unreviewed",
                    "source": source,
                    "relation": "tube_bottom_near_vortex_mixer_top",
                    "action_hypothesis": "vortex_mixing",
                    "start_s": group[0]["time_s"],
                    "end_s": round(group[-1]["time_s"] + sample_step, 3),
                    "tube_track_id": tube_id,
                    "mixer_track_id": mixer_id,
                    "supporting_frames": len(group),
                    "hand_near_tube_frames": sum(p["hand_near_tube"] for p in group),
                    "median_tube_detector_score": round(median(p["tube_detector_score"] for p in group), 4),
                    "median_mixer_detector_score": round(median(p["mixer_detector_score"] for p in group), 4),
                    "source_frame_indices": [p["frame_index"] for p in group],
                    "interpretation": "A tube box repeatedly meets a mixer box near its top. This does not prove the mixer was running or that liquid was mixed.",
                }
            )
    # A second box can be assigned to the same tube for a few frames. Keep the
    # longer enclosing interaction instead of logging a duplicate action.
    events.sort(key=lambda event: event["end_s"] - event["start_s"], reverse=True)
    unique_events = []
    for event in events:
        if any(
            event["mixer_track_id"] == saved["mixer_track_id"]
            and saved["start_s"] <= event["start_s"]
            and event["end_s"] <= saved["end_s"]
            for saved in unique_events
        ):
            continue
        unique_events.append(event)
    events = sorted(unique_events, key=lambda event: event["start_s"])
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "interactions.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    print(json.dumps({"interaction_candidates": len(events), "output": str(target)}))


if __name__ == "__main__":
    main()
