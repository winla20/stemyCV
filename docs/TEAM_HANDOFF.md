# SteMy CV — frontend/backend handoff

**Repo:** https://github.com/winla20/stemyCV

Offline Python worker + review interface for lab videos: object masks/tracks, speech transcripts and timestamped event candidates. Four videos and their processed results are included.

## Run locally without a GPU

On macOS or Linux, with Python 3.12/3.13 installed:

```bash
git clone https://github.com/winla20/stemyCV.git
cd stemyCV
python3 replay_demo.py
```

Open **http://127.0.0.1:8765** to inspect saved overlays, seek the timeline, approve/reject candidates and export JSON. No packages or models need installing. On Windows use `python`; stop with Ctrl+C.

For **fresh inference**, follow the [platform setup](https://github.com/winla20/stemyCV#process-videos-locally), then run `python run_workflow.py --output output/my_run`. Vision automatically uses NVIDIA CUDA, Apple MPS or CPU; speech uses CUDA or CPU. `--device cpu` forces CPU vision. Apple Silicon is the supported Mac inference target; Intel Macs can replay the saved demo. Physical Mac/Linux inference still needs target-machine validation.

## What to integrate

| Owner | Work |
| --- | --- |
| Backend | Add authenticated uploads, video storage and queued worker jobs. Pass an isolated `--output` and matching `--config` per job. Store candidates, sample IDs and reviewer decisions; expose job status, artifacts and reviews through an API. |
| Frontend | Use the viewer as a reference for synchronized video/masks, timeline seeking and review controls. Develop against `demo_results/` now; replace static artifact access and browser localStorage with the backend API. |
| Shared | Agree job/event/review endpoints. Preserve evidence timestamps, source hashes, model/run versions and uncertainty. Only server-authorized, approved observations should enter persistent memory. |

Per recording: `frames.jsonl` (objects/masks), `events.jsonl` (candidates), `objects.json` (histories), `transcript.json`, source video and provenance. For example, the replay server exposes `/video-94_singular_display/events.jsonl`. See the [event schema](https://github.com/winla20/stemyCV/blob/main/schemas/event.schema.json) and [integration guide](https://github.com/winla20/stemyCV/blob/main/docs/HANDOFF.md).

**Limits:** this is a demo worker, not an upload API. New recordings require configuration. Procedure descriptions are inferred annotations; automatic events use geometric rules. Aspiration/dispense, persistent sample identity and cell health remain unverified. Live intervention and microscopy analysis are future work.
