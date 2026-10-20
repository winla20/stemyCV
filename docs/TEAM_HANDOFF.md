# SteMy CV — frontend/backend handoff

**Repo:** https://github.com/winla20/stemyCV

This prototype turns lab video into object tracks, speech transcripts and timestamped event candidates. It includes four demo videos in `data/demo/` and a static review interface. The current pipeline runs locally; the upload service, API and database still need to be built.

**Run the demo**

Requires Python 3.12/3.13, an NVIDIA CUDA GPU, and FFmpeg and Node.js on PATH. From PowerShell:

```powershell
git clone https://github.com/winla20/stemyCV.git
cd stemyCV
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe run_workflow.py
```

On Linux, create the environment with `python3` and use `.venv/bin/python`. Adjust the CUDA build if required by the target GPU. First use downloads the models. The recorded baseline used a 4 GB RTX 3050 Ti; a clean installation on other hardware remains to be verified.

Open `output/real_workflow/index.html` to inspect video, masks, event timelines and review/export controls. Add `--skip-transcription` to omit speech. The optional SAM comparison is documented in the [README](https://github.com/winla20/stemyCV/blob/main/README.md).

**What to integrate**

| Owner | Work |
| --- | --- |
| Backend | Add authenticated uploads, video storage and queued GPU jobs with status/retries. Give each job an isolated output directory; generalize the current fixed paths, recording names and demo configuration. Store candidates, sample IDs and reviewer decisions. |
| Frontend | Reuse the viewer as a reference for video playback, synchronized overlays, timeline seeking and approve/reject controls. Load artifacts through the API and persist reviews on the backend; current reviews use browser localStorage. |
| Shared | Agree job/event/review endpoints. Preserve evidence timestamps, source hashes, model/run versions and uncertainty. Only server-authorized, approved observations should enter persistent memory. |

Each recording produces `frames.jsonl` (objects/masks), `events.jsonl` (candidates), `objects.json` (histories), `transcript.json`, and source/provenance metadata. Start with the [event schema](https://github.com/winla20/stemyCV/blob/main/schemas/event.schema.json), [synthetic example](https://github.com/winla20/stemyCV/blob/main/examples/event.json), and [technical integration guide](https://github.com/winla20/stemyCV/blob/main/docs/HANDOFF.md).

**Current limits:** procedural descriptions are inferred for these demo clips; automatic events use geometric rules. Aspiration/dispense, persistent sample identity and cell health are not verified. New recordings require configuration work. Live intervention and microscopy analysis are future work.
