# Lab video segmentation proof of concept

## Integration entry points

Start with [SETUP.md](SETUP.md) for a fresh environment and lightweight checks, and [INTEGRATION_HANDOFF.md](INTEGRATION_HANDOFF.md) for worker stages, frontend/backend boundaries, artifact semantics and remaining integration work. The current raw candidate-event contract is [schemas/event.schema.json](schemas/event.schema.json), with a [synthetic example](examples/event.json).

Git tracks application source, configuration and documentation. Source videos, results, model weights, environments and the original brainstorm PDF remain local and are ignored. The pipeline is an offline prototype; upload/API/database services are not implemented. The earlier machine-specific setup below is retained as historical run context; use SETUP.md for handoff.

This project runs text-prompted object classification, instance segmentation, and tracking on a short laboratory video without drawing masks or labeling frames. The output is an annotated video, one JSON record per sampled frame, binary mask PNGs, and a summary of when each class was detected. All predictions are marked **unreviewed**.

The action-classification extension ranks short clips against protocol descriptions and exports timestamped **action candidates**. A separate tracked-box pass logs narrow interaction candidates such as a tube contacting a mixer. See [ACTION_PLAN.md](ACTION_PLAN.md) and the [distinct-action run](ACTION_COMPARISON_RUN.md) for evidence and limits.

## Source

The initial source is the 77-second manual pipetting and pooling clip from Lonardi et al., 2013: https://commons.wikimedia.org/wiki/File:Combinatorial-Pooling-Enables-Selective-Sequencing-of-the-Barley-Gene-Space-pcbi.1003010.s015.ogv . It is published under CC BY 2.5; retain attribution if sharing derived media.

## Run

The current Windows setup uses a project-local Python virtual environment with system PyTorch 2.8.0 and CUDA 12.6. To recreate that environment on this machine, from PowerShell in this directory:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install ultralytics==8.4.163 lap==0.5.13 openai-whisper==20250625 transformers==4.57.1
```

Then run:

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) '.yolo'
.\.venv\Scripts\python.exe run_poc.py
```

To test another clip:

```powershell
.\.venv\Scripts\python.exe run_poc.py --source path\to\clip.mp4 --output output\new_run
```

Add `--include-audio` for a clip with useful sound. The run command converts the overlay to browser-compatible H.264, includes AAC audio when requested, and builds `viewer.html` automatically when FFmpeg is available.

To add speech to a narrated video:

```powershell
.\.venv\Scripts\python.exe transcribe_poc.py --source data\source\microscope_first_minute.webm --output output\microscope_multimodal\transcript.json --end-s 60
.\.venv\Scripts\python.exe build_viewer.py output\microscope_multimodal
```

To repeat the close-up action trial, download [Pipette mixing technique](https://commons.wikimedia.org/wiki/File:Pipette_mixing_technique.ogv) to `data/source/pipette_mixing.ogv` and obtain the `microsoft/xclip-base-patch32` Hugging Face checkpoint in `models/xclip-base-patch32` (including its tokenizer and processor files). For example:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/xclip-base-patch32', local_dir='models/xclip-base-patch32', allow_patterns=['config.json','preprocessor_config.json','merges.txt','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json','model.safetensors'])"
```

Then:

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) 'Ultralytics'
.\.venv\Scripts\python.exe run_poc.py --source data\source\pipette_mixing.ogv --output output\pipette_mixing_yoloe26s --classes pipette "microcentrifuge tube" "gloved hand" --sample-fps 8 --confidence 0.2
.\.venv\Scripts\python.exe classify_actions.py --source data\source\pipette_mixing.ogv --output output\pipette_mixing_yoloe26s --detections output\pipette_mixing_yoloe26s\detections.jsonl --window-s 3 --stride-s 3
.\.venv\Scripts\python.exe build_viewer.py output\pipette_mixing_yoloe26s
```

Edit `action_labels.json` and `action_requirements.json` to match the procedure. `actions.jsonl` records every candidate, its relative score against the supplied descriptions, and whether required objects were visible often enough. Neither the score nor this coverage gate certifies that an action occurred. To update coverage after a new detector run, pass `--recheck-coverage` with the new `--detections` path. X-CLIP runs locally on this machine's 4 GB GPU. The pretrained checkpoint is approximately 786 MB.

Whisper needs `ffmpeg`. The script finds the copy bundled with Stremio on this machine, or one on `PATH`.

Defaults: YOLOE-26s-seg, prompts `pipette`, `rack of test tubes`, `microplate`, `person`; 2 sampled frames per second; 960-pixel model input; minimum model confidence 0.25. Downloads of the model checkpoint and prompt encoder are required on first use. Ultralytics is distributed under AGPL-3.0 or an enterprise license; check license terms before product integration.

## Output

- `overlay.mp4`: masks, object labels, track boxes, and timestamps on sampled frames. `overlay_browser.mp4` is H.264 for browser playback; the narrated run instead has `overlay_with_audio.mp4` with H.264 video and AAC audio.
- `detections.jsonl`: source frame number, timestamp, predicted class, score, track ID, bounding box, and mask path.
- `masks/*.png`: binary object masks at the source resolution.
- `summary.json`: model settings, counts, visible time intervals, and run time.
- `viewer.html`: a clickable timeline for the overlay video and, when available, the transcript.

These outputs show what the model predicted. They do not establish segmentation accuracy or confirm actions such as aspiration or liquid transfer.

The separate narrated microscope example uses the first minute of https://commons.wikimedia.org/wiki/File:Using_a_Microscope_-_Work_by_the_C3BC_consortium_-_CC-BY-3.0.webm (CC BY 3.0). Its `transcript.json` was generated by local Whisper `base.en`. The overlay video was muxed with the original audio so its visual and spoken records share timestamps.

## Run findings

### Real SteMy recordings

The local evidence workspace is `output/real_workflow/index.html`. It contains all four recordings from `C:\Users\happy\Downloads\stemy vids`, full-speed 30 fps review video with original audio, direct predictions on every normalized frame, an inferred procedure/location map, object histories, interaction candidates, visibility prompts, speech, and review/export controls. The data remains local. No segmentation training or cloud inference was used.

```powershell
.\.venv\Scripts\python.exe run_workflow.py --source-dir "C:\Users\happy\Downloads\stemy vids" --with-temporal
```

Completed extraction runs are reused. `workflow_extract.py --overwrite` explicitly regenerates predictions; use a fresh output root for a different source or benchmark. The supplied `workflow_config.json` is an inferred map for these particular clips, not an SOP. Change expectations and visibility windows there, then rerun `workflow_events.py` and `build_workflow_viewer.py` to update procedural checks. Changes to phase timing/location require consistent regeneration of frame annotations.

`frames.jsonl` contains compact masks, boxes, scores, temporary track IDs, image-space appearance/motion properties, and source timestamps. `events.jsonl`, `objects.json`, `coverage.json`, `insights.json`, `transcript.json`, and `source_manifest.json` provide machine-readable evidence. The source and review-copy SHA-256 hashes preserve provenance. Frame skipping is removed; predicted masks can still flicker.

The optional SAM 2.1 tiny comparison follows a detector-initialized plate in video-94. In the viewer, select that recording and choose the SAM mask mode. It is a separate temporal prediction layer, not confirmed identity, and does not overwrite YOLOE evidence or clear visibility alerts. See [Ultralytics SAM 2 video predictor](https://docs.ultralytics.com/models/sam-2) for the underlying interface.

`VisibilityMonitor.update()` is a causal component suitable for a later camera loop: after two seconds, it prompts if prediction presence is below 60%, and recovers at 80%. These thresholds are configurable demo policies, not validated accuracy targets. Current prompts are replay-only; no live notifications are delivered.

Procedure interpretations from visual inspection, automatic geometric relations, response-related speech, and model predictions retain separate provenance. Pipette/cap/lid detection, persistent vessel identity, aspiration/dispense classification, cell-resolution morphology and final outcome are unresolved. Approval in the viewer is saved locally; exported JSON separates approved records from pending and rejected candidates. It does not automatically write to SteMy memory.

Validation: `python -m unittest test_workflow_visibility.py` checks causal prompts, intermittent detections, wrong-track rejection, and expectation expiry. `audit_workflow.py` checks source hashes, frame/timestamp continuity, event bounds, pending-review status and JavaScript syntax. Browser automation cannot open the local file in this environment, so the interface has not been visually verified through that tool.

- Pooling clip: 1,722 video frames decoded from a header that reports 1,841. The 144 sampled frames produced 344 masks. The pipette was detected on 13 sampled frames, and the model frequently missed the thin moving tool. The audio produced an implausible transcription, preserved only under `output/diagnostics/` and excluded from the viewer.
- Narrated microscope excerpt: 120 sampled frames produced 186 masks. Whisper returned nine timestamped speech segments for the first minute. These are unreviewed and should be checked against the audio before entering persistent memory.
