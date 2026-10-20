# SteMy CV

Local lab-video extraction with YOLOE object segmentation/tracking, Whisper speech transcription, and timestamped procedure candidates. Includes four demo recordings (about 77 MB) in [`data/demo`](data/demo). No separate video download is needed after cloning.

## Run the demo

Requirements: Python 3.12 or 3.13, an NVIDIA CUDA GPU, FFmpeg and Node.js on PATH. The recorded baseline used an RTX 3050 Ti with 4 GB VRAM. Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe run_workflow.py
```

On Linux, use `python3` to create the environment and `.venv/bin/python` for the remaining commands. Select a compatible PyTorch/CUDA build if your GPU requires one. These dependencies reflect the demo environment; a clean GPU installation has not been retested during this cleanup.

The first run downloads YOLOE and its prompt dependencies, plus Whisper `base.en`. Outputs go to `output/real_workflow/`. Open **`output/real_workflow/index.html`** in your browser to play video, inspect masks and events, review candidates, and export JSON. Completed recordings are reused on subsequent runs.

To skip speech transcription, add `--skip-transcription`. `--source-dir PATH` overrides the included videos, but new filenames/procedures also require a matching `workflow_config.json`.

## Optional SAM comparison

The existing SAM experiment follows one plate in `video-94`; it is not full-workflow action recognition.

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt -OutFile models/sam2.1_t.pt
.\.venv\Scripts\python.exe run_workflow.py --with-temporal
```

## Handoff

- **[Integration guide](docs/HANDOFF.md):** entry points, output files and frontend/backend work.
- **[Event schema](schemas/event.schema.json)** and **[synthetic example](examples/event.json)** for integration.
- **[Earlier reports](docs/archive/README.md):** results, limitations and previous experiments.

Objects and geometric relations are automatic predictions. Procedure descriptions in `workflow_config.json` were inferred from the demo footage. Actions remain unreviewed candidates; aspiration/dispense, persistent sample identity and cell health are not verified. Reviews are stored in browser localStorage until exported. Upload services, APIs, a database and live notifications are future work.

## Check changes

These checks need only `requirements-dev.txt` and Node.js; no GPU or model download:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/check_handoff.py
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

Only the four named demo videos are committed. Model weights, generated outputs, other recordings and environments are ignored. Ultralytics uses AGPL-3.0 or Enterprise licensing; resolve the applicable model/dependency terms before product deployment. No repository-wide license has been selected.
