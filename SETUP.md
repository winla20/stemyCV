# Setup and verification

Run commands from the repository root. Use Python 3.12 or 3.13, Node.js 22+ for viewer syntax checks, and FFmpeg on PATH. The recorded inference baseline was PyTorch 2.8.0 with CUDA 12.6 on an NVIDIA RTX 3050 Ti (4 GB). Inference scripts currently select GPU 0. Full model execution is not covered by CPU-only CI.

## Check the handoff without models or footage

```powershell
py -3.12 -m venv .venv-dev
.\.venv-dev\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-dev\Scripts\python.exe scripts/check_handoff.py
.\.venv-dev\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

On Linux, use `python3.12 -m venv .venv-dev` and `.venv-dev/bin/python` for the same commands. The synthetic event fixture supports backend contract work without access to the original recordings. Node must be on PATH.

## Recreate an inference environment

Create a fresh environment instead of copying `.venv` or relying on global packages. The original environment used `--system-site-packages` and has a machine-specific Python path; it is not portable.

```powershell
py -3.12 -m venv .venv-inference
.\.venv-inference\Scripts\python.exe -m pip install --upgrade pip
.\.venv-inference\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-inference\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv-inference\Scripts\python.exe -m pip check
.\.venv-inference\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
node --version
```

This recipe recreates direct dependencies from the demo, but a clean GPU install has not been verified during the Git handoff. Select a compatible CUDA build for the target GPU/driver using the official PyTorch installation guidance. Preserve a resolved environment manifest after validating the deployment host.

## Obtain model assets

Weights are excluded from Git. First-use downloads need network access; bake them into a deployment image or controlled model store for offline workers.

```powershell
.\.venv-inference\Scripts\python.exe -c "from workflow_models import load_detector; load_detector()"
.\.venv-inference\Scripts\python.exe -c "import whisper; whisper.load_model('base.en', download_root='models')"
```

The first command obtains `yoloe-26s-seg.pt` and the text-prompt encoder through Ultralytics; it may install/download additional prompt dependencies. The optional plate comparison requires the official [SAM 2.1 tiny checkpoint](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt) saved as `models/sam2.1_t.pt`. X-CLIP is an earlier, separate experiment; see the README for its download command. Model downloads and local caches are ignored by Git.

## Run the existing four-recording demonstration

Obtain the recordings separately and use their original filenames. `workflow_config.json` contains an inferred map for those names. For a new dataset, adapt the assumptions described in [INTEGRATION_HANDOFF.md](INTEGRATION_HANDOFF.md) first.

```powershell
.\.venv-inference\Scripts\python.exe run_workflow.py --source-dir "C:\path\to\stemy vids"
# Optional, fixed plate-tracking comparison:
.\.venv-inference\Scripts\python.exe workflow_temporal.py
.\.venv-inference\Scripts\python.exe build_workflow_viewer.py
```

Open `output/real_workflow/index.html` in a browser. Outputs are local and ignored. `run_workflow.py` reuses completed runs; it does not verify changed configuration before skipping them. Use a new job directory and the individual stage CLIs for integration experiments, following the handoff notes. The audit step requires originals to remain accessible at manifest paths and can rewrite ASR/summary fields.

To validate a generated candidate file without modifying it:

```powershell
.\.venv-dev\Scripts\python.exe scripts/check_handoff.py --events output/real_workflow/video-108_singular_display/events.jsonl
```
