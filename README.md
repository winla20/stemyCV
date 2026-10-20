# SteMy CV

Local lab-video extraction with YOLOE object segmentation/tracking, Whisper speech transcription, and timestamped procedure candidates.

## Review the saved demo — no GPU or model installation

Requires Git, Python 3.12/3.13 and a browser on macOS, Linux or Windows:

```bash
git clone https://github.com/winla20/stemyCV.git
cd stemyCV
python3 replay_demo.py
```

On Windows, use `python` instead of `python3`. Open **http://127.0.0.1:8765**. Stop with Ctrl+C; use `--port 8766` if the port is occupied.

All four original videos (`data/demo`, 77 MB) and processed results (`demo_results`, 186 MB) are included in Git. The viewer displays saved masks, tracks, transcripts, event candidates and the SAM comparison, with review/export controls. Replay requires **no pip packages, FFmpeg, Node.js, model downloads or inference**. It builds the viewer locally and serves only the processed-demo folder on localhost. See the [bundle notes](demo_results/README.md).

## Process videos locally

Use **Python 3.12** for the most consistent setup (the current Windows environment also uses 3.13). Install FFmpeg first: `brew install ffmpeg` on macOS, or `sudo apt-get install ffmpeg python3-venv` on Ubuntu/Debian. On Windows, install FFmpeg and add it to PATH. `FFMPEG_BINARY` can override its location.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **one** matching PyTorch build ([official version instructions](https://pytorch.org/get-started/previous-versions/#v280)):

| Computer | Command |
| --- | --- |
| Apple Silicon Mac (M1 or newer) | `python -m pip install torch==2.8.0 torchvision==0.23.0` |
| Linux CPU | `python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu` |
| Linux NVIDIA with a CUDA 12.6-compatible driver | `python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126` |

Then:

```bash
python -m pip install -r requirements.txt
python run_workflow.py
```

The pinned inference environment targets Apple Silicon Macs and x86-64 Linux/Windows. Intel Macs can use saved-result replay; fresh inference there requires a separate compatible dependency set.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_workflow.py
```

For a CPU-only Windows setup, replace `cu126` with `cpu`.

### Device selection and outputs

- Vision automatically selects **CUDA → Apple MPS → CPU**. CUDA uses FP16 and batches of 8; MPS/CPU use FP32 and batches of 1. Override with `--device cpu`, `--device mps`, or `--device cuda:0`; lower `--batch-size` if needed.
- Whisper selects CUDA when available, otherwise CPU. Override separately with `--speech-device cpu`. Whisper does not use MPS in this pipeline.
- MPS operator fallback is enabled before importing PyTorch. If an operation still fails on a Mac, use `--device cpu --output output/cpu_run`.
- First use downloads the model weights and prompt dependencies. Processing then runs locally. CPU inference is much slower than replay; start integration work with the saved results.
- New outputs go to `output/real_workflow/`. Open its `index.html` to review them. Use `--output output/my_run` for an isolated run. Completed recordings are reused; **use a fresh output directory when changing configuration or device settings**.
- `--skip-transcription` omits speech. `--source-dir PATH --config PATH` selects other inputs/configuration; new recordings still require matching names and procedure configuration.
- The demo config explicitly interprets these exports as SDR BT.709 for consistent browser color. Existing bundled review copies have corrected metadata, with verified unchanged decoded frames/audio/timing. For other footage, choose its color policy deliberately; genuine HDR needs tone mapping, not an SDR tag override. See the [bundle notes](demo_results/README.md).
- `--with-temporal` adds the SAM 2.1 tiny plate comparison for `video-94`; its weights download on first use. It follows one plate and does not classify actions across the workflow.

**Verification:** real YOLOE, Whisper and SAM smoke inference passed on CPU and automatically selected CUDA on the Windows development machine (66 detector frames and 12 SAM frames per run). Automated tests cover device selection, output routing, replay HTTP seeking and event contracts. CI is configured for model-free checks on macOS, Linux and Windows. Physical Mac/MPS and Linux model inference have not been verified here; this is not a fully locked dependency environment.

## Handoff

- **[Short team handoff](docs/TEAM_HANDOFF.md):** shareable run instructions and frontend/backend responsibilities.
- **[Integration guide](docs/HANDOFF.md):** entry points, output files and API integration.
- **[Event schema](schemas/event.schema.json)** and **[synthetic example](examples/event.json)**.
- **[Earlier reports](docs/archive/README.md):** results and previous experiments.

Objects and geometric relations are automatic predictions. Procedure descriptions in `workflow_config.json` were inferred from the demo footage. Actions remain unreviewed candidates; aspiration/dispense, persistent sample identity and cell health are not verified. Reviews stay in browser localStorage until exported. Upload services, APIs, a database and live notifications are future work.

## Check changes

These checks need only the development dependencies and Node.js, with no GPU or model download:

```bash
python -m pip install -r requirements-dev.txt
python scripts/check_handoff.py
python replay_demo.py --build-only
python -m unittest discover -p "test_*.py" -v
```

Model weights, new outputs, other recordings and environments are ignored. Ultralytics uses AGPL-3.0 or Enterprise licensing; resolve the applicable model/dependency terms before product deployment. No repository-wide license has been selected.
