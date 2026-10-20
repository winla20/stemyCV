# Demo recordings

Four original SteMy workflow recordings are included directly in Git. `python run_workflow.py` uses this directory by default. Total size is about 77 MB; no Git LFS setup is required.

To review their already processed results without a GPU or ML packages, run `python3 replay_demo.py` from the repository root. See [demo_results](../../demo_results/README.md).

| File | Inferred activity |
| --- | --- |
| `video-60_singular_display.mov` | Retrieve bottles and place them at the bench |
| `video-102_singular_display.mov` | Prepare cabinet, retrieve plate and handle microscope |
| `video-108_singular_display.mov` | Handle bottle, plate and pipette at the cabinet |
| `video-94_singular_display.mov` | Carry plate to incubator and close doors |

Descriptions are provisional visual interpretations. The videos retain their original audio and metadata. [manifest.json](manifest.json) records filenames, sizes and SHA-256 hashes for integrity checks; it contains no machine-specific paths. Generated review videos and predictions are written to `output/`, outside version control. These recordings are distinct from the public clips referenced in historical reports; no reuse license is asserted here.
