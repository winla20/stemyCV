# Distinct close-up action run: vortex mixing

Source: [Vortex Shaker.webm](https://commons.wikimedia.org/wiki/File:Vortex_Shaker.webm), 30 seconds, 360 × 640, by Nattapat.aur under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The clip visibly shows a gloved hand holding a tube of green liquid against a vortex shaker. It gives a different tool and motion from the earlier [pipette-mixing clip](https://commons.wikimedia.org/wiki/File:Pipette_mixing_technique.ogv).

## Run and result

YOLOE-26s-seg processed 112 sampled frames at approximately 4 fps, with prompts `vortex mixer`, `test tube`, and `gloved hand`. It detected those classes in 87, 54, and 109 distinct sampled frames respectively. At some times it also mislabeled the background water jug as a mixer, so class presence alone remains insufficient evidence.

X-CLIP scored three 3-second windows (5–8, 11–14, 17–20 s) against the same seven action descriptions used on the pipette clip. It ranked `liquid_transfer`, `liquid_transfer`, and `pipette_mixing` first. `vortex_mixing` received only 0.037–0.071 of the relative score across the windows. A rewrite with shorter, parallel descriptions did not fix this confusion. These are unreviewed rankings, and the object-coverage gate rejected the pipette-related candidates because no pipette was detected.

`infer_interactions.py` instead examined tracked boxes for a tube bottom repeatedly near the top of the mixer. It produced two distinct, reviewable interaction intervals: **10.133–13.067 s** (11 sampled frames) and **14.400–16.534 s** (8 sampled frames). A gloved hand was near the tube in every supporting frame. The emitted relation is `tube_bottom_near_vortex_mixer_top`, with an action hypothesis of `vortex_mixing`. It does not prove that the motor was running or that the liquid was mixed.

The [viewer](output/vortex_shaker_yoloe26s/viewer.html) links model rankings and interaction intervals to the annotated video. Raw candidates are in [actions.jsonl](output/vortex_shaker_yoloe26s/actions.jsonl) and [interactions.jsonl](output/vortex_shaker_yoloe26s/interactions.jsonl). The first X-CLIP wording is preserved in `actions_initial.jsonl`.

## Decision from this test

For a procedural log, use tracked object interactions and task-specific state rules as the primary path for **observable** events. Keep X-CLIP as an exploratory candidate source, never as an automatic action assertion. The pipette remains an unresolved detector target, so liquid aspiration, dispense, and transfer cannot yet be logged reliably from the current videos. The next decisive test is a controlled dry run with the pipette tip, both vessels, and liquid level visible, plus reviewed positive and negative clips. Those clips can support a small trained action head or a better targeted detector.

## Reproduce

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) 'Ultralytics'
.\.venv\Scripts\python.exe run_poc.py --source data\source\vortex_shaker.webm --output output\vortex_shaker_yoloe26s --classes "vortex mixer" "test tube" "gloved hand" --sample-fps 4 --confidence 0.15 --image-size 640
.\.venv\Scripts\python.exe classify_actions.py --source data\source\vortex_shaker.webm --output output\vortex_shaker_yoloe26s --labels action_labels_benchmark.json --detections output\vortex_shaker_yoloe26s\detections.jsonl --start-s 5 --end-s 20 --window-s 3 --stride-s 6
.\.venv\Scripts\python.exe infer_interactions.py --detections output\vortex_shaker_yoloe26s\detections.jsonl --output output\vortex_shaker_yoloe26s
.\.venv\Scripts\python.exe build_viewer.py output\vortex_shaker_yoloe26s
```
