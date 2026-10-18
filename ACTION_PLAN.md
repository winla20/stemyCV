# Stemy action classification: selected route

## Decision

Keep the existing YOLOE-26s-seg object pipeline and add **tracked object relations and procedure-specific state rules** as the primary path to an observable action log. X-CLIP remains a secondary **action candidate** layer: it compares eight frames from a short interval with procedure descriptions, then writes rankings and timestamps to `actions.jsonl`. `infer_interactions.py` is a first narrow example of the primary path for tube/mixer contact. The viewer links both kinds of candidate back to video. Neither a text score nor box proximity proves that liquid moved.

This is a local, no-training starting point. It runs on the RTX 3050 Ti with 4 GB of VRAM. The X-CLIP checkpoint is 786 MB on disk; no rented GPU was needed for this trial. A larger video language model is premature while detector coverage and the observable action vocabulary are still unresolved.

## What the first test showed

The [3.2-second close-up pipette-mixing clip](https://commons.wikimedia.org/wiki/File:Pipette_mixing_technique.ogv) is clearer than the earlier wide pooling clip. It is CC BY-SA 4.0; credit Cirosantilli2 and preserve the license when sharing derivatives. With the initial wording, X-CLIP ranked `pipette_mixing` first (relative score 0.391), ahead of `liquid_transfer` (0.291) and `pipette_positioning` (0.245). With the shorter seven-label comparison wording, `pipette_positioning` narrowly edged `pipette_mixing` (0.319 vs 0.314). This prompt sensitivity reinforces that the scores remain hypotheses. The video does not clearly show the fluid level or plunger state.

YOLOE at 8 sampled frames/s detected `pipette` in 3/24 frames and `microcentrifuge tube` in 0/24. Changing the tube prompt to `plastic tube` detected it in 23/24 frames, while the pipette remained at 2/24. Prompt choice helped the tube, but the pipette track is still too sparse for a reliable interaction rule. The action model can rank the clip independently of detections, but that does not establish which tracked item changed.

As a negative control, X-CLIP ranked `tube_handling` first (0.275) in a 6–9 s microscope clip that showed a person beside a microscope, with no tube detection. `no_relevant_action` (0.267) and `microscope_adjustment` (0.247) were close behind. This is a concrete false candidate: a highest-ranked text description must never be treated as a confirmed lab event. `action_requirements.json` now checks required object coverage and marks unsupported candidates; even a passed coverage check is only a prerequisite, not action proof.

A distinct [vortex-shaker run](ACTION_COMPARISON_RUN.md) sharpened this decision: X-CLIP mislabeled all three vortexing windows as pipette-related actions despite simpler wording, while tracked tube/mixer geometry yielded two narrow, reviewable contact intervals. X-CLIP is therefore useful for exploration here, but not yet reliable enough to lead the event log.

## Path to a procedural event log

1. **Define one procedure and its observable vocabulary.** For pipetting: pipette body/tip, source tube, destination tube, hand, vessel IDs, and work area. Record expected steps and allowed variants. Exclude unrelated movement.
2. **Measure object visibility before action claims.** Test prompt variants and a clear fixed camera view. If thin tools remain missed, try a visual prompt, targeted detector, or a larger YOLOE checkpoint on a small reviewed set. Track IDs are temporary visual identities; use readable labels or fiducials to bind a tube to a persistent sample ID.
3. **Infer object relations, then use text scores as supporting hypotheses.** Track geometry contributes facts such as “pipette tip approached tube A, then tube B” or “tube contacted mixer.” X-CLIP can rank proposed descriptions, but a weak or unsupported ranking must resolve to `unknown`. Keep independent evidence for each claim. Do not equate proximity with aspiration or dispense.
4. **Resolve to a protocol event only with sufficient evidence.** A procedure-aware state machine can check order, location, and allowed variants. Store start/end time, action, sample IDs, observed properties, source video range, model scores, detector coverage, and review status. If a required object is not visible, mark `unobserved` and ask the operator to show the work area during a live run.
5. **Evaluate before memory or intervention.** Review a small set of short positive, negative, and variant clips; measure action misses, false events, timing, and whether the right sample was attached. Only then calibrate event confidence or train a small classifier on frozen video embeddings, as in [IBM's lab-action pipeline](https://github.com/IBM/lab-activity-recognition).

For a later benchmark, [IBM's egocentric lab-action dataset](https://zenodo.org/records/14235875) provides action labels, but its video archive is 33.6 GB. A [7-second close-up liquid-transfer clip](https://www.pexels.com/video/a-person-putting-liquid-substance-on-test-tubes-7230308/) offers a distinct public action to compare with mixing. The most useful Stemy test would be a consented dry run with colored liquid, a fixed view of both tubes and the pipette tip, 8–15 or more frames/s, visible sample IDs, and intentionally recorded `no action` intervals. Public footage tests recognition; it cannot test sample identity, live prompts, or full protocol history.

Cell morphology and experiment response need a separate, time-aligned microscopy pipeline with outcome measurements. Bench video is the primary audit evidence for visible steps, but it is not ground truth for hidden quantities, sterility, freeze history, or biological outcome.

## Alternatives considered

| Approach | Fit now | Main limitation |
| --- | --- | --- |
| Detector tracks plus geometry/rules | Interpretable for visible approach/contact/placement | Current pipette coverage is too sparse; liquid movement may be invisible. |
| X-CLIP video/text similarity | Runs locally, no action training labels, supports procedure wording | Misclassified the clear vortexing example; use only as a secondary hypothesis. |
| X3D/VideoMAE fine-tuned action model | Plausible after an action dataset exists | General-action weights do not define Stemy protocol steps. |
| Large video language model/cloud API | Flexible narratives | Higher compute/privacy burden and no guarantee of event correctness. |

References: [Ultralytics YOLOE documentation](https://docs.ultralytics.com/models/yoloe), [X-CLIP checkpoint](https://huggingface.co/microsoft/xclip-base-patch32), [X-CLIP inference documentation](https://huggingface.co/docs/transformers/model_doc/xclip).
