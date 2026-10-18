# Segmentation proof of concept: first run

Run date: 2026-09-26. The pipeline used YOLOE-26s-seg with text prompts, ByteTrack for object IDs, and two sampled frames per second. It saved a mask for each detection, a JSONL frame record, an overlay video, a clickable timeline, and a summary. No hand-drawn masks or training labels were used.

| Clip | Decoded / sampled frames | Masks | Selected detections |
| --- | ---: | ---: | --- |
| [Manual pipetting and pooling](https://commons.wikimedia.org/wiki/File:Combinatorial-Pooling-Enables-Selective-Sequencing-of-the-Barley-Gene-Space-pcbi.1003010.s015.ogv) | 1,722 / 144 | 344 | Person 139, microplate 141, rack of test tubes 51, pipette 13 |
| [Narrated microscope demonstration](https://commons.wikimedia.org/wiki/File:Using_a_Microscope_-_Work_by_the_C3BC_consortium_-_CC-BY-3.0.webm), first minute | 1,799 / 120 | 186 | Person 118, microscope 45, hand 14, microscope slide 9 |

The microscope run also produced nine timestamped Whisper `base.en` speech segments. Its browser video contains the original audio and the mask overlay, so timeline clicks can be checked against both channels.

## Observed strengths and failures

- The pipeline runs end to end on the local RTX 3050 Ti laptop GPU and produces machine-readable records with source timestamps, model name, confidence, mask paths, and an explicit `unreviewed` status.
- On the pooling clip, the model usually segments the visible person and often recognizes the microplate and tube rack. It detects the thin moving pipette in only 13 sampled frames. Some broad masks overlap neighboring equipment; labels and track IDs need review before being treated as facts.
- On the microscope clip, the microscope is clear in the opening wide shot. The small glass slide is often missed in close shots, even when it is visible to a person.
- The pooling source's container header reports 1,841 frames, but only 1,722 frames decoded. This leaves about 72 seconds of usable visual frames. The clip's speech recognition output was implausible, so it is retained only under `output/diagnostics/` and excluded from the searchable timeline.

This run demonstrates ingestion, segmentation, tracking, timestamped speech, export, and evidence navigation. It does **not** measure model accuracy or verify actions such as aspiration, dispense, or completed protocol steps.

## What to try next

Use a clear 60-second, fixed-camera dry run with the pipette, tip, tube, and culture vessel filling more of the frame. Compare its predictions with this public-video baseline. If the pipette remains intermittent, try a targeted detector or one visual prompt per new object, then evaluate a small reviewed sample before feeding events into persistent memory.

## Vortex playback comparison: 2026-10-09

The previous vortex preview retained one frame in eight: 112 frames at 3.75 fps from a 30 fps source. Re-running the same YOLOE-26s model, prompts, 640-pixel input and 0.15 threshold on every decoded frame produced 894 frames at 30 fps and 3,380 masks. The new viewer is `output/vortex_shaker_fullfps/viewer.html`; `comparison.json` records both runs.

The full-frame processing loop took 66.50 seconds for 29.8 seconds of video (13.44 frames/second) on the local RTX 3050 Ti with 4 GB VRAM. This includes decoding, inference, tracking, mask export and overlay writing; it excludes model initialization and H.264 conversion. Offline 30 fps output works, but this export pipeline did not sustain live 30 fps.

Frame skipping explains jerky playback. It does not explain all mask instability: masks are predicted separately per frame, and ByteTrack links detection identities rather than propagating masks. The 15-second full-frame preview still mislabels the background water jug as a vortex mixer. Detection presence is not accuracy, and the full-frame run has not been scored against reviewed masks.

Next comparison: use a fixed, sharper close-up and a procedure-specific work area or visual exemplar to reduce ambiguity, then test detector-initialized SAM 2 video mask propagation. Benchmark this separately on the local GPU. Temporal stabilization must retain uncertainty and avoid carrying stale masks through occlusion; action events should continue to link to original video evidence.

## Real SteMy workflow extraction: 2026-10-09

Processed all four provided recordings locally: 13,243 timestamp-normalized frames, about 7 minutes 21 seconds of video. Original files in Downloads are unchanged. Review copies retain audio and 30 fps playback; every normalized video frame receives a direct YOLOE-26s-seg prediction. The main viewer is `output/real_workflow/index.html`.

| Recording | Inferred procedure | Analyzed frames | Evidence records |
| --- | --- | ---: | ---: |
| video-60 | Retrieve bottles from refrigerator and place at bench | 1,775 | 13 |
| video-102 | Prepare cabinet, retrieve plate, handle microscope | 5,400 | 16 |
| video-108 | Prepare bottle, plate and pipette at cabinet | 5,400 | 83 |
| video-94 | Carry plate to incubator and close doors | 668 | 9 |

The 121 records comprise 15 separately attributed visual interpretations, 78 automatic object-relation candidates, 23 replay visibility prompts, and 5 response-related speech candidates. The procedural descriptions and room/workstation labels were inferred from inspected frames, as requested; they are not automatic action-classifier outputs or an authoritative SOP. Geometric relations describe proximity and image-space motion, not verified liquid transfer. All records remain unreviewed until an operator reviews the evidence.

### Expected objects and histories

The configurable inventory covers gloves, reagent bottles, culture plates, pipettes/controllers, spray bottles, wipes, bottle caps, plate lids, equipment/location context and cells. Unsupported objects/properties are explicitly listed. Temporary histories include observations, gaps, detector scores, class consistency, mask area, red-pixel fraction for bottles, and approximate camera-corrected image-space motion. These properties do not establish volume, reagent identity, cellular growth or health. Sample IDs, age, freeze history and passage number remain unbound. Tracks fragment and reset at location changes; execution resumes also create an explicitly recorded identity boundary.

### What improved, and what failed

- Batched FP16 inference avoids frame dropping and separate per-mask PNG/video export overhead. On the same 4 GB RTX 3050 Ti, the resumed video-108 segment processed at 20.21 fps and the resumed video-102 segment at 30.28 fps. Video-60 processed end to end at 26.83 fps. The resumed timings exclude earlier interrupted work and initialization outside the processing loop; they are not whole-recording or live latency benchmarks. The host model and the GPU inference backend are separate objects; an early video-94 report captured the host device and is explicitly corrected to unknown backend reporting rather than claiming CPU inference.
- Baseline plate prediction presence was 81% during video-94 pickup, 13% during carry, and 12% during placement. At 13.5 seconds, visual inspection found a background shelf incorrectly masked as a plate. Presence checks can be satisfied by the wrong instance and are not accuracy measurements.
- SAM 2.1 tiny followed the initialized plate at 30 output fps over 421 frames, taking 89.6 seconds (4.7 processed fps; approximately 449 MB peak PyTorch CUDA allocations, excluding other GPU/driver usage). Carry-window mask presence increased from 12/90 YOLOE frames to 73/90 SAM frames. At 8 seconds, SAM retained the actual carried plate that YOLOE missed. It lost the target during occluded placement and did not reacquire it; this is a useful temporal comparison, not proof of continuous identity or a calibrated accuracy score. `video-94_singular_display/temporal_comparison.jpg` shows selected same-time views.
- Pipette prediction presence in video-108 was only 19% during the first selected work interval and 7% during the second. No shaft-endpoint/plate relation passed the persistence rule. A bottom-work-area crop at 960-pixel input with the specific `serological pipette` prompt produced detections in only 2/8 checked views and also produced broad incorrect plate masks. Increasing input size alone is insufficient here. Three visual exemplars improved one open-plate view but degraded other views; they were not adopted as the main detector.
- The rolling monitor flags poor evidence even when isolated detections appear. It uses a two-second window, a configurable 60% presence threshold and 80% recovery threshold. Prompts become active at their causal trigger timestamp, not at the first missing frame. No live operator notifications were sent.
- Local Whisper speech provides discussion of cell response/passage effects, but terms and timing need review. Repeated or implausible ASR is marked uncertain. Speech is kept as reported context; no biological outcome or chemical identity is inferred from it.

### Mapping to SteMy and the next work

The immediate output is a reviewable experiment evidence ledger: timestamped procedure candidates, object histories, visibility failures and speech linked to video and source hashes. The viewer supports approval/rejection, corrections and sample-ID assignment. Export separates approved memory records from pending and rejected candidates. No external memory store is written automatically.

The next detector work should prioritize gloves versus controller confusion, thin pipettes, caps/lids, and hard negative examples such as cabinet surfaces and incubator shelves. A small reviewed benchmark and a targeted lab-object detector are needed before claiming comprehensive coverage. Detector-initialized temporal segmentation is promising for carried vessels, with reinitialization and explicit identity uncertainty after occlusion. Readable vessel IDs/fiducials are required for persistent sample history. Action states should then distinguish cap/lid changes, source-to-destination tool movement, liquid transfer and well order using validated visual evidence and an SOP. Current dwell/visibility measurements cannot certify microvariations in volume or timing of aspiration/dispense.

For live intervention, connect the shared causal monitor to capture and validated procedure context, bind the active vessel/tool rather than accepting any background instance, and measure prompt precision and latency. For experiment response, add timestamped microscopy of the same samples/wells, pixel calibration, comparable fields/timepoints and recorded outcomes. Bench footage here cannot measure morphology, growth rate or differentiation; these fields remain explicitly unobserved.

Integrity checks passed for source hashes, frame continuity, timestamps, event ranges and pending-review state. Three causal-monitor tests passed; viewer JavaScript syntax passed. Segmentation/action accuracy has not been measured against human-reviewed ground truth. Browser automation refused the local file protocol, so UI rendering was not verified through it.
