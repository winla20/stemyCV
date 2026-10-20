# Stemy visual observation proof of concept

## What was implemented

A local, no-labeling demo processes laboratory video with YOLOE-26s-seg to classify visible objects, draw segmentation masks, and assign track IDs. It samples roughly two frames per second and saves an annotated video, per-frame JSONL records, mask images, and a clickable evidence timeline. A second clip adds Whisper speech transcription with timestamps. The two runs are described in [RUN_REPORT.md](RUN_REPORT.md).

## How it functions

The user supplies a video and a short list of object names, such as *pipette* or *microscope*. The model predicts where those objects appear; the tracker links detections across sampled frames. Each record retains its source timestamp, class, confidence, track ID, bounding box, mask path, model name, and `unreviewed` status. Where speech is useful, Whisper adds timestamped transcript segments. Timeline items seek back to the supporting moment in the video.

## How it maps to Stemy

This is an early implementation of the brainstorm's **Observing** role: converting visual and spoken lab activity into structured, inspectable evidence. Its records could eventually feed Stemy's memory bank and help Bench Support, Troubleshooting, Data Analysis, and protocol-oriented agents retrieve what was seen or said during a run. They are observations for review, not confirmed protocol events or instructions for an agent to act on.

## Open decisions and next work

1. **Define the target workflow and vocabulary.** Choose one stem-cell procedure, its objects and observable steps, and what Stemy should answer about it. Record a clear, consented dry-run video with synchronized narration and protocol metadata.
2. **Measure and improve reliability.** The public-video model often missed the thin moving pipette and small microscope slide. Review a small sample of predictions, measure misses and false labels, then compare better prompts, camera placement, targeted detection, or a cell-specific model for microscopy.
3. **Turn observations into reviewed events.** Specify how visual tracks, speech, timing, and protocol context combine into candidate actions; keep uncertainty and source links. Require review before asserting completion, quantities, deviations, or other persistent memories.
4. **Integrate with Stemy.** Define an event schema and retrieval interface for the memory bank, then test questions such as “show the evidence for this step” across sessions. Decide retention, access, and model-license terms before using real lab footage in a product.
