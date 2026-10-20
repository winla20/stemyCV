# Integration handoff

Start with the [README](../README.md). This is an offline Python worker plus a static review page. The four inputs are included in `data/demo`; run `python run_workflow.py` from your configured environment.

## Pipeline

| Stage | File | Purpose |
| --- | --- | --- |
| Entry point | `run_workflow.py` | Prepare inputs, extract, transcribe, build events, audit and build viewer |
| Video preparation | `prepare_workflow.py` | Create 30 fps H.264/AAC review copies and source hashes |
| Objects | `workflow_extract.py` | YOLOE masks, ByteTrack IDs and image-space motion |
| Speech | `workflow_transcribe.py` | Local Whisper transcript with uncertainty |
| Events | `workflow_events.py` | Geometric relations, inferred descriptions, visibility gaps and histories |
| Review UI | `workflow_viewer.html`, `build_workflow_viewer.py` | Video seeking, overlays, review and JSON export |
| Optional tracking | `workflow_temporal.py` | Fixed SAM plate-tracking comparison |

`workflow_config.json` describes these four demo clips, including inferred phase timings and expected objects. Earlier `run_poc.py`, `classify_actions.py`, `infer_interactions.py` and probe/compare scripts are experiments, not required for the current demo.

## Frontend/backend contract

Each recording writes a folder under `output/real_workflow/`:

| File | Use |
| --- | --- |
| `source_browser.mp4`, `source_manifest.json` | Playback, source/review hashes and timestamp basis |
| `frames.jsonl` | Frame timestamps, masks, boxes and temporary object IDs |
| `events.jsonl` | Candidate events; [schema](../schemas/event.schema.json), [example](../examples/event.json) |
| `objects.json`, `coverage.json` | Track histories and prediction-presence intervals |
| `transcript.json`, `insights.json` | Speech, limitations and unobserved outcomes |
| `extraction_summary.json` | Model settings and processing measurements |

Times are seconds in the normalized review video. Boxes are `[x1,y1,x2,y2]` in its pixels; polygon coordinates are normalized to `[0,1]`. Temporary track IDs do not establish persistent sample identity. Prediction presence is not recall; detector scores are not action probabilities.

Raw events are `unreviewed` / `pending_review`, with null action confidence and sample ID. Their `record_type` distinguishes automatic object relations, visual interpretations, visibility prompts and response-related speech. Display visibility prompts only at/after `trigger_s`. Preserve the source hash, evidence range and `basis` when storing events; scope IDs by run as well as event ID.

Reviews currently live in browser localStorage. The `stemy.workflow.review_export.v1` export separates `approved_memory_records`, `pending_candidates` and `rejected_candidates`, and includes object histories and unobserved responses. It adds `operator_review` and updates `memory_status` while retaining the original raw review status. It is a separate format from the raw-event schema. The backend must authenticate reviewers and enforce approval before memory ingestion.

## Next integration work

1. Add uploads, object storage and queued jobs with isolated working directories, retries and progress. The current runner uses a fixed output root and GPU 0.
2. Generalize recording names (`*_singular_display`), `.mov` discovery and the demo phase config before accepting arbitrary videos. Completed-run skips currently do not check configuration changes.
3. Serve authorized video URLs, paginated events and bounded frame windows. The demo viewer loads full recordings into JavaScript; manifests contain worker-local paths.
4. Persist reviews and sample IDs in the backend. Add a validated temporal action classifier; current rules do not verify aspiration, dispensing or experiment outcomes.

`audit_workflow.py` rewrites ASR/summary fields and assumes the fixed demo root. Separate that cleanup from read-only validation before using it in a service. Live intervention needs capture and a causal notification loop; microscopy and sample/well IDs are required for cell-response measurements.

CI checks syntax, the candidate contract, bundled-video hashes and monitor behavior. It does not run GPU inference, evaluate model accuracy or visually test the UI. Details and measured results remain in the [archive](archive/README.md).
