# SteMy video extraction: integration handoff

This repository is an offline Python extraction worker and a static evidence-review prototype. It processes local lab recordings with YOLOE, optional SAM video tracking, Whisper transcription, and deterministic relation/visibility rules. It does not yet contain an HTTP API, upload service, authenticated frontend, database, job scheduler, trained lab-action classifier, or live camera service.

Start with [SETUP.md](SETUP.md). The existing capabilities and observed failures are recorded in [RUN_REPORT.md](RUN_REPORT.md). No model weights, source footage, generated results, virtual environments, or original brainstorm PDF are committed. `workflow_config.json` and the reports do include the provisional interpretation of the four demonstration clips; replace this configuration for a new experiment.

## Existing components and integration ownership

| Component | Existing entry point | Integration responsibility |
| --- | --- | --- |
| Review-video preparation | `prepare_workflow.py SOURCE_DIR --output ROOT` | Backend: stage input, verify hash, normalize video timestamps, retain audio |
| Detection and temporary tracking | `workflow_extract.py --root ROOT --config CONFIG --resume` | GPU worker: ordered frames, explicit model/config version, separate output per job |
| Transcription | `workflow_transcribe.py --root ROOT` | CPU worker: speech candidates and ASR uncertainty |
| Relations, visibility and histories | `workflow_events.py --root ROOT --config CONFIG` | Backend: turn frame evidence into candidates, preserve evidence basis |
| Review workspace | `build_workflow_viewer.py --root ROOT --config CONFIG` | Frontend reference for video seeking, overlays, candidate review and export |
| Temporal segmentation comparison | `workflow_temporal.py --folder FOLDER` | Experimental GPU stage, currently a single plate and fixed early interval |
| Causal visibility policy | `VisibilityMonitor.update(time_s, objects, expectations)` | Future live service: provide validated procedure context and active identities |
| Integrity audit | `audit_workflow.py` | Demo maintenance script; see side effects below |

`run_workflow.py --source-dir SOURCE_DIR` sequences the demo at the fixed `output/real_workflow` root. It assumes repository working directory, GPU 0 and the four configured recording names. It is not a general multi-tenant job runner. Use the individual stages while designing the backend adapter.

## Files the backend can ingest

| Artifact per recording | Contents / semantics |
| --- | --- |
| `source_manifest.json` | Original and normalized-video SHA-256, local paths, capture-time metadata estimate, timestamp basis |
| `extraction_summary.json` | Model/settings, dimensions, frame count, throughput and timing scope; records may cover a resumed segment |
| `frames.jsonl` | One record per normalized frame: `frame_index`, `time_s`, objects, phase/location and source hash |
| `events.jsonl` | Raw candidates conforming to [schemas/event.schema.json](schemas/event.schema.json) |
| `objects.json` | Temporary track histories; sample IDs and unobserved historical properties remain null |
| `coverage.json` | Class prediction presence in expected intervals; this is not measured recall |
| `transcript.json` | Speech segments, timestamps, ASR evidence status and review state |
| `insights.json` | Candidate counts, unsupported properties and explicitly unobserved experiment responses |
| `temporal_frames.jsonl` | Optional SAM masks; separate evidence stream, not verified identity |

Coordinates: boxes are `[x1,y1,x2,y2]` in normalized review-video pixels. `polygons_normalized` holds polygon paths of `[x/width,y/height]` points. `time_s` refers to the 30 fps review copy; use its video for seeking. A capture-time metadata estimate is not a synchronized experiment clock. Track IDs are temporary and may change across occlusion, zones, restarts and recordings.

The schema formalizes the current raw event envelope, not every artifact or a complete REST API. [examples/event.json](examples/event.json) is synthetic and contains no actual experiment evidence. Validate raw candidates with `python scripts/check_handoff.py --events PATH_TO_EVENTS_JSONL`.

### Evidence and review semantics

- `object_relation_candidate`: automatic geometric proximity/motion rules. Does not establish contact or liquid transfer.
- `visual_interpretation_candidate`: assistant inspection of timestamped frames, stored in the demo config. Not an automatic action-classifier output.
- `visibility_prompt_candidate`: causal detector-evidence gap. Display a prompt only at or after `trigger_s`; start time includes its preceding observation window.
- `reported_response_candidate`: keyword-selected ASR discussion, not measured cellular response.

Raw events have `review_status: unreviewed`, `memory_status: pending_review`, `event_confidence: null`, and `persistent_sample_id: null`. Detector confidence is not action probability. Preserve method-specific fields and original evidence when enriching a record. Candidate event IDs depend on source and content; rerunning after changes may produce new IDs. Scope database uniqueness by job/run as well as event ID so alternate configurations cannot overwrite each other.

The viewer's review export uses `stemy.workflow.review_export.v1`, with `approved_memory_records`, `pending_candidates`, `rejected_candidates`, `object_histories`, and `unobserved_experiment_response`. Reviews currently live in browser localStorage and add `operator_review` (`status`, `sample_id`, `note`, `reviewed_at`). The original `review_status` is retained; `memory_status` reflects the operator review. This export is not governed by the raw-candidate schema. Backend integration must authenticate reviewers, record revision history, validate sample ownership, and enforce the approval gate server-side; a client-exported approval is not authorization by itself.

## Proposed frontend/backend boundary (not implemented)

1. Backend creates an experiment and upload session; binds the uploaded video to tenant, experiment and recording IDs.
2. Object storage receives the video. Backend verifies the completed upload and enqueues a job with immutable model/config versions and input hash.
3. Worker stages files in a job-specific directory, executes the pipeline, checkpoints progress, and publishes artifacts plus status. CPU preparation/transcription can be scheduled separately from GPU inference.
4. Backend stores candidates and exposes paginated events/frame windows and authorized video URLs. Keep worker-local paths out of public responses.
5. Frontend displays evidence and submits review decisions to the backend. Only approved observations enter persistent memory, with links to the original evidence and model/run provenance.

Suggested API resources are uploads, jobs, recordings, events and reviews. Agree their authentication, pagination, status transitions and error contract with the wider application before implementing routes. Use `queued`, `preparing`, `extracting`, `postprocessing`, `ready`, `failed` and `cancelled` as a starting job lifecycle. Make retries idempotent and prevent concurrent writes to one output directory.

## Work required before accepting arbitrary customer uploads

- Generalize the fixed output root, GPU selection, recording filename suffix (`*_singular_display`) and per-recording config requirements. Preparation currently discovers `.mov` files only.
- Replace fixed demo phase timestamps with explicit protocol context or a validated action model. Do not apply the four-clip interpretation to unrelated videos.
- Make model paths and FFmpeg configuration explicit, verify model hashes, and bake downloads into deployment images. The current scripts prefer FFmpeg on PATH and otherwise try a Windows Stremio path.
- Make resume compatibility cover all settings and dependencies; completed-run skips currently bypass configuration comparison. Retain identity boundaries on resumes.
- Split `audit_workflow.py` into pure validation and explicit migration/ASR cleanup. It currently rewrites transcripts/summaries, assumes the demo root, and calls Node for JS syntax checks. Do not call it as an untrusted-upload validation endpoint.
- Stream frame artifacts and load bounded video windows. The current event builder and viewer load entire recordings into memory/JavaScript and are intended for short clips.
- Persist reviews in the backend. Replace local file paths with authorized asset references; implement retention/deletion across originals, derivatives, histories and backups.
- Add a reviewed action/segmentation benchmark. Thin pipettes, caps/lids, aspiration/dispense and persistent vessel IDs remain unreliable. No trained temporal classifier has been added.
- For live intervention, connect capture, causal procedure context and delivery; benchmark prompt precision and latency. Offline upload processing cannot intervene in an experiment already completed.
- Cell morphology/growth/differentiation need timestamped microscopy and sample/well IDs; bench footage does not supply those measurements.

## Validation and release state

CI runs Python parsing, JSON/schema checks, synthetic contract validation, visibility/contract tests and viewer JavaScript syntax without GPU access or private data. It does not measure action accuracy, execute model inference or verify visual UI rendering. GPU dependencies are direct pins, not a transitive lockfile. The October 9 extraction is the recorded model benchmark; the handoff changes do not rerun it.

Repository licensing has not been chosen. Model and dependency licenses remain separate; the README records the Ultralytics licensing decision needed for product integration. Do not infer a license grant from this Git snapshot.

## Git workflow

The default branch is `main`. Keep source/config/schema changes in Git; place weights and experiment artifacts in separate storage. Do not force-add ignored footage, results, credentials or models. Use feature branches and PRs for integration changes, and run the handoff checks before merging. Remote synchronization requires a repository URL and authenticated Git access; no remote destination is assumed by this document.
