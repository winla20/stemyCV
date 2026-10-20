# Processed demo

Saved results for all four bundled recordings: 13,243 analyzed frames, 121 unreviewed event candidates and the SAM plate comparison. These are the original October 9 extraction results, not new predictions made by replay.

From the repository root, run `python3 replay_demo.py` (Windows: `python replay_demo.py`) and open http://127.0.0.1:8765. Only Python and a browser are needed; no PyTorch, GPU, FFmpeg or Node installation is required for replay. The server binds to localhost and supports byte-range video seeking.

`manifest.json` contains hashes for all 38 saved artifacts (~186 MB). Source paths in manifests are relative to the repository root. Inputs remain in `data/demo/`; masks, events and normalized H.264/AAC review videos are here. Generated viewer HTML and JavaScript are ignored and rebuilt by the replay command.

Use the JSON/JSONL files as integration fixtures. Browser review choices are stored in localStorage and exported on request. Raw saved candidates remain unreviewed. Inference writes to `output/` and must not overwrite this bundle.

Playback color correction: these demo exports carried HLG/BT.2020 tags that caused blown highlights and saturation in browser playback. Their review copies now explicitly use SDR BT.709 metadata. The H.264 headers/container were remuxed without re-encoding pixels; every decoded YUV frame, copied audio packet and timestamp was checked for equality. Manifests record both the correction and prior review-file hash; extraction summaries retain `inference_review_video_sha256`. Original `.mov` files and saved predictions are unchanged. This policy is specific to these exports; genuine HDR footage needs proper tone mapping.
