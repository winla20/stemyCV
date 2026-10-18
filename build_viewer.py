"""Build a self-contained timeline page beside a run's exported video and JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    transcript_path = args.run_dir / "transcript.json"
    transcript = (
        json.loads(transcript_path.read_text(encoding="utf-8"))
        if transcript_path.exists()
        else None
    )
    credit_path = args.run_dir / "source_credit.json"
    credit = json.loads(credit_path.read_text(encoding="utf-8")) if credit_path.exists() else None
    if transcript and transcript["source"] != summary["source"]:
        raise SystemExit("Transcript source does not match the segmentation source")
    video_name = next(
        name
        for name in ("overlay_with_audio.mp4", "overlay_browser.mp4", "overlay.mp4")
        if (args.run_dir / name).exists()
    )
    rows: list[dict] = []
    for label, intervals in summary["visible_intervals_by_class"].items():
        for interval in intervals:
            rows.append({"kind": "visual", "label": label, **interval})
    action_path = args.run_dir / "actions.jsonl"
    if action_path.exists():
        with action_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                action = json.loads(line)
                if action["source"] != summary["source"]:
                    raise SystemExit("Action candidate source does not match segmentation source")
                rows.append(
                    {
                        "kind": "action",
                        "label": action["top_label"].replace("_", " "),
                        "start_s": action["start_s"],
                        "end_s": action["end_s"],
                        "evidence_status": action.get("coverage_gate", {}).get("status", "unreviewed"),
                    }
                )
    interaction_path = args.run_dir / "interactions.jsonl"
    if interaction_path.exists():
        with interaction_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event["source"] != summary["source"]:
                    raise SystemExit("Interaction source does not match segmentation source")
                rows.append(
                    {
                        "kind": "interaction",
                        "label": event["relation"].replace("_", " "),
                        "start_s": event["start_s"],
                        "end_s": event["end_s"],
                    }
                )
    if transcript:
        for segment in transcript["segments"]:
            rows.append(
                {
                    "kind": "speech",
                    "label": segment["text"].strip(),
                    "start_s": segment["start"],
                    "end_s": segment["end"],
                }
            )
    rows.sort(key=lambda row: (row["start_s"], row["kind"]))
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(args.run_dir.name.replace("_", " ").title())
    classes = ", ".join(summary["classes"])
    credit_html = (
        f'<p class="small">Source: <a href="{html.escape(credit["url"], quote=True)}">{html.escape(credit["title"])}</a> '
        f'by {html.escape(credit["author"])} · <a href="{html.escape(credit["license_url"], quote=True)}">{html.escape(credit["license"])}</a>. '
        'Overlay and predictions added for this demo.</p>'
        if credit else ""
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} | Lab video proof of concept</title>
<style>
body {{ margin:0; background:#f3f5f5; color:#142c31; font:16px/1.45 system-ui, sans-serif; }}
main {{ max-width:1100px; margin:auto; padding:28px; }}
h1 {{ margin:0 0 6px; font-size:30px; }}
p {{ margin:0 0 18px; }}
.note {{ background:#fff1cf; border-left:5px solid #bd831b; padding:12px 16px; margin:18px 0; }}
.grid {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(290px,1fr); gap:18px; }}
.panel {{ background:white; border:1px solid #d7e0e1; border-radius:12px; padding:16px; }}
video {{ width:100%; background:#111; border-radius:8px; }}
.small {{ color:#506268; font-size:13px; }}
input {{ box-sizing:border-box; width:100%; padding:10px; border:1px solid #b7c7cb; border-radius:7px; margin-bottom:10px; }}
.timeline {{ max-height:68vh; overflow:auto; }}
.item {{ display:block; width:100%; text-align:left; background:#f6fafb; border:1px solid #d9e3e5; border-radius:8px; padding:9px; margin:0 0 8px; cursor:pointer; color:inherit; }}
.item:hover {{ background:#e9f4f6; }}
.speech {{ border-left:4px solid #9658a5; }}
.visual {{ border-left:4px solid #268d89; }}
.action {{ border-left:4px solid #c36b33; }}
.interaction {{ border-left:4px solid #4471b5; }}
@media(max-width:780px) {{ .grid {{ grid-template-columns:1fr; }} .timeline {{ max-height:none; }} }}
</style></head><body><main>
<h1>{title}</h1>
<p>Automatic segmentation and timestamped evidence. Classes: {html.escape(classes)}.</p>
<div class="note"><strong>Unreviewed model output.</strong> A visual interval means the model detected an object. Action candidates rank only the supplied text descriptions; their scores are not calibrated confidence or proof of a procedure step. Interaction candidates are inferred from tracked boxes and need review. Speech is an automatic transcript and may contain errors.</div>
<div class="grid"><section class="panel"><video id="video" controls preload="metadata" src="{video_name}"></video>
<p class="small">Click an item in the timeline to seek. The overlay is sampled at {summary['sampled_fps']:.1f} frames per second. Source: {html.escape(summary['source'])}.</p>
{credit_html}
<p class="small"><a href="summary.json">Summary JSON</a> · <a href="detections.jsonl">Frame records JSONL</a>{' · <a href="actions.jsonl">Action candidates JSONL</a>' if action_path.exists() else ''}{' · <a href="interactions.jsonl">Interaction candidates JSONL</a>' if interaction_path.exists() else ''}{' · <a href="transcript.json">Transcript JSON</a>' if transcript else ''}</p>
</section><section class="panel"><label for="filter"><strong>Evidence timeline</strong></label><input id="filter" placeholder="Filter by object or speech"><div class="timeline" id="timeline"></div></section></div>
</main><script>
const rows = {payload};
const timeline = document.getElementById('timeline');
const video = document.getElementById('video');
const filter = document.getElementById('filter');
function render() {{
  timeline.replaceChildren();
  const q = filter.value.trim().toLowerCase();
  for (const row of rows.filter(r => r.label.toLowerCase().includes(q))) {{
    const button = document.createElement('button');
    button.className = 'item ' + row.kind;
    const time = row.start_s.toFixed(1) + '–' + row.end_s.toFixed(1) + 's';
    button.textContent = time + ' | ' + (row.kind === 'speech' ? 'Spoken: ' : row.kind === 'action' ? 'Action candidate: ' : row.kind === 'interaction' ? 'Interaction candidate: ' : 'Detected: ') + row.label + (row.kind === 'action' ? ' (' + row.evidence_status.replaceAll('_', ' ') + ')' : '');
    button.onclick = () => {{ video.currentTime = row.start_s; video.play(); }};
    timeline.appendChild(button);
  }}
}}
filter.addEventListener('input', render); render();
</script></body></html>"""
    target = args.run_dir / "viewer.html"
    target.write_text(page, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
