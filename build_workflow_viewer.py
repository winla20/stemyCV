"""Build a local review workspace that works without a server or cloud uploads."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('output/real_workflow'))
    parser.add_argument('--config',type=Path,default=Path('workflow_config.json'))
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    runs=[]
    for folder in sorted(args.root.glob('*_singular_display')):
        if not (folder/'events.jsonl').exists():continue
        summary=json.loads((folder/'extraction_summary.json').read_text())
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        (folder/'frames.js').write_text('window.WORKFLOW_FRAMES['+json.dumps(folder.name)+']='+json.dumps(frames,separators=(',',':'))+';',encoding='utf-8')
        temporal=[]
        if (folder/'temporal_frames.jsonl').exists():
            temporal=[json.loads(line) for line in (folder/'temporal_frames.jsonl').read_text().splitlines()]
        (folder/'temporal.js').write_text('window.WORKFLOW_TEMPORAL['+json.dumps(folder.name)+']='+json.dumps(temporal,separators=(',',':'))+';',encoding='utf-8')
        transcript=json.loads((folder/'transcript.json').read_text()) if (folder/'transcript.json').exists() else {'segments':[]}
        # Keep suspect ASR out of the main transcript; retain it for explicit inspection.
        runs.append({'id':folder.name,'summary':summary,'procedure':config['recordings'][folder.name],'events':events,'objects':json.loads((folder/'objects.json').read_text()),'coverage':json.loads((folder/'coverage.json').read_text()),'insights':json.loads((folder/'insights.json').read_text()),'transcript':transcript,'temporal_summary':json.loads((folder/'temporal_summary.json').read_text()) if (folder/'temporal_summary.json').exists() else None})
    runs.sort(key=lambda r:r['summary']['source_manifest'].get('recording_time_from_metadata') or r['id'])
    (args.root/'data.js').write_text('window.WORKFLOW_DATA='+json.dumps({'config':config,'runs':runs},separators=(',',':'))+';',encoding='utf-8')
    template=Path(__file__).with_name('workflow_viewer.html').read_text(encoding='utf-8')
    (args.root/'index.html').write_text(template,encoding='utf-8')
    print(f'Built {args.root / "index.html"} with {len(runs)} recordings')


if __name__=='__main__':main()
