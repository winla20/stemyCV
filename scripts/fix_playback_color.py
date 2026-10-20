"""Repair the known demo exports' stale HDR tags without changing decoded samples."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflow_runtime import find_ffmpeg
from workflow_video import SDR_POLICY, repair_sdr_tags, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'output/real_workflow')
    args=parser.parse_args()
    config=json.loads((ROOT/'workflow_config.json').read_text(encoding='utf-8'))
    inputs=json.loads((ROOT/'data/demo/manifest.json').read_text(encoding='utf-8'))['files']
    known_hashes={Path(entry['filename']).stem:entry['sha256'] for entry in inputs}
    ffmpeg=find_ffmpeg()
    for name in config['recordings']:
        folder=args.root/name
        manifest_path=folder/'source_manifest.json'
        if not manifest_path.exists():continue
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['original_sha256']!=known_hashes[name]:
            raise ValueError(f'This repair is only for the known demo source exports: {name}')
        if manifest.get('review_color_policy')==SDR_POLICY:
            print(f'Already corrected: {name}',flush=True)
            continue
        video=folder/'source_browser.mp4'
        if sha256(video)!=manifest['review_video_sha256']:
            raise ValueError(f'Existing review hash mismatch: {name}')
        print(f'Correcting and checking every frame/audio packet: {name}',flush=True)
        correction=repair_sdr_tags(ffmpeg,video)
        manifest.update(review_video_sha256=sha256(video),review_color_policy=SDR_POLICY,
                        review_color_correction=correction)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
        summary_path=folder/'extraction_summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text(encoding='utf-8'))
            summary.setdefault('inference_review_video_sha256',correction['previous_review_video_sha256'])
            summary['source_manifest']=manifest
            summary_path.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(f'Passed: {name}; frames, audio and timing unchanged',flush=True)
    # Keep the worker's top-level preparation inventory consistent.
    inventory=args.root/'manifest.json'
    if inventory.exists() and isinstance(json.loads(inventory.read_text(encoding='utf-8')),list):
        entries=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.root.glob('*/source_manifest.json'))]
        inventory.write_text(json.dumps(entries,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
