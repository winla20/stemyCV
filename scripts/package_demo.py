"""Bundle existing processed results with portable paths; does not run inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ['source_browser.mp4', 'frames.jsonl', 'events.jsonl', 'objects.json',
         'coverage.json', 'insights.json', 'transcript.json', 'extraction_summary.json',
         'source_manifest.json', 'temporal_frames.jsonl', 'temporal_summary.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'output/real_workflow')
    args = parser.parse_args()
    target = ROOT/'demo_results'
    config = json.loads((ROOT/'workflow_config.json').read_text())
    inventory = []
    for recording in config['recordings']:
        source = args.source/recording
        dest = target/recording
        dest.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((source/'source_manifest.json').read_text())
        manifest.update(original_path=f'data/demo/{recording}.mov',
                        review_video=f'demo_results/{recording}/source_browser.mp4',
                        path_basis='repository_root')
        with (ROOT/manifest['original_path']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['original_sha256']:
                raise ValueError(f'Input mismatch: {recording}')
        with (source/'source_browser.mp4').open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['review_video_sha256']:
                raise ValueError(f'Review video mismatch: {recording}')
        for name in FILES:
            path = source/name
            if not path.exists():
                if name.startswith('temporal_'):continue
                raise FileNotFoundError(path)
            destination = dest/name
            if name == 'source_manifest.json':
                destination.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
            elif name == 'extraction_summary.json':
                summary = json.loads(path.read_text())
                summary['source_manifest'] = manifest
                destination.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
            elif path.suffix in ('.json','.jsonl'):
                # Match Git's LF checkout policy so hashes also hold on other OSes.
                destination.write_text(path.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
            else:
                shutil.copyfile(path,destination)
            with destination.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            inventory.append({'path':destination.relative_to(target).as_posix(),
                              'size_bytes':destination.stat().st_size,'sha256':digest})
    (target/'manifest.json').write_text(json.dumps({
        'schema_version':'stemy.processed_demo.v1',
        'description':'Saved October 9 extraction; raw candidates remain unreviewed. No inference runs during replay.',
        'files':inventory},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Packaged {len(inventory)} artifacts, {sum(f["size_bytes"] for f in inventory)/1e6:.1f} MB')


if __name__=='__main__':main()
