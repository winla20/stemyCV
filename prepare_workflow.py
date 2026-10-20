"""Make timestamp-aligned CFR review copies without changing source recordings."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from workflow_runtime import find_ffmpeg
from workflow_video import SDR_POLICY, COLOR_OPTIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source_dir', type=Path)
    p.add_argument('--output', type=Path, default=Path('output/real_workflow'))
    p.add_argument('--color-policy',choices=['preserve',SDR_POLICY],default='preserve')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    manifest = []
    for source in sorted(args.source_dir.glob('*.mov')):
        folder = args.output/source.stem
        folder.mkdir(exist_ok=True)
        target = folder/'source_browser.mp4'
        with source.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        previous_manifest=folder/'source_manifest.json'
        if previous_manifest.exists() and json.loads(previous_manifest.read_text())['original_sha256']!=digest:
            raise RuntimeError(f'Source changed for {source.name}; choose a fresh --output directory to preserve existing evidence')
        if not target.exists():
            print(f'Preparing {source.name}', flush=True)
            color=COLOR_OPTIONS if args.color_policy==SDR_POLICY else []
            subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source), '-map', '0:v:0', '-map', '0:a:0?', '-vf', 'fps=fps=30:start_time=0', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', *color, '-c:a', 'aac', '-map_metadata', '-1', '-movflags', '+faststart', str(target)], check=True)
        info = subprocess.run([ffmpeg, '-hide_banner', '-i', str(source)], capture_output=True, text=True).stderr
        recording_time = re.search(r'com.apple.quicktime.creationdate:\s*(\S+)', info)
        with target.open('rb') as f:
            review_digest=hashlib.file_digest(f,'sha256').hexdigest()
        if previous_manifest.exists():
            previous=json.loads(previous_manifest.read_text())
            if previous.get('review_video_sha256') and previous['review_video_sha256']!=review_digest:
                raise RuntimeError(f'Review video changed for {source.name}; choose a fresh output directory')
        entry = {'recording_id': source.stem, 'original_path': str(source.resolve()), 'original_sha256': digest, 'review_video': str(target.resolve()), 'review_video_sha256':review_digest,'review_fps': 30, 'recording_time_from_metadata': recording_time.group(1) if recording_time else None, 'timestamp_basis': 'Seconds in timestamp-normalized 30 fps review copy; audio retained. Source recordings are unchanged.'}
        entry['review_color_policy']=args.color_policy
        if previous_manifest.exists():
            previous=json.loads(previous_manifest.read_text())
            # Existing files are reused, not silently relabeled by a config change.
            entry['review_color_policy']=previous.get('review_color_policy','preserve')
            if 'review_color_correction' in previous:entry['review_color_correction']=previous['review_color_correction']
            if entry['review_color_policy']!=args.color_policy:
                raise RuntimeError('Review color policy changed; use a fresh output root, or repair the known demo with scripts/fix_playback_color.py')
        (folder/'source_manifest.json').write_text(json.dumps(entry, indent=2), encoding='utf-8')
        manifest.append(entry)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
