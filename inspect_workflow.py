"""Inventory local workflow recordings and make timestamped visual contact sheets."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/real_workflow/inventory'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recordings = []
    ffprobe = shutil.which('ffprobe')
    for source in sorted(args.source_dir.iterdir()):
        if source.suffix.lower() not in {'.mov', '.mp4', '.mkv', '.webm', '.avi'}:
            continue
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open {source}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps
        meta = {'source': str(source.resolve()), 'source_sha256': hashlib.file_digest(source.open('rb'), 'sha256').hexdigest(), 'fps': fps, 'frame_count_header': frames, 'duration_s_header': duration, 'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        if ffprobe:
            probe = subprocess.run([ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(source)], capture_output=True, text=True, check=True)
            meta['probe'] = json.loads(probe.stdout)
        sheet = Image.new('RGB', (4 * 480, 3 * 300), '#151923')
        draw = ImageDraw.Draw(sheet)
        samples = []
        for i in range(12):
            t = min(duration - 1/fps, duration * (i + .3) / 12)
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rgb.thumbnail((480, 272))
            x, y = (i % 4) * 480, (i // 4) * 300
            sheet.paste(rgb, (x + (480-rgb.width)//2, y))
            draw.text((x + 8, y + 276), f'{source.name} | {t:.2f}s', fill='white')
            samples.append(round(t, 3))
        target = args.output / f'{source.stem}_contact.jpg'
        sheet.save(target, quality=90)
        meta['contact_sheet'] = str(target.resolve())
        meta['sample_times_s'] = samples
        capture.release()
        recordings.append(meta)
    (args.output / 'inventory.json').write_text(json.dumps(recordings, indent=2), encoding='utf-8')
    print(json.dumps(recordings, indent=2))


if __name__ == '__main__':
    main()
