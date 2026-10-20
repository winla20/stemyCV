"""Check prompt coverage on sparse, timestamped workflow frames before a full run."""
import os
from pathlib import Path
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).parent/'Ultralytics'))
import json
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLOE

LABELS = ['gloved hand', 'bottle', 'multiwell plate', 'pipette', 'pipette controller', 'spray bottle', 'paper towel', 'microscope', 'incubator', 'refrigerator']
TIMES = {'video-60': [16, 23, 27, 45, 51, 55], 'video-94': [1, 4, 9, 12, 15, 18], 'video-102': [19, 35, 50, 66, 79, 138], 'video-108': [20, 49, 80, 110, 140, 169]}


def main():
    root = Path('output/real_workflow/probe')
    root.mkdir(parents=True, exist_ok=True)
    model = YOLOE('yoloe-26s-seg.pt')
    model.set_classes(LABELS)
    records = []
    for source in sorted((Path(__file__).resolve().parent/'data/demo').glob('*.mov')):
        key = source.stem.split('_')[0]
        cap = cv2.VideoCapture(str(source))
        sheet = Image.new('RGB', (3*480, 2*670), '#111827')
        draw = ImageDraw.Draw(sheet)
        for i, t in enumerate(TIMES[key]):
            cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
            ok, frame = cap.read()
            if not ok:
                continue
            result = model.predict(frame, imgsz=640, conf=.15, device=0, half=True, retina_masks=True, verbose=False)[0]
            objects = [{'label': result.names[int(b.cls)], 'score': round(float(b.conf),3), 'bbox': [round(float(x),1) for x in b.xyxy[0]]} for b in result.boxes]
            records.append({'source':source.name, 'time_s':t, 'objects':objects})
            visual = Image.fromarray(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
            visual.thumbnail((480,640))
            x,y = (i%3)*480,(i//3)*670
            sheet.paste(visual,(x+(480-visual.width)//2,y))
            draw.text((x+8,y+645),f'{key} | {t}s',fill='white')
        cap.release()
        sheet.save(root/f'{key}.jpg',quality=90)
        print(f'Finished {key}',flush=True)
    (root/'predictions.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    main()
