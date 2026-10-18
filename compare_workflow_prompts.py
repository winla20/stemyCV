"""Compare text and visual prompt predictions on workflow probe frames."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw
from workflow_models import load_detector


def main():
    output=Path('output/real_workflow/probe')
    times={'video-108':[80,110,140,176], 'video-94':[1,4,12,15]}
    records=[]
    for mode in ['text','visual']:
        model=load_detector(mode)
        for name, samples in times.items():
            cap=cv2.VideoCapture(str(Path('output/real_workflow')/(name+'_singular_display')/'source_browser.mp4'))
            sheet=Image.new('RGB',(4*360,510),'#111827')
            draw=ImageDraw.Draw(sheet)
            for i,t in enumerate(samples):
                cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
                ok,frame=cap.read()
                assert ok
                result=model.predict(frame,imgsz=640,conf=.2,device=0,quantize=16,retina_masks=True,verbose=False)[0]
                records.append({'mode':mode,'source':name,'time_s':t,'objects':[{'label':result.names[int(b.cls)],'score':round(float(b.conf),3),'bbox':[round(float(v),1) for v in b.xyxy[0]]} for b in result.boxes]})
                image=Image.fromarray(cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB))
                image.thumbnail((360,480))
                sheet.paste(image,(i*360+(360-image.width)//2,0))
                draw.text((i*360+5,485),f'{name} {t}s | {mode}',fill='white')
            cap.release()
            sheet.save(output/f'{name}_{mode}_comparison.jpg',quality=90)
            print(f'{mode} {name} done',flush=True)
        del model
    (output/'prompt_comparison.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__': main()
