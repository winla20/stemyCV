"""Test a specific serological-pipette prompt and work-area crop on failed intervals."""
import json
import os
from pathlib import Path
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).parent/'Ultralytics'))
import cv2
from PIL import Image,ImageDraw
import torch
from ultralytics import YOLOE


def main():
    cv2.setNumThreads(2);torch.set_num_threads(4)
    root=Path('output/real_workflow/probe');root.mkdir(exist_ok=True)
    model=YOLOE('yoloe-26s-seg.pt')
    labels=['serological pipette','multiwell plate','gloved hand','bottle']
    model.set_classes(labels)
    cap=cv2.VideoCapture('output/real_workflow/video-108_singular_display/source_browser.mp4')
    times=[96,100,105,110,140,155,160,176];records=[]
    sheet=Image.new('RGB',(4*360,2*420),'#111827');draw=ImageDraw.Draw(sheet)
    for i,t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read();assert ok
        height,width=frame.shape[:2];y0=round(height*.45);crop=frame[y0:]
        result=model.predict(crop,imgsz=960,conf=.15,device='0',quantize=16,retina_masks=True,verbose=False)[0]
        objects=[{'label':result.names[int(box.cls)],'score':round(float(box.conf),3),'bbox_xyxy':[round(float(v),1) for v in box.xyxy[0]]} for box in result.boxes]
        records.append({'time_s':t,'crop_origin_xy':[0,y0],'input_size':960,'objects':objects})
        image=Image.fromarray(cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB));image.thumbnail((360,390))
        x,y=(i%4)*360,(i//4)*420;sheet.paste(image,(x,y));draw.text((x+5,y+396),f'{t}s | focused / 960px',fill='white')
        print(t,[o for o in objects if o['label']=='serological pipette'],flush=True)
    cap.release();sheet.save(root/'focused_tools.jpg',quality=90)
    (root/'focused_tools.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
