"""A locally run SAM 2.1 temporal segmentation comparison for the carried plate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).parent/'Ultralytics'))
import cv2
import torch
from ultralytics.models.sam import SAM2VideoPredictor
from workflow_extract import compact_polygons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=Path('output/real_workflow/video-94_singular_display'))
    args=p.parse_args();cv2.setNumThreads(2);torch.set_num_threads(4)
    frames=[json.loads(line) for line in (args.folder/'frames.jsonl').read_text().splitlines()]
    # The initialization comes from a detector box, not a hand-drawn mask.
    candidates=[(frame,obj) for frame in frames if frame['time_s']<2 for obj in frame['objects'] if obj['label']=='multiwell plate']
    if not candidates:raise RuntimeError('No plate detection to initialize temporal segmentation')
    seed_frame,seed=max(candidates,key=lambda pair:pair[1]['detector_score'])
    start=seed_frame['frame_index']/30;end=16
    ffmpeg=shutil.which('ffmpeg') or str(Path.home()/'AppData/Local/Programs/Stremio/ffmpeg.exe')
    clip=args.folder/'sam_comparison_input.mp4'
    subprocess.run([ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(args.folder/'source_browser.mp4'),'-t',str(end-start),'-an','-vf','fps=30','-c:v','libx264','-preset','fast','-crf','18',str(clip)],check=True)
    predictor=SAM2VideoPredictor(overrides={'task':'segment','mode':'predict','imgsz':1024,'model':'models/sam2.1_t.pt','device':0,'quantize':16,'verbose':False,'save':False})
    records=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for i,result in enumerate(predictor(source=str(clip),bboxes=[seed['bbox_xyxy']],stream=True)):
        objects=[]
        if result.masks is not None:
            height,width=result.orig_shape
            for mask in result.masks.data.cpu().numpy():
                mask=(mask>.5).astype('uint8')
                if mask.shape!=(height,width):mask=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
                contours=compact_polygons(mask,width,height)
                ys,xs=__import__('numpy').where(mask)
                if not len(xs):continue
                objects.append({'label':'multiwell plate','track_id':seed['track_id']+':sam2','bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],'polygons_normalized':contours,'mask_frame_fraction':round(float(mask.mean()),5),'segmentation_confidence':None,'semantic_label_basis':'detector_initialized_prompt','evidence_kind':'temporally_propagated_mask','review_status':'unreviewed'})
        records.append({'frame_index':seed_frame['frame_index']+i,'time_s':round(start+i/30,4),'objects':objects})
        if i%60==0:print(f'SAM 2: {i} frames | {start+i/30:.1f}s',flush=True)
    elapsed=time.perf_counter()-started
    (args.folder/'temporal_frames.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records),encoding='utf-8')
    summary={'model':'sam2.1_t.pt','semantic_label':'multiwell plate','start_s':start,'end_s':start+len(records)/30,'processed_fps':30,'frames':len(records),'processing_s':round(elapsed,2),'processing_throughput_fps':round(len(records)/elapsed,2),'initialization':{'frame_index':seed_frame['frame_index'],'bbox_xyxy':seed['bbox_xyxy'],'detector_score':seed['detector_score'],'track_id':seed['track_id']},'peak_cuda_allocated_mb':round(torch.cuda.max_memory_allocated()/1024**2,1),'mask_presence_frames':sum(bool(record['objects']) for record in records),'review_status':'unreviewed','limitations':['Temporal mask presence is not verified tracking accuracy.','The predictor returns constant quality placeholders; these are omitted rather than reported as confidence.','Occlusion and new similar plates can cause drift; this comparison does not establish persistent identity.','Temporal masks are a separate comparison and do not overwrite detector evidence or clear visibility alerts.']}
    (args.folder/'temporal_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
