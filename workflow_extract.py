"""Full-frame, local workflow extraction with compact masks and explicit provenance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from workflow_models import load_detector, LABELS


def phase_at(config, t):
    return next((p for p in config['phases'] if p['start_s'] <= t < p['end_s']), {'id':'unknown','zone':'unknown','description':'Outside inferred procedure map','expected':[]})


def iou(a, b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def compact_polygons(mask, width, height):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    paths=[]
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:4]:
        if cv2.contourArea(contour)<8:continue
        contour=cv2.approxPolyDP(contour,max(1,.002*cv2.arcLength(contour,True)),True).reshape(-1,2)
        paths.append([[round(float(x)/width,4),round(float(y)/height,4)] for x,y in contour])
    return paths


def camera_transform(previous, current):
    if previous is None:return None
    points=cv2.goodFeaturesToTrack(previous,100,.02,12)
    if points is None or len(points)<12:return None
    moved,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=2)
    if moved is None:return None
    valid=status.ravel()==1
    if valid.sum()<12:return None
    matrix,inliers=cv2.estimateAffinePartial2D(points[valid],moved[valid],method=cv2.RANSAC,ransacReprojThreshold=3)
    if matrix is None or inliers.sum()<12:return None
    if not .85<math.hypot(matrix[0,0],matrix[0,1])<1.15:return None
    return matrix


def batched_predictions(cap, model, recording, fps, start_frame, args, timing):
    """Batch GPU work while applying tracking in frame order and resetting at location changes."""
    batch=[];batch_zone=None;previous_zone=None;index=start_frame
    while True:
        tick=time.perf_counter();ok,frame=cap.read();timing['decode_s']+=time.perf_counter()-tick
        zone=phase_at(recording,index/fps)['zone'] if ok else None
        if batch and (not ok or zone!=batch_zone or len(batch)>=args.batch_size):
            if previous_zone is not None and previous_zone!=batch_zone:
                for tracker in getattr(model.predictor,'trackers',[]):tracker.reset()
            tick=time.perf_counter()
            results=model.track([image for _,image in batch],persist=True,tracker='bytetrack.yaml',imgsz=args.image_size,conf=args.confidence,device='0',quantize=16,retina_masks=True,batch=args.batch_size,verbose=False)
            timing['detector_tracking_s']+=time.perf_counter()-tick
            assert len(results)==len(batch)
            for (frame_index,image),result in zip(batch,results):yield frame_index,image,result
            previous_zone=batch_zone;batch=[]
        if not ok:break
        if not batch:batch_zone=zone
        batch.append((index,frame));index+=1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/real_workflow'))
    p.add_argument('--config',type=Path,default=Path('workflow_config.json'))
    p.add_argument('--image-size',type=int,default=640)
    p.add_argument('--confidence',type=float,default=.2)
    p.add_argument('--recordings',nargs='+')
    p.add_argument('--prompt-mode',choices=['text','visual'],default='text')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    # Large default CPU thread pools slow down small per-frame operations on this laptop.
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    config=json.loads(args.config.read_text())
    model=load_detector(args.prompt_mode,args.image_size)
    manifests=sorted(args.root.glob('*_singular_display/source_manifest.json'))
    if args.recordings:
        manifests.sort(key=lambda m:args.recordings.index(m.parent.name) if m.parent.name in args.recordings else len(args.recordings))
    for manifest in manifests:
        provenance=json.loads(manifest.read_text())
        name=provenance['recording_id']
        if args.recordings and name not in args.recordings:continue
        folder=manifest.parent
        if (folder/'extraction_summary.json').exists() and not args.overwrite:
            print(f'Skipping completed {name}',flush=True);continue
        recording=config['recordings'][name]
        parameters={'model':'yoloe-26s-seg.pt','image_size':args.image_size,'confidence':args.confidence,'prompt_mode':args.prompt_mode,'phase_map':recording['phases']}
        with Path(parameters['model']).open('rb') as weight_file:
            parameters['model_sha256']=hashlib.file_digest(weight_file,'sha256').hexdigest()
        parameter_path=folder/'run_parameters.json'
        if args.resume and parameter_path.exists() and json.loads(parameter_path.read_text())!=parameters:
            raise RuntimeError('Detector settings or phase map changed; use a fresh output root or explicit --overwrite')
        parameter_path.write_text(json.dumps(parameters,indent=2),encoding='utf-8')
        source=Path(provenance['review_video'])
        cap=cv2.VideoCapture(str(source))
        fps=cap.get(cv2.CAP_PROP_FPS); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or abs(fps-30)>.01:raise RuntimeError('Prepare a timestamp-normalized 30 fps source first')
        # Each recording and location gets a separate visual tracking namespace.
        if model.predictor and hasattr(model.predictor,'trackers'):
            for tracker in model.predictor.trackers:tracker.reset()
        counter=Counter();present=Counter();frame_index=0;previous_gray=None;previous_centers={};previous_zone=None;scene_epoch=0
        existing=[]
        if args.resume and not args.overwrite and (folder/'frames.jsonl').exists():
            for line in (folder/'frames.jsonl').read_text().splitlines():
                try:existing.append(json.loads(line))
                except json.JSONDecodeError:break
            assert [f['frame_index'] for f in existing]==list(range(len(existing)))
            assert all(f['source_sha256']==provenance['original_sha256'] for f in existing), 'Resume source hash mismatch'
            assert all(f['phase_id']==phase_at(recording,f['time_s'])['id'] for f in existing), 'Phase configuration changed; use a fresh output root'
            assert all(f['location']==phase_at(recording,f['time_s'])['zone'] for f in existing), 'Location configuration changed; use a fresh output root'
            frame_index=len(existing)
            for record in existing:
                counter.update(o['label'] for o in record['objects'])
                present.update({o['label'] for o in record['objects']})
            epochs=[int(o['track_id'].split(':z')[1].split(':')[0]) for record in existing for o in record['objects'] if o['track_id']]
            scene_epoch=max(epochs,default=-1)+1
            (folder/'frames.jsonl').write_text(''.join(json.dumps(f,separators=(',',':'))+'\n' for f in existing),encoding='utf-8')
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
            print(f'Resuming {name} at {frame_index/fps:.1f}s; new temporary identity epoch',flush=True)
        segment_start_frame=frame_index
        started=time.perf_counter(); timing=Counter()
        with (folder/'frames.jsonl').open('a' if existing else 'w',encoding='utf-8') as stream:
            for frame_index,frame,result in batched_predictions(cap,model,recording,fps,frame_index,args,timing):
                t=frame_index/fps
                phase=phase_at(recording,t)
                if previous_zone is not None and previous_zone!=phase['zone']:
                    scene_epoch+=1;previous_centers={};previous_gray=None
                previous_zone=phase['zone']
                # Optical flow estimates background movement; residuals remain image-space observations.
                tick=time.perf_counter()
                gray=cv2.cvtColor(cv2.resize(frame,(320,round(height*320/width))),cv2.COLOR_BGR2GRAY)
                affine=camera_transform(previous_gray,gray)
                timing['camera_motion_s']+=time.perf_counter()-tick
                tick=time.perf_counter()
                boxes=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy();classes=result.boxes.cls.cpu().numpy().astype(int)
                ids=result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else [None]*len(boxes)
                masks=result.masks.data.cpu().numpy() if result.masks is not None else []
                objects=[];accepted=[];centers={}
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                for idx in np.argsort(-scores):
                    label=result.names[int(classes[idx])];bbox=boxes[idx]
                    if any(label==other_label and iou(bbox,other_bbox)>.5 for other_label,other_bbox in accepted):continue
                    accepted.append((label,bbox))
                    track_id=f'{name}:z{scene_epoch}:t{ids[idx]}' if ids[idx] is not None else None
                    center=np.array([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])*320/width
                    speed=None
                    if affine is not None and track_id in previous_centers:
                        prediction=affine@np.r_[previous_centers[track_id],1]
                        speed=round(float(np.linalg.norm(center-prediction))*fps/320,4)
                    if track_id:centers[track_id]=center
                    polygons=[];area=None;red_fraction=None
                    if idx<len(masks):
                        mask=(masks[idx]>.5).astype(np.uint8)
                        if mask.shape!=(height,width):mask=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
                        area=round(float(mask.mean()),5)
                        polygons=compact_polygons(mask,width,height)
                        if label=='bottle' and mask.sum():
                            red=((hsv[:,:,0]<12)|(hsv[:,:,0]>165))&(hsv[:,:,1]>65)&(hsv[:,:,2]>45)
                            red_fraction=round(float((red&mask.astype(bool)).sum()/mask.sum()),4)
                    objects.append({'label':label,'detector_score':round(float(scores[idx]),4),'track_id':track_id,'persistent_sample_id':None,'bbox_xyxy':[round(float(v),1) for v in bbox],'polygons_normalized':polygons,'mask_frame_fraction':area,'red_pixel_fraction_within_mask':red_fraction,'camera_corrected_speed_frame_widths_per_s':speed,'evidence_kind':'direct_frame_prediction','review_status':'unreviewed'})
                    counter[label]+=1
                for label in {o['label'] for o in objects}:present[label]+=1
                record={'schema_version':'stemy.workflow.frame.v1','recording_id':name,'source_sha256':provenance['original_sha256'],'frame_index':frame_index,'time_s':round(t,4),'location':phase['zone'],'location_basis':'inferred_visual_phase_map','phase_id':phase['id'],'phase_basis':'provisional_visual_interpretation','camera_motion_estimate_available':affine is not None,'objects':objects}
                stream.write(json.dumps(record,separators=(',',':'))+'\n')
                timing['mask_properties_export_s']+=time.perf_counter()-tick
                previous_gray=gray;previous_centers=centers;frame_index+=1
                if frame_index%300==0:print(f'{name}: {t:.1f}s | {(frame_index-segment_start_frame)/(time.perf_counter()-started):.1f} processed fps',flush=True)
        cap.release()
        elapsed=time.perf_counter()-started
        summary={'schema_version':'stemy.workflow.run.v1','recording_id':name,'created_at_utc':datetime.now(timezone.utc).isoformat(),'model':'yoloe-26s-seg.pt','prompt_mode':args.prompt_mode,'classes':LABELS,'input_size':args.image_size,'confidence_threshold':args.confidence,'compute_precision':'fp16','batch_size':args.batch_size,'processed_frames':frame_index,'processed_fps':fps,'duration_s':frame_index/fps,'width':width,'height':height,'processing_loop_s':round(elapsed,2),'processing_throughput_fps':round((frame_index-segment_start_frame)/elapsed,2),'timing_scope':'Current processing segment only; earlier interrupted work excluded' if existing else 'Entire recording processing loop','segment_start_frame':segment_start_frame,'resume_identity_boundary_s':segment_start_frame/fps if existing else None,'detections_by_class':dict(counter),'frames_with_class_prediction':dict(present),'source_manifest':provenance,'protocol_status':config['protocol_status'],'review_status':'unreviewed','limitations':['Detection presence is not verified recall or accuracy.','Frame scores are not calibrated event probabilities.','Track IDs are local to a recording and location epoch, not persistent sample identities.','Image-space color, area and motion do not measure liquid volume, cellular growth or sterility.']}
        summary['processing_breakdown_s']={k:round(v,3) for k,v in timing.items()}
        summary['host_model_device']=str(next(model.model.parameters()).device)
        summary['detector_device']=str(model.predictor.device)
        summary['peak_cuda_allocated_mb']=round(torch.cuda.max_memory_allocated()/1024**2,1)
        (folder/'extraction_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps({k:summary[k] for k in ['recording_id','processed_frames','processing_loop_s','processing_throughput_fps']}),flush=True)


if __name__=='__main__':main()
