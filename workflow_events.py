"""Convert frame evidence into conservative histories, relations and visibility prompts."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
import hashlib
import math
import re
from pathlib import Path
from statistics import median
from workflow_visibility import VisibilityMonitor


def bbox_gap(a,b):
    return math.hypot(max(0,a[0]-b[2],b[0]-a[2]),max(0,a[1]-b[3],b[1]-a[3]))


def contiguous(points, step, max_gap_steps=1.5):
    groups=[];current=[]
    for point in points:
        if current and point['time_s']-current[-1]['time_s']>step*max_gap_steps:
            groups.append(current);current=[]
        current.append(point)
    if current:groups.append(current)
    return groups


def endpoint_near_plate(tool,plate,width,height):
    # Either shaft endpoint can qualify; orientation does not identify the liquid-handling tip.
    import numpy as np
    points=np.array([point for path in tool['polygons_normalized'] for point in path])
    if len(points)<4:return False
    points*=np.array([width,height])
    centered=points-points.mean(axis=0)
    _,singular,vectors=np.linalg.svd(centered,full_matrices=False)
    if len(singular)<2 or singular[0]/max(singular[1],1)<3:return False
    projection=centered@vectors[0]
    ends=points[[projection.argmin(),projection.argmax()]]
    box=plate['bbox_xyxy']
    return min(bbox_gap([x,y,x,y],box) for x,y in ends)<.035*width


def iso_time(manifest, seconds):
    if not manifest.get('recording_time_from_metadata'):return None
    dt=datetime.fromisoformat(manifest['recording_time_from_metadata'].replace('Z','+00:00'))
    return (dt+timedelta(seconds=seconds)).isoformat()


def extract_recording(folder, config):
    summary=json.loads((folder/'extraction_summary.json').read_text())
    frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines() if line]
    if not frames:raise RuntimeError(f'No frames for {folder}')
    name=summary['recording_id'];recording=config['recordings'][name]
    step=1/summary['processed_fps'];width=summary['width'];height=summary['height']
    events=[];histories=defaultdict(list);relation_points=defaultdict(list)
    for frame in frames:
        hands=[o for o in frame['objects'] if o['label']=='gloved hand']
        vessels=[o for o in frame['objects'] if o['label'] in ['bottle','multiwell plate']]
        plates=[o for o in vessels if o['label']=='multiwell plate']
        for obj in frame['objects']:
            if obj['track_id']:
                histories[obj['track_id']].append({'time_s':frame['time_s'],'frame_index':frame['frame_index'],'location':frame['location'],**obj})
        if frame['location']=='transition':continue
        for vessel in vessels:
            if vessel['track_id'] is None:continue
            near=[h for h in hands if bbox_gap(h['bbox_xyxy'],vessel['bbox_xyxy'])<=width*.035]
            if near:
                relation_points[('hand_near_vessel',vessel['track_id'],vessel['label'],None)].append({'time_s':frame['time_s'],'frame_index':frame['frame_index'],'location':frame['location'],'scores':[vessel['detector_score'],max(h['detector_score'] for h in near)]})
                speed=vessel.get('camera_corrected_speed_frame_widths_per_s')
                if speed is not None and speed>.03:
                    relation_points[('vessel_motion_near_hand',vessel['track_id'],vessel['label'],None)].append({'time_s':frame['time_s'],'frame_index':frame['frame_index'],'location':frame['location'],'scores':[vessel['detector_score']], 'speed':speed})
        for tool in [o for o in frame['objects'] if o['label']=='pipette' and o['track_id']]:
            for plate in plates:
                if plate['track_id'] and endpoint_near_plate(tool,plate,width,height):
                    relation_points[('pipette_shaft_endpoint_near_plate',plate['track_id'],'multiwell plate',tool['track_id'])].append({'time_s':frame['time_s'],'frame_index':frame['frame_index'],'location':frame['location'],'scores':[tool['detector_score'],plate['detector_score']]})
    for key,points in relation_points.items():
        relation,track,label,tool_track=key
        for group in contiguous(points,step):
            duration=group[-1]['time_s']-group[0]['time_s']+step
            if duration<config['minimum_relation_seconds']:continue
            events.append({'record_type':'object_relation_candidate','action':relation,'description':f'{relation.replace("_"," ")} ({label})','start_s':group[0]['time_s'],'end_s':round(group[-1]['time_s']+step,4),'duration_s':round(duration,3),'location':group[0]['location'],'track_ids':[value for value in [track,tool_track] if value],'supporting_frames':len(group),'median_detector_score':round(median(score for point in group for score in point['scores']),4),'event_confidence':None,'basis':'tracked_boxes_and_masks; geometric_proximity_does_not_prove_contact_or_liquid_transfer','review_status':'unreviewed'})
    # These descriptions came from explicit visual inspection and remain separately attributed.
    for observed in recording.get('visual_observations',[]):
        events.append({'record_type':'visual_interpretation_candidate',**observed,'duration_s':observed['end_s']-observed['start_s'],'location':next((p['zone'] for p in recording['phases'] if p['start_s']<=observed['start_s']<p['end_s']),'unknown'),'track_ids':[],'event_confidence':None,'basis':'assistant_visual_inspection_of_timestamped_frames; not_automatically_classified','review_status':'unreviewed'})
    response_mentions=[]
    if (folder/'transcript.json').exists():
        transcript=json.loads((folder/'transcript.json').read_text())
        for segment in transcript['segments']:
            if segment['evidence_status']!='asr_candidate' or segment['start_s']>=summary['duration_s']:continue
            terms=re.findall(r'\b(?:cells?|passage|alive|toxicity|differentiat\w*|morpholog\w*|growth|die\s+off|die\s*off)\b',segment['text'],re.I)
            if not terms:continue
            mention={'record_type':'reported_response_candidate','action':'response_related_speech','start_s':segment['start_s'],'end_s':min(segment['end_s'],summary['duration_s']),'description':segment['text'],'matched_terms':terms,'location':next((p['zone'] for p in recording['phases'] if p['start_s']<=segment['start_s']<p['end_s']),'unknown'),'track_ids':[],'event_confidence':None,'basis':'ASR_keyword_matched_speech; may_be_discussion_rather_than_a_sample_observation','review_status':'unreviewed','interpretation':'Reported words only; no cell-health, differentiation or chemical-identity conclusion.'}
            events.append(mention);response_mentions.append(mention.copy())
    coverage=[];alerts=[]
    windows=[]
    for phase in recording['phases']:
        for label in phase['expected']:
            windows.append({'label':label,'start_s':phase['start_s'],'end_s':phase['end_s'],'basis':'inferred_phase_expectation','phase_id':phase['id']})
    windows.extend({**window,'basis':'inferred_critical_visibility_window','phase_id':None} for window in recording.get('visibility_windows',[]))
    for window in windows:
        selected=[f for f in frames if window['start_s']<=f['time_s']<window['end_s']]
        if not selected:continue
        missing=[];present=0
        for frame in selected:
            if any(o['label']==window['label'] for o in frame['objects']):present+=1
            else:missing.append({'time_s':frame['time_s'],'frame_index':frame['frame_index']})
        groups=contiguous(missing,step)
        max_gap=max((g[-1]['time_s']-g[0]['time_s']+step for g in groups),default=0)
        coverage.append({**window,'evaluated_frames':len(selected),'frames_with_class_prediction':present,'prediction_presence_fraction':round(present/len(selected),4),'longest_no_prediction_gap_s':round(max_gap,3),'interpretation':'Any instance of this class may satisfy presence; identity and actual visibility still require review. This is not recall.'})
    monitor=VisibilityMonitor(config['visibility_gap_seconds'],config.get('minimum_rolling_prediction_presence',.6),config.get('recovery_rolling_prediction_presence',.8))
    for frame in frames:
        expectations=[{'key':f'{i}:{window["label"]}','label':window['label'],'basis':window['basis']} for i,window in enumerate(windows) if window['start_s']<=frame['time_s']<window['end_s']]
        monitor.update(frame['time_s'],frame['objects'],expectations)
    alerts=monitor.finish(summary['duration_s'])
    for event in sorted(events+alerts,key=lambda e:(e['start_s'],e['end_s'])):
        identity={key:event.get(key) for key in ['record_type','action','start_s','end_s','track_ids','basis','description']}
        identity['source_sha256']=summary['source_manifest']['original_sha256']
        identity['detector_settings']={key:summary.get(key) for key in ['model','prompt_mode','input_size','confidence_threshold']}
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        event.update({'schema_version':'stemy.workflow.event.v1','event_id':f'{name}:e_{digest}','recording_id':name,'source_sha256':summary['source_manifest']['original_sha256'],'evidence':{'video':'source_browser.mp4','start_s':event['start_s'],'end_s':event['end_s']},'recording_time_metadata_estimate':iso_time(summary['source_manifest'],event['start_s']),'persistent_sample_id':None,'memory_status':'pending_review'})
    combined=sorted(events+alerts,key=lambda e:(e['start_s'],e['end_s']))
    (folder/'events.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in combined),encoding='utf-8')
    tracks=[]
    for track,points in histories.items():
        if len(points)<3:continue
        paths=contiguous(points,step)
        strongest=max(points,key=lambda p:p['detector_score'])
        reds=[p['red_pixel_fraction_within_mask'] for p in points if p['red_pixel_fraction_within_mask'] is not None]
        tracks.append({'track_id':track,'label':points[0]['label'],'persistent_sample_id':None,'first_seen_s':points[0]['time_s'],'last_seen_s':points[-1]['time_s'],'observed_frames':len(points),'observation_intervals':[{'start_s':g[0]['time_s'],'end_s':round(g[-1]['time_s']+step,4)} for g in paths],'median_detector_score':round(median(p['detector_score'] for p in points),4),'median_red_pixel_fraction':round(median(reds),4) if reds else None,'evidence_time_s':strongest['time_s'],'locations':sorted({p['location'] for p in points}),'identity_status':'temporary_visual_track; no_cross_recording_or_zone_identity_binding','history_properties':{'reagent_identity':None,'freeze_thaw_count':None,'sample_age':None,'passage_number':None},'review_status':'unreviewed'})
        classes=__import__('collections').Counter(p['label'] for p in points)
        tracks[-1]['label']=classes.most_common(1)[0][0]
        tracks[-1]['class_predictions']=dict(classes)
        tracks[-1]['class_consistency_fraction']=round(classes.most_common(1)[0][1]/len(points),4)
        if len(classes)>1:tracks[-1]['identity_status']+='; class_changed_requires_review'
    (folder/'objects.json').write_text(json.dumps(tracks,indent=2),encoding='utf-8')
    (folder/'coverage.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    insights={'recording_id':name,'event_counts':dict(__import__('collections').Counter(e['record_type'] for e in combined)),'temporary_tracks':len(tracks),'microvariation_candidates':[{'metric':'pipette_plate_proximity_intervals','value':sum(e['action']=='pipette_shaft_endpoint_near_plate' for e in events),'unit':'candidate_intervals','limitations':'Depends on shaft/tip detection and temporary track identity; not a count of verified dispense/aspirate actions.'},{'metric':'visibility_gap_intervals','value':len(alerts),'unit':'intervals','limitations':'Replay prompts from inferred requirements; actual detector misses versus camera occlusion are unresolved.'}],'experiment_response':{'cell_morphology':'not_observed_at_cell_resolution','growth_rate':None,'spontaneous_differentiation':None,'final_outcome':None,'required_inputs':['Time-stamped microscopy images from the same sample and comparable fields','Pixel-size calibration and acquisition metadata','Sample/plate/well IDs','Recorded final outcome and any assay measurements']},'unsupported_critical_properties':['Bottle cap state and exact open duration','Plate lid state and exact uncovered duration','Aspiration versus dispense, volume and well identity','Persistent vessel identity across occlusion and recordings'],'readiness':'Reviewable offline extraction; not a validated autonomous experiment record'}
    insights['experiment_response']['response_related_speech_candidates']=response_mentions
    (folder/'insights.json').write_text(json.dumps(insights,indent=2),encoding='utf-8')
    return {'recording_id':name,**insights['event_counts'],'tracks':len(tracks)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/real_workflow'))
    p.add_argument('--config',type=Path,default=Path('workflow_config.json'))
    args=p.parse_args();config=json.loads(args.config.read_text())
    print(json.dumps([extract_recording(file.parent,config) for file in sorted(args.root.glob('*/extraction_summary.json'))],indent=2))


if __name__=='__main__':main()
