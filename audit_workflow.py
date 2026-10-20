"""Check evidence integrity and demote ASR with implausible timings or repetition."""
from collections import Counter
import json
import hashlib
from pathlib import Path
import argparse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('output/real_workflow'))
    root=parser.parse_args().root;reports=[]
    for folder in sorted(root.glob('*_singular_display')):
        manifest_path=folder/'source_manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            for key,digest_key in [('original_path','original_sha256'),('review_video','review_video_sha256')]:
                if manifest.get(digest_key):
                    with Path(manifest[key]).open('rb') as source:
                        assert hashlib.file_digest(source,'sha256').hexdigest()==manifest[digest_key], 'Source evidence hash mismatch'
        transcript=folder/'transcript.json'
        if transcript.exists():
            data=json.loads(transcript.read_text());counts=Counter(s['text'].lower().strip() for s in data['segments'])
            for s in data['segments']:
                if counts[s['text'].lower().strip()]>=3:
                    s['evidence_status']='low_confidence_asr';s['audit_reason']='repeated_short_text; possible_ASR_hallucination'
                if (folder/'extraction_summary.json').exists():
                    duration=json.loads((folder/'extraction_summary.json').read_text())['duration_s']
                    if s['start_s']>=duration or s['end_s']>duration+.5:
                        s['evidence_status']='low_confidence_asr';s['audit_reason']='ASR_time_range_extends_past_video; not_valid_visual_evidence'
            transcript.write_text(json.dumps(data,indent=2),encoding='utf-8')
        if not (folder/'events.jsonl').exists():continue
        summary=json.loads((folder/'extraction_summary.json').read_text())
        summary['source_manifest']=json.loads(manifest_path.read_text())
        if summary.get('detector_device')=='cpu' and 'host_model_device' not in summary:
            summary['host_model_device']='cpu'
            summary['detector_device']=None
            summary['device_reporting_note']='Original single-frame run recorded the host model device, not the separate inference backend. Requested GPU device was 0.'
        (folder/'extraction_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        assert len(frames)==summary['processed_frames']
        assert [f['frame_index'] for f in frames]==list(range(len(frames)))
        assert all(abs(f['time_s']-f['frame_index']/30)<.0001 for f in frames)
        assert len({e['event_id'] for e in events})==len(events)
        assert all(e['source_sha256']==summary['source_manifest']['original_sha256'] for e in events)
        assert all(0<=e['start_s']<e['end_s']<=summary['duration_s']+.0001 for e in events)
        assert all(e['review_status']=='unreviewed' and e['memory_status']=='pending_review' for e in events)
        assert all(e.get('trigger_s',e['start_s'])>=e['start_s'] for e in events)
        if (folder/'temporal_frames.jsonl').exists():
            temporal=[json.loads(line) for line in (folder/'temporal_frames.jsonl').read_text().splitlines()]
            assert all(0<=f['frame_index']<len(frames) for f in temporal)
            assert all(abs(f['time_s']-f['frame_index']/30)<.0001 for f in temporal)
        reports.append({'recording_id':folder.name,'frames':len(frames),'events':len(events),'evidence_integrity':'passed','action_accuracy':'not_evaluated_against_human_ground_truth'})
    root.joinpath('validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    html=Path('workflow_viewer.html').read_text(encoding='utf-8')
    script=html.rsplit('<script>',1)[1].split('</script>',1)[0]
    script_path=root/'viewer_check.js';script_path.write_text(script,encoding='utf-8')
    # JS syntax is checked by scripts/check_handoff.py in development/CI.
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
