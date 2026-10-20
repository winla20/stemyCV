"""Transcribe all review recordings locally with retained ASR uncertainty."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import whisper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/real_workflow'))
    p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(4)
    model=whisper.load_model('base.en',device=args.device,download_root='models')
    ffmpeg=shutil.which('ffmpeg') or str(Path.home()/'AppData/Local/Programs/Stremio/ffmpeg.exe')
    for source in sorted(args.root.glob('*/source_browser.mp4')):
        target=source.parent/'transcript.json'
        if target.exists():continue
        raw=subprocess.run([ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-i',str(source),'-f','f32le','-ac','1','-ar','16000','pipe:1'],capture_output=True,check=True).stdout
        samples=np.frombuffer(raw,dtype=np.float32).copy()
        print(f'Transcribing {source.parent.name}',flush=True)
        result=model.transcribe(samples,language='en',fp16=args.device!='cpu',verbose=False,condition_on_previous_text=False)
        segments=[]
        for segment in result['segments']:
            uncertain=segment['no_speech_prob']>.6 or segment['avg_logprob']<-1 or segment['compression_ratio']>2.4
            segments.append({'start_s':round(segment['start'],3),'end_s':round(segment['end'],3),'text':segment['text'].strip(),'avg_logprob':round(segment['avg_logprob'],4),'no_speech_prob':round(segment['no_speech_prob'],4),'compression_ratio':round(segment['compression_ratio'],3),'evidence_status':'low_confidence_asr' if uncertain else 'asr_candidate','review_status':'unreviewed','interpretation':'Spoken words do not establish that an action occurred.'})
        target.write_text(json.dumps({'recording_id':source.parent.name,'model':'whisper-base.en','device':args.device,'segments':segments,'review_status':'unreviewed'},indent=2),encoding='utf-8')
        print(f'{source.parent.name}: {len(segments)} ASR segments',flush=True)


if __name__=='__main__':main()
