"""Repeat the local SteMy workflow extraction and build its evidence-review interface."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'data/demo',help='Input directory (default: bundled data/demo recordings)')
    parser.add_argument('--with-temporal',action='store_true')
    parser.add_argument('--skip-transcription',action='store_true')
    parser.add_argument('--device',default='auto',help='Vision device: auto, cpu, mps, cuda:N')
    parser.add_argument('--speech-device',default='auto',help='Whisper device: auto, cpu, cuda:N (auto uses CPU on Macs)')
    parser.add_argument('--batch-size',type=int,help='Default: CUDA 8, MPS/CPU 1')
    parser.add_argument('--output',type=Path,default=ROOT/'output/real_workflow')
    parser.add_argument('--config',type=Path,default=ROOT/'workflow_config.json')
    return parser.parse_args(argv)


def main():
    args=parse_args()
    source_dir=args.source_dir.resolve()
    output=args.output.resolve()
    config=args.config.resolve()
    if output==ROOT/'demo_results' or (ROOT/'demo_results') in output.parents:
        raise SystemExit('Keep bundled demo_results immutable; use --output output/NEW_RUN.')
    if not source_dir.is_dir() or not any(source_dir.glob('*.mov')):
        raise SystemExit(f'No .mov recordings found in {source_dir}')
    def run(script,*arguments):
        subprocess.run([sys.executable,str(ROOT/script),*map(str,arguments)],cwd=ROOT,check=True)
    color_policy=json.loads(config.read_text(encoding='utf-8')).get('review_color_policy','preserve')
    run('prepare_workflow.py',source_dir,'--output',output,'--color-policy',color_policy)
    batch=['--batch-size',args.batch_size] if args.batch_size is not None else []
    run('workflow_extract.py','--root',output,'--config',config,'--resume','--device',args.device,*batch)
    if not args.skip_transcription:run('workflow_transcribe.py','--root',output,'--device',args.speech_device)
    run('audit_workflow.py','--root',output)
    run('workflow_events.py','--root',output,'--config',config)
    if args.with_temporal:
        run('workflow_temporal.py','--folder',output/'video-94_singular_display','--device',args.device)
    run('audit_workflow.py','--root',output)
    run('build_workflow_viewer.py','--root',output,'--config',config)


if __name__=='__main__':main()
