"""Repeat the local SteMy workflow extraction and build its evidence-review interface."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--with-temporal',action='store_true')
    parser.add_argument('--skip-transcription',action='store_true')
    args=parser.parse_args()
    def run(script,*arguments):
        subprocess.run([sys.executable,script,*map(str,arguments)],check=True)
    run('prepare_workflow.py',args.source_dir)
    run('workflow_extract.py','--resume','--batch-size','8')
    if not args.skip_transcription:run('workflow_transcribe.py')
    run('audit_workflow.py')
    run('workflow_events.py')
    if args.with_temporal:
        if not Path('models/sam2.1_t.pt').exists():raise RuntimeError('Download the official Ultralytics SAM 2.1 tiny checkpoint into models/sam2.1_t.pt first')
        run('workflow_temporal.py')
    run('audit_workflow.py')
    run('build_workflow_viewer.py')


if __name__=='__main__':main()
