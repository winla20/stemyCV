"""Repeat the local SteMy workflow extraction and build its evidence-review interface."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'data/demo',help='Input directory (default: bundled data/demo recordings)')
    parser.add_argument('--with-temporal',action='store_true')
    parser.add_argument('--skip-transcription',action='store_true')
    return parser.parse_args(argv)


def main():
    args=parse_args()
    source_dir=args.source_dir.resolve()
    if not source_dir.is_dir() or not any(source_dir.glob('*.mov')):
        raise SystemExit(f'No .mov recordings found in {source_dir}')
    if args.with_temporal and not (ROOT/'models/sam2.1_t.pt').exists():
        raise SystemExit('Save the official SAM 2.1 tiny checkpoint to models/sam2.1_t.pt first; see README.md')
    def run(script,*arguments):
        subprocess.run([sys.executable,str(ROOT/script),*map(str,arguments)],cwd=ROOT,check=True)
    run('prepare_workflow.py',source_dir)
    run('workflow_extract.py','--resume','--batch-size','8')
    if not args.skip_transcription:run('workflow_transcribe.py')
    run('audit_workflow.py')
    run('workflow_events.py')
    if args.with_temporal:
        run('workflow_temporal.py')
    run('audit_workflow.py')
    run('build_workflow_viewer.py')


if __name__=='__main__':main()
