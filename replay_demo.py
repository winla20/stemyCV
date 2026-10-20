"""Serve saved demo results on localhost using only Python's standard library."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEMO = ROOT/'demo_results'


class ReplayHandler(SimpleHTTPRequestHandler):
    """Single byte ranges let the video player seek without downloading entire clips."""
    def send_head(self):
        self.byte_range = None
        path = Path(self.translate_path(self.path)).resolve()
        base = Path(self.directory).resolve()
        if path != base and base not in path.parents:
            self.send_error(403)
            return None
        if path.is_dir():
            path = path/'index.html'
        if not path.is_file():
            self.send_error(404)
            return None
        size = path.stat().st_size
        request_range = self.headers.get('Range')
        start, end = 0, size-1
        if request_range:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)',request_range)
            valid = bool(match and any(match.groups()))
            if valid:
                left,right = match.groups()
                if left:
                    start=int(left);end=min(int(right),size-1) if right else size-1
                else:
                    start=max(0,size-int(right))
                valid = size>0 and 0<=start<=end<size
            if not valid:
                self.send_response(416)
                self.send_header('Content-Range',f'bytes */{size}')
                self.send_header('Content-Length','0')
                self.end_headers()
                return None
            self.byte_range = start,end
        stream = path.open('rb')
        self.send_response(206 if request_range else 200)
        self.send_header('Content-Type','application/x-ndjson' if path.suffix=='.jsonl' else self.guess_type(str(path)))
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Cache-Control','no-cache')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Last-Modified',self.date_time_string(path.stat().st_mtime))
        if request_range:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        stream.seek(start)
        return stream

    def copyfile(self,source,outputfile):
        if self.byte_range is None:
            return super().copyfile(source,outputfile)
        remaining=self.byte_range[1]-self.byte_range[0]+1
        while remaining:
            chunk=source.read(min(1024*1024,remaining))
            if not chunk:break
            outputfile.write(chunk);remaining-=len(chunk)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--build-only',action='store_true',help='Generate the local viewer without starting a server')
    args=parser.parse_args()
    if not (DEMO/'manifest.json').is_file():
        raise SystemExit('Processed demo missing. Pull the complete repository first.')
    subprocess.run([sys.executable,str(ROOT/'build_workflow_viewer.py'),'--root',str(DEMO),'--config',str(ROOT/'workflow_config.json')],cwd=ROOT,check=True)
    if args.build_only:return
    handler=partial(ReplayHandler,directory=str(DEMO))
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
        print(f'Open http://127.0.0.1:{server.server_port} — saved results, no inference. Ctrl+C to stop.',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__=='__main__':main()
