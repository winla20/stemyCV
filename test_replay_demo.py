"""Verify video seeking and asset boundaries in the localhost replay server."""
from functools import partial
from http.server import ThreadingHTTPServer
import http.client
from pathlib import Path
import threading
import unittest

from replay_demo import ReplayHandler, DEMO


class QuietHandler(ReplayHandler):
    def log_message(self,*args):pass


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(DEMO)))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self,path,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        try:
            connection.request('GET',path,headers=headers or {})
            response=connection.getresponse()
            return response.status,dict(response.getheaders()),response.read()
        finally:connection.close()

    def test_video_range_matches_saved_bytes(self):
        name='video-94_singular_display/source_browser.mp4'
        status,headers,body=self.request('/'+name,{'Range':'bytes=16-31'})
        self.assertEqual(status,206)
        self.assertEqual(headers['Content-Length'],'16')
        with (DEMO/name).open('rb') as stream:
            stream.seek(16)
            self.assertEqual(body,stream.read(16))

    def test_rejects_invalid_range_and_does_not_serve_repository_files(self):
        status,_,_=self.request('/video-94_singular_display/source_browser.mp4',{'Range':'bytes=9-1'})
        self.assertEqual(status,416)
        status,_,_=self.request('/../workflow_config.json')
        self.assertEqual(status,404)

    def test_event_api_fixture_is_accessible(self):
        status,headers,body=self.request('/video-94_singular_display/events.jsonl')
        self.assertEqual(status,200)
        self.assertIn('ndjson',headers['Content-Type'])
        self.assertIn(b'pending_review',body)


if __name__=='__main__':unittest.main()
