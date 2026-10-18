"""Check causal visibility prompts against intermittent detections and recovery."""
import unittest
from workflow_visibility import VisibilityMonitor


class VisibilityTests(unittest.TestCase):
    def test_intermitten_detection_cannot_hide_poor_coverage(self):
        monitor=VisibilityMonitor(window_seconds=2)
        spec=[{'key':'plate-step','label':'multiwell plate','basis':'test'}]
        triggered=[]
        for i in range(91):
            objects=[{'label':'multiwell plate','track_id':'plate-1'}] if i%20==0 else []
            triggered.extend(monitor.update(i/30,objects,spec))
        self.assertEqual(len(triggered),1)
        self.assertGreaterEqual(triggered[0]['trigger_s'],2)
        self.assertLess(triggered[0]['window_presence_at_trigger'],.6)

    def test_stable_visibility_does_not_prompt_and_wrong_track_does(self):
        monitor=VisibilityMonitor()
        spec=[{'key':'plate-step','label':'multiwell plate','basis':'test','track_id':'target-plate'}]
        for i in range(91):
            self.assertFalse(monitor.update(i/30,[{'label':'multiwell plate','track_id':'target-plate'}],spec))
        prompts=[]
        for i in range(91,181):
            prompts.extend(monitor.update(i/30,[{'label':'multiwell plate','track_id':'background-plate'}],spec))
        self.assertEqual(len(prompts),1)

    def test_expectation_end_closes_alert_and_stops_prompting(self):
        monitor=VisibilityMonitor()
        spec=[{'key':'plate-step','label':'multiwell plate','basis':'test'}]
        for i in range(91):monitor.update(i/30,[],spec)
        self.assertFalse(monitor.update(3.1,[],[]))
        completed=monitor.finish(5)
        self.assertEqual(len(completed),1)
        self.assertEqual(completed[0]['end_s'],3.1)


if __name__=='__main__':unittest.main()
