"""Causal visibility monitoring shared by replay and a future live capture loop."""
from collections import deque


class VisibilityMonitor:
    def __init__(self, window_seconds=2, minimum_presence=.6, recovery_presence=.8):
        self.window_seconds=window_seconds
        self.minimum_presence=minimum_presence
        self.recovery_presence=recovery_presence
        self.states={}
        self.completed=[]

    def update(self, time_s, objects, expectations):
        """Use only current/past evidence. Expectations must come from procedure context."""
        prompts=[]
        expected={spec['key']:spec for spec in expectations}
        for key in list(self.states):
            if key not in expected:
                self._close(key,time_s)
                del self.states[key]
        for key,spec in expected.items():
            state=self.states.setdefault(key,{'first_s':time_s,'history':deque(),'alert':None})
            present=any(o['label']==spec['label'] and (not spec.get('track_id') or o['track_id']==spec['track_id']) for o in objects)
            state['history'].append((time_s,present))
            while state['history'] and state['history'][0][0]<time_s-self.window_seconds-1e-4:
                state['history'].popleft()
            if time_s-state['first_s']<self.window_seconds-1e-4:continue
            fraction=sum(seen for _,seen in state['history'])/len(state['history'])
            if state['alert'] is None and fraction<self.minimum_presence:
                state['alert']={'record_type':'visibility_prompt_candidate','action':'request_better_view','start_s':round(time_s-self.window_seconds,4),'trigger_s':round(time_s,4),'label':spec['label'],'description':f'Show the {spec["label"]} clearly before documenting the next manipulation.','reason':'Too little detector evidence over the preceding visibility window. Occlusion, out-of-view, wrong identity, and detection failure remain possible.','basis':spec['basis'],'window_presence_at_trigger':round(fraction,4),'monitor_window_seconds':self.window_seconds,'minimum_presence_policy':self.minimum_presence,'event_confidence':None,'review_status':'unreviewed','live_delivery_status':'replay_only; no_operator_notification_sent'}
                prompts.append(state['alert'].copy())
            elif state['alert'] is not None and fraction>=self.recovery_presence:
                self._close(key,time_s)
        return prompts

    def _close(self,key,time_s):
        state=self.states[key]
        if state['alert'] is not None:
            alert=state['alert']
            alert['end_s']=round(time_s,4)
            alert['duration_s']=round(time_s-alert['start_s'],3)
            self.completed.append(alert)
            state['alert']=None

    def finish(self,time_s):
        for key in self.states:self._close(key,time_s)
        return self.completed
