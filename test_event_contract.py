"""Boundary checks for ingestion of raw, unreviewed candidate events."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, ValidationError
from scripts.check_handoff import validate_event

ROOT = Path(__file__).parent


class EventContractTests(unittest.TestCase):
    def setUp(self):
        self.event = json.loads((ROOT / 'examples/event.json').read_text())
        self.validator = Draft202012Validator(json.loads((ROOT / 'schemas/event.schema.json').read_text()))

    def test_accepts_candidate_without_promoting_confidence(self):
        validate_event(self.event, self.validator)
        for key, value in [('event_confidence', .99), ('memory_status', 'approved_for_ingestion')]:
            changed = deepcopy(self.event)
            changed[key] = value
            with self.assertRaises(ValidationError):
                validate_event(changed, self.validator)

    def test_rejects_invalid_evidence_range(self):
        self.event['evidence']['end_s'] = 9
        with self.assertRaises(ValueError):
            validate_event(self.event, self.validator)

    def test_prompt_requires_causal_trigger(self):
        self.event.update(record_type='visibility_prompt_candidate', label='multiwell plate')
        with self.assertRaises(ValidationError):
            validate_event(self.event, self.validator)
        self.event['trigger_s'] = 1
        with self.assertRaises(ValueError):
            validate_event(self.event, self.validator)
        self.event['trigger_s'] = 2.5
        validate_event(self.event, self.validator)


if __name__ == '__main__':
    unittest.main()
