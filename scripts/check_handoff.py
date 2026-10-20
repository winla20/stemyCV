"""Validate source syntax and the raw-event contract without loading any ML model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def validate_event(event, validator):
    validator.validate(event)
    if event['end_s'] <= event['start_s']:
        raise ValueError('Event must end after it starts')
    evidence = event['evidence']
    if (evidence['start_s'], evidence['end_s']) != (event['start_s'], event['end_s']):
        raise ValueError('Evidence range must match the candidate range')
    if 'trigger_s' in event and not event['start_s'] <= event['trigger_s'] <= event['end_s']:
        raise ValueError('Visibility trigger must fall inside the event range')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, help='Optionally validate a local events.jsonl')
    args = parser.parse_args()
    sources = list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    for path in ROOT.glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'))
    demo = ROOT / 'data/demo'
    inputs = json.loads((demo / 'manifest.json').read_text(encoding='utf-8'))['files']
    configured = set(json.loads((ROOT / 'workflow_config.json').read_text())['recordings'])
    if {Path(entry['filename']).stem for entry in inputs} != configured:
        raise ValueError('Bundled recordings do not match the demo configuration')
    for entry in inputs:
        path = demo / entry['filename']
        if path.stat().st_size != entry['size_bytes']:
            raise ValueError(f'Bundled video size mismatch: {path.name}')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                raise ValueError(f'Bundled video hash mismatch: {path.name}')
    schema = json.loads((ROOT / 'schemas/event.schema.json').read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validate_event(json.loads((ROOT / 'examples/event.json').read_text(encoding='utf-8')), validator)
    count = 0
    if args.events:
        seen = set()
        with args.events.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                validate_event(event, validator)
                if event['event_id'] in seen:
                    raise ValueError('Duplicate event ID in recording')
                seen.add(event['event_id'])
                count += 1
    html = (ROOT / 'workflow_viewer.html').read_text(encoding='utf-8')
    script = html.rsplit('<script>', 1)[1].split('</script>', 1)[0]
    subprocess.run(['node', '--check'], input=script, text=True, check=True)
    print(f'Passed: {len(sources)} Python files, JSON config, {len(inputs)} bundled video hashes, event schema/example, viewer JS; {count} supplied events')


if __name__ == '__main__':
    main()
