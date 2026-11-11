"""Recover failed checkpoints only, preserving original attempts and explicit overrides."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graph_rag.common import digest, file_digest, write_json
from graph_rag.extract import EXTRACTION_PROMPT, EXTRACTION_SCHEMA, validate_extraction
from graph_rag.llm import VLLMModel

PROFILE = {'repetition_penalty': 1.1, 'max_tokens': 2048,
           'additional_instruction': 'Use short exact entity names, not lists or sentences. Never repeat text to fill a field. Return at most 6 entities and 4 relations. Omit uncertain items. Close the JSON object.'}


class RecoveryModel(VLLMModel):
    source_quotes = None

    def request(self, path, payload=None):
        if path == '/v1/chat/completions':
            payload['repetition_penalty'] = PROFILE['repetition_penalty']
            payload['messages'][0]['content'] += '\nRecovery instruction: ' + PROFILE['additional_instruction']
            if self.source_quotes is not None:
                schema = json.loads(json.dumps(EXTRACTION_SCHEMA))
                schema['properties']['entities']['maxItems'] = 6
                schema['properties']['relations']['maxItems'] = 4
                schema['properties']['relations']['items']['properties']['quote'] = {
                    'type': 'string', 'enum': self.source_quotes}
                payload.pop('response_format', None)
                payload['structured_outputs'] = {'json': schema, 'whitespace_pattern': r'[ \n\t\r]{0,8}'}
        return super().request(path, payload)


def recover(bundle, output):
    import fcntl
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((output / 'extraction_config.json').read_text())
    chunks = bundle / 'chunks/chunks.jsonl'
    if file_digest(chunks) != config['chunks_sha256']:
        raise ValueError('Source chunks changed')
    failed = []
    for path in sorted((output / 'records').glob('*.json')):
        record = json.loads(path.read_text())
        if record['status'] != 'ok':
            failed.append((path, record))
    if not failed:
        print('No failed checkpoints')
        return
    ids = {r['chunk_id'] for _, r in failed}
    texts = {}
    with chunks.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['chunk_id'] in ids:
                texts[row['chunk_id']] = row
    if set(texts) != ids:
        raise ValueError('Missing failed source chunks')
    model = RecoveryModel(weight_manifest=ROOT / 'reports/graph/vllm_weights.json',
                          server_config=ROOT / 'configs/graph_vllm_5080_bounded.json')
    if model.identity != config['model']:
        raise ValueError('Primary model identity changed')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    recovery = output / 'recovery' / stamp
    (recovery / 'original_records').mkdir(parents=True)
    manifest = {'primary_extraction_config_sha256': file_digest(output / 'extraction_config.json'),
                'script_sha256': file_digest(__file__), 'model': model.identity,
                'profile': PROFILE, 'chunk_ids': sorted(ids), 'results': [],
                'reason': 'Targeted retry of repeated/truncated extraction; successful original records untouched. Primary configuration retained; per-attempt overrides explicit.'}
    write_json(recovery / 'manifest.json', manifest)
    for path, record in failed:
        chunk = texts[record['chunk_id']]
        if digest(chunk['text']) != record['text_sha256'] or chunk['source_document_id'] != record['source_document_id']:
            raise ValueError('Checkpoint source mismatch')
        shutil.copyfile(path, recovery / 'original_records' / path.name)
        attempt = {'recovery_profile': PROFILE, 'recovery_manifest': str((recovery / 'manifest.json').relative_to(output))}
        if PROFILE.get('source_quote_policy'):
            lines = chunk['text'].splitlines(keepends=True)
            model.source_quotes = sorted({''.join(lines[i:i+n]).strip() for i in range(len(lines))
                                          for n in range(1, 4)
                                          if 0 < len(''.join(lines[i:i+n]).strip()) <= 650})
            attempt['source_quote_candidates'] = model.source_quotes
        started = time.monotonic()
        try:
            raw, metrics = model.chat(EXTRACTION_PROMPT, chunk['text'], EXTRACTION_SCHEMA, max_tokens=PROFILE['max_tokens'])
            attempt.update(raw=raw, metrics=metrics)
            if metrics['done_reason'] != 'stop':
                raise ValueError(f'Incomplete recovery output: {metrics["done_reason"]}')
            record.update(validate_extraction(json.loads(raw), chunk['text']), status='ok',
                          recovery_manifest=attempt['recovery_manifest'])
            record.pop('error', None)
        except Exception as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {exc}')
        attempt['attempt_wall_seconds'] = time.monotonic() - started
        record['attempts'].append(attempt)
        write_json(path, record)
        result = {'chunk_id': record['chunk_id'], 'status': record['status'],
                  'entities': len(record.get('entities', [])), 'relations': len(record.get('relations', [])),
                  'record_sha256': file_digest(path)}
        manifest['results'].append(result)
        write_json(recovery / 'manifest.json', manifest)
        print(json.dumps(result), flush=True)
    if any(r['status'] != 'ok' for r in manifest['results']):
        raise RuntimeError('Some recovery attempts failed; originals and attempts preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'data/exports/corpus_15000')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/graph/corpus_15000')
    parser.add_argument('--source-quotes', action='store_true', help='Constrain quotations to exact 1–3-line source spans')
    args = parser.parse_args()
    if args.source_quotes:
        PROFILE.update(source_quote_policy='exact nonempty 1–3 consecutive source lines, stripped, max 650 chars',
                       repetition_penalty=1.15)
    recover(args.bundle.resolve(), args.output.resolve())
