"""Benchmark local vLLM concurrency on the fixed pilot, then run quality probes."""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graph_rag.common import write_json, file_digest, digest
from graph_rag.llm import VLLMModel
from graph_rag.extract import EXTRACTION_PROMPT, EXTRACTION_SCHEMA, validate_extraction
from graph_rag.benchmark import profile_extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8001')
    parser.add_argument('--weight-manifest', type=Path, required=True)
    parser.add_argument('--server-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 4, 8, 16])
    args = parser.parse_args()
    output = args.output.resolve()
    chunks, manifest = ROOT / 'data/graph/shared_chunks/chunks.jsonl', ROOT / 'manifests/pilot_corpus.jsonl'
    sample = ROOT / 'data/graph/pilot_sample.json'
    deadline = time.monotonic() + 300
    while True:
        try:
            with urllib.request.urlopen(args.endpoint + '/health', timeout=3) as response:
                if response.status == 200:
                    break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Local vLLM did not become healthy in 300 seconds')
            time.sleep(3)
    model = VLLMModel(endpoint=args.endpoint, weight_manifest=args.weight_manifest, server_config=args.server_config)
    warmup, metrics = model.chat(EXTRACTION_PROMPT, 'Project Cedar belongs to Northwind Labs.', EXTRACTION_SCHEMA)
    write_json(output / 'warmup.json', {'raw': warmup, 'metrics': metrics, 'model': model.identity})
    random_sample = json.loads(sample.read_text())
    random_sample['chunk_ids'] = random_sample['random_chunk_ids']
    random_sample['demo_chunk_ids'] = []
    write_json(output / 'random_sample.json', random_sample)
    results = []
    for workers in args.workers:
        directory = output / f'workers_{workers}'
        if (directory / 'profile.json').exists():
            result = json.loads((directory / 'profile.json').read_text())
            if result['model'] != model.identity:
                raise ValueError('Existing benchmark uses another model/configuration')
            extraction_config = json.loads((directory / 'extraction_config.json').read_text())
            if extraction_config['prompt'] != EXTRACTION_PROMPT:
                raise ValueError('Existing benchmark uses another extraction prompt')
        else:
            result = profile_extract(chunks, manifest, directory, model, output / 'random_sample.json', workers)
        if not result['accepted_entity_mentions'] or not result['accepted_relations']:
            raise RuntimeError('Quality gate: extraction collapsed to empty graphs; do not report this as a speedup')
        results.append(result)
        print(json.dumps({'workers': workers, 'seconds': result['elapsed_wall_seconds'],
                          'tokens_per_second': result['completion_tokens_per_second'],
                          'gpu_mean_percent': result['gpu'].get('mean_gpu_utilization_percent'),
                          'accepted_relations': result['accepted_relations'], 'status': result['status']}), flush=True)
    successful = [r for r in results if r['status']['complete']]
    if not successful:
        raise RuntimeError('No concurrency setting completed extraction')
    best = max(successful, key=lambda r: r['chunks_per_second'])
    write_json(output / 'concurrency.json', {'selection_metric': 'completed chunks / measured batch wall seconds',
                'sample_sha256': file_digest(output / 'random_sample.json'), 'selected_workers': best['workers'],
                'results': [{**r, 'gpu': {k: v for k, v in r['gpu'].items() if k != 'samples'}} for r in results]})
    cases = json.loads((ROOT / 'tests/fixtures/graph_extraction_quality.json').read_text())

    def probe(case):
        path = output / 'quality_records' / (case['id'] + '.json')
        fingerprint = digest([model.identity, EXTRACTION_PROMPT, EXTRACTION_SCHEMA, case])
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['fingerprint'] != fingerprint:
                raise ValueError('Quality probe settings changed')
            return previous
        raw, metrics = model.chat(EXTRACTION_PROMPT, case['text'], EXTRACTION_SCHEMA)
        result = {**case, 'raw': raw, 'metrics': metrics, 'fingerprint': fingerprint}
        try:
            if metrics['done_reason'] == 'length':
                raise ValueError('Truncated extraction')
            result['validated'] = validate_extraction(json.loads(raw), case['text'])
            result['status'] = 'ok'
        except (ValueError, TypeError, KeyError) as exc:
            result.update(status='error', error=str(exc))
        write_json(path, result)
        return result

    with ThreadPoolExecutor(max_workers=best['workers']) as executor:
        probes = list(executor.map(probe, cases['cases']))
    write_json(output / 'quality_probes.json', {'purpose': cases['purpose'], 'model': model.identity, 'cases': probes})
    if any(p['status'] != 'ok' or not p['validated']['relations'] for p in probes):
        raise RuntimeError('Quality gate: every explicit-fact probe must produce a valid, nonempty extraction')
    if not (output / 'pilot/profile.json').exists():
        profile_extract(chunks, manifest, output / 'pilot', model, sample, best['workers'])
    print(f'Completed; selected concurrency={best["workers"]}; quality probes require semantic review', flush=True)


if __name__ == '__main__':
    main()
