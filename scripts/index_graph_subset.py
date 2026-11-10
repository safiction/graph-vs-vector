"""Prepare a whole-document graph subset, or index it as a resumable local job."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import fcntl
import json
from pathlib import Path
import random
import sys
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graph_rag.common import file_digest, read_jsonl, verified_source, write_json


def save_fixed(path, content):
    path = Path(path)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f'Existing fixed input differs: {path}')
    if not path.exists():
        path.write_bytes(content)


def select_documents(records, count, required=(), seed=42):
    by_source = defaultdict(list)
    for record in records:
        by_source[record['path'].split('/')[0]].append(record['source_document_id'])
    ids = {r['source_document_id'] for r in records}
    if len(ids) != len(records) or not 1 <= count <= len(records) or not set(required) <= ids:
        raise ValueError('Invalid document count, duplicate IDs or unknown mandatory documents')
    quotas = {s: count * len(rows) // len(records) for s, rows in by_source.items()}
    order = sorted(by_source, key=lambda s: (-(count * len(by_source[s]) % len(records)), s))
    for source in order[:count - sum(quotas.values())]:
        quotas[source] += 1
    rng, selected = random.Random(seed), set()
    for source, rows in sorted(by_source.items()):
        mandatory = sorted(set(rows) & set(required))
        if len(mandatory) > quotas[source]:
            raise ValueError(f'Mandatory documents exceed quota for {source}')
        selected.update(mandatory)
        selected.update(rng.sample(sorted(set(rows) - set(mandatory)), quotas[source] - len(mandatory)))
    return selected, quotas


def prepare(output, count=2000, seed=42):
    output = Path(output).resolve()
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    parent_manifest = ROOT / 'manifests/pilot_corpus.jsonl'
    parent_chunks = ROOT / 'data/graph/shared_chunks/chunks.jsonl'
    chunk_config = json.loads((ROOT / 'manifests/pilot_chunk_config.json').read_text())
    if file_digest(parent_chunks) != chunk_config['chunks_sha256']:
        raise ValueError('Parent chunks differ from the agreed shared chunks')
    records = read_jsonl(parent_manifest)
    questions = json.loads((ROOT / 'manifests/questions.json').read_text())
    question_file = ROOT.parent / 'EnterpriseRAG-Bench/questions.jsonl'
    verified_source(question_file.read_bytes(), questions['source_sha256'])
    dev = set(questions['development_ids'])
    required = {doc for q in read_jsonl(question_file) if q['question_id'] in dev for doc in q['expected_doc_ids']}
    selected, quotas = select_documents(records, count, required, seed)
    doc_source = {r['source_document_id']: r['path'].split('/')[0] for r in records}
    manifest_lines = [line for line in parent_manifest.read_bytes().splitlines(keepends=True)
                      if json.loads(line)['source_document_id'] in selected]
    chunk_lines, chunk_counts, documents_with_chunks = [], Counter(), set()
    for line in parent_chunks.read_bytes().splitlines(keepends=True):
        chunk = json.loads(line)
        if chunk['source_document_id'] in selected:
            chunk_lines.append(line)
            chunk_counts[doc_source[chunk['source_document_id']]] += 1
            documents_with_chunks.add(chunk['source_document_id'])
    if documents_with_chunks != selected:
        raise ValueError('Not every selected document has shared chunks')
    manifest, chunks = inputs / 'corpus_manifest.jsonl', inputs / 'chunks.jsonl'
    save_fixed(manifest, b''.join(manifest_lines))
    save_fixed(chunks, b''.join(chunk_lines))
    save_fixed(inputs / 'document_ids.txt', (''.join(d + '\n' for d in sorted(selected))).encode())
    costs = json.loads((ROOT / 'reports/graph/vllm_cost.json').read_text())
    parallelism = costs['parallel_calibration']['measured_effective_parallelism']
    extraction_seconds = sum(chunk_counts[s] * costs['strata'][s]['seconds_mean'] for s in chunk_counts) / parallelism
    summary_seconds = len(chunk_lines) * costs['actual_summaries']['elapsed_wall_seconds'] / costs['pilot_chunks_attempted']
    plan = {'purpose': 'graph_bounded_corpus_candidate_not_global_comparison_freeze',
            'document_count': len(selected), 'chunk_count': len(chunk_lines), 'seed': seed,
            'selection_method': 'source-proportional largest-remainder quotas; include development gold documents; seeded random sampling',
            'source_document_counts': dict(sorted(quotas.items())), 'source_chunk_counts': dict(sorted(chunk_counts.items())),
            'development_gold_documents_included': len(required),
            'manifest_sha256': file_digest(manifest), 'chunks_sha256': file_digest(chunks),
            'document_ids_sha256': file_digest(inputs / 'document_ids.txt'),
            'parent_manifest_sha256': file_digest(parent_manifest), 'parent_chunks_sha256': file_digest(parent_chunks),
            'parent_chunk_version': chunk_config['chunk_version'],
            'chunk_policy': 'all original chunks of every selected document, unchanged text, IDs and tokenization',
            'estimated_extraction_hours': extraction_seconds / 3600,
            'rough_summary_hours': summary_seconds / 3600,
            'rough_total_hours': (extraction_seconds + summary_seconds) / 3600,
            'estimation_limits': 'Small-pilot extrapolation; community merging, retries and longer output caps can change duration.',
            'workers': 16, 'extraction_max_tokens': 2048,
            'summary_failure_policy': 'exact extractive excerpts after two failed model synthesis attempts',
            'paths': {'output': str(output), 'manifest': str(manifest), 'chunks': str(chunks)}}
    save_fixed(output / 'plan.json', (json.dumps(plan, ensure_ascii=False, indent=2) + '\n').encode())
    return plan


def run(output):
    from graph_rag.llm import VLLMModel
    from graph_rag.extract import extract
    from graph_rag.index import build_index, summarize
    from graph_rag.audit import audit
    from graph_rag.retrieve import GraphLocalRetriever, GraphGlobalRetriever
    from graph_rag.run import retrieval_demo

    output = Path(output).resolve()
    plan = json.loads((output / 'plan.json').read_text())
    lock_file = (output / '.run.lock').open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest, chunks = Path(plan['paths']['manifest']), Path(plan['paths']['chunks'])
    if file_digest(manifest) != plan['manifest_sha256'] or file_digest(chunks) != plan['chunks_sha256']:
        raise ValueError('Fixed subset inputs changed')
    model = VLLMModel(weight_manifest=ROOT / 'reports/graph/vllm_weights.json', server_config=ROOT / 'configs/graph_vllm_5080.json')
    configuration = {'plan_sha256': file_digest(output / 'plan.json'), 'model': model.identity,
                     'implementation_sha256': {str(p.relative_to(ROOT)): file_digest(p)
                                              for p in sorted(list((ROOT / 'src/graph_rag').glob('*.py')) + [Path(__file__).resolve()])}}
    save_fixed(output / 'run_config.json', (json.dumps(configuration, ensure_ascii=False, indent=2) + '\n').encode())
    state = {'state': 'running', 'stage': 'starting', 'started_at': datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
             'document_count': plan['document_count'], 'chunk_count': plan['chunk_count'], 'workers': plan['workers']}
    stopped, state_lock = threading.Event(), threading.Lock()
    started = time.monotonic()

    def checkpoint_count(stage):
        directory = output / ('summaries/records' if stage == 'summaries' else 'records')
        return sum(1 for _ in directory.glob('*.json'))

    def set_stage(stage, total=None):
        with state_lock:
            state.update(stage=stage, stage_started=time.monotonic(), stage_total=total,
                         stage_initial_checkpoints=checkpoint_count(stage))
            write_json(output / 'run_status.json', state)
        print(f'STAGE {stage}', flush=True)

    def heartbeat():
        while not stopped.is_set():
            with state_lock:
                snapshot = dict(state)
            snapshot['elapsed_seconds'] = time.monotonic() - started
            snapshot['updated_at'] = datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
            snapshot['checkpointed_items'] = checkpoint_count(snapshot['stage'])
            write_json(output / 'progress.json', snapshot)
            stopped.wait(10)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        set_stage('extraction', plan['chunk_count'])
        extraction = extract(chunks, manifest, output, model, workers=plan['workers'], max_tokens=plan['extraction_max_tokens'])
        for attempt in range(2):
            if extraction['complete']:
                break
            set_stage('extraction_retry', extraction['error_chunks'])
            extraction = extract(chunks, manifest, output, model, workers=plan['workers'],
                                 max_tokens=plan['extraction_max_tokens'], retry_errors=True)
        if not extraction['complete']:
            raise RuntimeError(f'Extraction still has {extraction["error_chunks"]} failed chunks; no complete index is published')
        set_stage('building_graph')
        index = build_index(chunks, manifest, output, ROOT.parent / 'EnterpriseRAG-Bench', output / 'index.json')
        set_stage('summaries', len(index['communities']))
        summary = summarize(output / 'index.json', output / 'summaries', model, workers=plan['workers'], extractive_fallback=True)
        if not summary['complete']:
            summary = summarize(output / 'index.json', output / 'summaries', model, workers=plan['workers'],
                                retry_errors=True, extractive_fallback=True)
        if not summary['complete']:
            raise RuntimeError('Community summaries remain incomplete')
        set_stage('auditing_sources')
        source_audit = audit(output / 'index.json', manifest, ROOT.parent / 'EnterpriseRAG-Bench',
                             output / 'source_audit.json', output / 'summaries')
        if not source_audit['passed']:
            raise RuntimeError('Source audit failed')
        set_stage('development_retrieval')
        retrieval_demo([GraphLocalRetriever(output / 'index.json'), GraphGlobalRetriever(output / 'index.json', output / 'summaries')],
                       ROOT / 'manifests/questions.json', ROOT.parent / 'EnterpriseRAG-Bench', output / 'development')
        with state_lock:
            state.update(state='complete', stage='complete', elapsed_seconds=time.monotonic() - started,
                         completed_at=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
                         entities=len(index['nodes']), relations=len(index['edges']),
                         communities=len(index['communities']), summary_fallbacks=summary['extractive_fallbacks'],
                         source_audit_passed=True)
            write_json(output / 'run_status.json', state)
        print('INDEXING COMPLETE', flush=True)
    except Exception as exc:
        with state_lock:
            state.update(state='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic() - started)
            write_json(output / 'run_status.json', state)
        raise
    finally:
        stopped.set()
        thread.join(timeout=15)
        with state_lock:
            write_json(output / 'progress.json', state)
        lock_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=ROOT / 'data/graph/bounded_2000')
    parser.add_argument('--documents', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.output, args.documents, args.seed), ensure_ascii=False, indent=2))
    else:
        run(args.output)
