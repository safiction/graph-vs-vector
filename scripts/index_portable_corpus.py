"""Index the verified portable corpus with separate processes for memory-heavy stages."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graph_rag.common import file_digest, read_jsonl, write_json


def stage(name, bundle, output):
    from graph_rag.llm import VLLMModel
    chunks = bundle / 'chunks/chunks.jsonl'
    manifest = bundle / 'corpus_manifest.jsonl'
    if name in {'extraction', 'summaries'}:
        model = VLLMModel(weight_manifest=ROOT / 'reports/graph/vllm_weights.json',
                          server_config=ROOT / 'configs/graph_vllm_5080_bounded.json')
    if name == 'extraction':
        from graph_rag.extract import extract
        for attempt in range(3):
            result = extract(chunks, manifest, output, model, workers=8, max_tokens=2048,
                             retry_errors=attempt > 0)
            if result['complete']:
                return
        raise RuntimeError('Unresolved extraction errors; checkpoints retained')
    if name == 'graph':
        from graph_rag.index import build_index
        build_index(chunks, manifest, output, bundle / 'dataset', output / 'index.json')
    elif name == 'summaries':
        from graph_rag.index import summarize
        for attempt in range(2):
            result = summarize(output / 'index.json', output / 'summaries', model, workers=8,
                               retry_errors=attempt > 0, extractive_fallback=True)
            if result['complete']:
                return
        raise RuntimeError('Incomplete summaries; checkpoints retained')
    elif name == 'audit':
        from graph_rag.audit import audit
        result = audit(output / 'index.json', manifest, bundle / 'dataset',
                       output / 'source_audit.json', output / 'summaries')
        if not result['passed']:
            raise RuntimeError('Source audit failed')
    elif name in {'graph-local', 'graph-global'}:
        from graph_rag.retrieve import GraphLocalRetriever, GraphGlobalRetriever
        # Development queries only; no label files enter retrieval or routing.
        retriever = (GraphLocalRetriever(output / 'index.json') if name == 'graph-local' else
                     GraphGlobalRetriever(output / 'index.json', output / 'summaries'))
        queries = read_jsonl(bundle / 'questions/development.queries.jsonl')
        target = output / 'development'
        target.mkdir(exist_ok=True)
        with (target / f'{name}.retrieval.jsonl').open('w') as stream:
            for q in queries:
                stream.write(json.dumps({**q, **retriever.retrieve_with_trace(q['question'], k=10),
                                         'status': 'retrieval_only_pending_shared_generator'}) + '\n')


def run(bundle, output):
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run([sys.executable, str(bundle / 'verify.py'), str(bundle)], check=True)
    metadata = json.loads((bundle / 'bundle.json').read_text())
    settings = {'bundle_sha256sums': file_digest(bundle / 'SHA256SUMS'), 'workers': 8,
                'max_tokens': 2048, 'server_config_sha256': file_digest(ROOT / 'configs/graph_vllm_5080_bounded.json'),
                'code_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
                               sorted(list((ROOT / 'src/graph_rag').glob('*.py')) + [Path(__file__).resolve()])}}
    config = output / 'run_config.json'
    if config.exists() and json.loads(config.read_text()) != settings:
        raise ValueError('Run configuration changed; use a new output directory')
    write_json(config, settings)
    started = time.monotonic()
    status = {'state': 'running', 'started_at': datetime.now().astimezone().isoformat(),
              'document_count': metadata['document_count'], 'chunk_count': metadata['chunk_count'], 'workers': 8}
    try:
        for name in ['extraction', 'graph', 'summaries', 'audit', 'graph-local', 'graph-global']:
            status.update(stage=name, checkpointed_items=0)
            write_json(output / 'progress.json', status)
            print(f'STAGE {name}', flush=True)
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--bundle', str(bundle),
                                      '--output', str(output), '--stage', name])
            while child.poll() is None:
                directory = output / ('summaries/records' if name == 'summaries' else 'records')
                status.update(elapsed_seconds=time.monotonic() - started,
                              updated_at=datetime.now().astimezone().isoformat(),
                              checkpointed_items=sum(1 for _ in directory.glob('*.json')))
                write_json(output / 'progress.json', status)
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'{name} exited {child.returncode}; checkpoints preserved')
        status.update(state='complete', stage='complete', completed_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic() - started
        write_json(output / 'progress.json', status)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'data/exports/corpus_15000')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/graph/corpus_15000')
    parser.add_argument('--stage', choices=['extraction', 'graph', 'summaries', 'audit', 'graph-local', 'graph-global'])
    args = parser.parse_args()
    if args.stage:
        stage(args.stage, args.bundle.resolve(), args.output.resolve())
    else:
        run(args.bundle.resolve(), args.output.resolve())
