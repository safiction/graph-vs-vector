"""Audit the completed two-week outputs and export a small reviewable report."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_rag.common import environment, file_digest, read_json, read_jsonl, write_json, write_jsonl
from baseline_rag.corpus import Corpus
from baseline_rag.generation import make_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/baseline_15000.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/baselines/implementation.json'))
    args = parser.parse_args()
    config = read_json(args.config)
    root = Path(config['output'])
    corpus = Corpus(config['bundle'])
    tokenizer = make_adapter(config['generation_settings'])
    metadata = read_json(root / 'questions/metadata.json')
    corpus_map = {c['chunk_id']: c for c in corpus.chunks}
    report = {'corpus': corpus.audit(), 'environment': environment(),
              'generation_settings': config['generation_settings'], 'runs': {}, 'artifacts': {},
              'tests': read_json(root / 'tests.json') if (root / 'tests.json').exists() else None}
    import torch
    report['hardware'] = {'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}
    report['indexing'] = {name: read_json(root / 'indexes' / name / 'metadata.json') for name in ['bm25', 'vector']}
    frozen = read_json(root / 'experiment.json')
    comparison = {'shared_chunks_sha256': corpus.identity['chunks_sha256'],
        'frozen_source_manifest_sha256': corpus.identity['source_manifest_sha256'],
        'preliminary_question_ids': frozen['preliminary_question_ids'],
        'retrieval_k': frozen['retrieval_k'], 'context_token_budget': frozen['context_token_budget'],
        'generation_settings': frozen['generation_settings']}
    write_json(root / 'comparison.json', comparison)
    write_jsonl(root / 'questions/preliminary.graph.jsonl', [
        {**q, **metadata[q['question_id']]} for q in read_jsonl(root / 'questions/preliminary.queries.jsonl')])
    attempts = successes = 0
    problems = []
    for split, expected in [('development', 12), ('preliminary', 40)]:
        summary = read_json(root / 'runs' / split / 'summary.json')
        report['runs'][split] = summary
        first_path = root / 'runs' / split / 'first_execution_summary.json'
        if first_path.exists():
            report['runs'][split]['first_execution_elapsed_seconds'] = read_json(first_path)['invocation_elapsed_seconds']
        questions = read_jsonl(root / 'questions' / f'{split}.queries.jsonl')
        expected_ids = {q['question_id'] for q in questions}
        if len(expected_ids) != expected:
            problems.append(f'{split}: incorrect question count')
        if split == 'preliminary' and Counter(metadata[qid]['project_type'] for qid in expected_ids) != Counter({
                'single-fact': 10, 'multi-hop': 10, 'corpus-global': 10, 'fuzzy': 10}):
            problems.append('Preliminary selection is not balanced')
        for name in ['bm25', 'vector']:
            path = root / 'runs' / split / f'{name}.predictions.jsonl'
            rows = read_jsonl(path)
            if len(rows) != expected or {r['question_id'] for r in rows} != expected_ids:
                problems.append(f'{split}/{name}: missing or duplicate attempts')
            for row in rows:
                attempts += 1
                if row['status'] != 'success':
                    problems.append(f'{split}/{name}/{row["question_id"]}: {row.get("error")}')
                    continue
                successes += 1
                context_tokens = tokenizer.count_tokens(row['context'])
                supplied_links = [s for e in row['evidence'] for s in e['source_links']]
                supplied_ids = sorted({s['source_document_id'] for s in supplied_links})
                supplied_chunks = sorted({s['chunk_id'] for s in supplied_links})
                if (context_tokens != row['context_tokens'] or context_tokens > config['context_token_budget'] or
                        supplied_ids != row['final_source_document_ids'] or supplied_chunks != row['chunk_ids']):
                    problems.append(f'{split}/{name}/{row["question_id"]}: budget or provenance mismatch')
                for item in row['evidence']:
                    for link in item['source_links']:
                        chunk = corpus_map.get(link['chunk_id'])
                        if not chunk or chunk['source_document_id'] != link['source_document_id'] or chunk['text'] != item['text']:
                            problems.append(f'{split}/{name}/{row["question_id"]}: source text mismatch')
            report['artifacts'][str(path)] = {'sha256': file_digest(path), 'bytes': path.stat().st_size}
    for relative in ['experiment.json', 'comparison.json', 'questions/selection.json', 'indexes/bm25/index.sqlite',
                     'indexes/bm25/metadata.json', 'indexes/vector/embeddings.npy', 'indexes/vector/metadata.json']:
        path = root / relative
        report['artifacts'][str(path)] = {'sha256': file_digest(path), 'bytes': path.stat().st_size}
    report.update(attempts=attempts, successful_answers=successes, audit_problems=problems,
                  baseline_deliverables_complete=attempts == successes == 104 and not problems)
    startup = root / 'runs/development_startup_failure'
    if startup.exists():
        startup_rows = [read_json(p) for p in (startup / 'attempts').glob('*/*.json')]
        report['startup_diagnostics'] = {'path': str(startup), 'preserved_attempts': len(startup_rows),
            'statuses': dict(Counter(r['status'] for r in startup_rows)),
            'errors': dict(Counter(r.get('error', '') for r in startup_rows)),
            'scope': 'Development startup before compatibility fixes; excluded from the frozen production outputs'}
    report['handoff_blockers'] = []
    profile = root / 'runs/development_fp16_profile'
    if profile.exists():
        profiles = [read_json(p) for p in (profile / 'attempts').glob('*/*.json')]
        report['fp16_development_profile'] = {'path': str(profile), 'preserved_attempts': len(profiles),
            'statuses': dict(Counter(r['status'] for r in profiles)),
            'generation_seconds': [r.get('generation_metrics', {}).get('wall_seconds') for r in profiles],
            'scope': 'Development runtime profiling before NF4 settings were frozen; not preliminary evaluation'}
    if not Path(config['graph_index']).exists():
        report['handoff_blockers'].append('Complete 15k graph index unavailable locally; graph answer runs not executed')
    if not Path(config['graph_summaries']).exists():
        report['handoff_blockers'].append('Complete 15k graph summaries unavailable locally')
    report['handoff_blockers'].append('Answer correctness and human/judge calibration remain with evaluation owner; coverage is not correctness')
    report['handoff_blockers'].append('The 15k sample includes gold documents by construction; corpus-global gold answers may not be exhaustive on this subset')
    cache_check = root / 'cache_validation.json'
    report['cache_validation'] = read_json(cache_check) if cache_check.exists() else None
    write_json(args.output, report)
    print(json.dumps({'report': str(args.output), 'attempts': attempts, 'successful_answers': successes,
                      'baseline_deliverables_complete': report['baseline_deliverables_complete'],
                      'audit_problems': problems}, indent=2))
    if not report['baseline_deliverables_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
