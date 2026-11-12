"""Shared retrieval, packing, generation, resumable attempts and coverage exports."""
from collections import defaultdict
import math
from pathlib import Path
import re
import statistics
import time
from graph_rag.retrieve import pack_context
from .common import environment, file_digest, freeze, implementation, output_lock, read_json, read_jsonl, write_json, write_jsonl
from .generation import make_adapter
from .retrieval import load_retriever


def _artifact_identity(config, name):
    root = Path(config['output']) / 'indexes'
    if name in {'bm25', 'vector'}:
        paths = [root / name / 'metadata.json']
        if name == 'vector':
            paths.append(root / 'bm25/metadata.json')
    else:
        paths = [Path(config['graph_index'])]
        if name == 'graph-global':
            paths += [Path(config['graph_summaries']) / 'summary_status.json',
                      Path(config['graph_summaries']) / 'summary_config.json']
    return {str(p): file_digest(p) if p.exists() else None for p in paths}


def validate_evidence(trace, corpus):
    for item in trace['evidence']:
        if not isinstance(item.get('text'), str) or not item['text'].strip() or not item.get('source_links'):
            raise ValueError('Evidence must have text and explicit source links')
        for link in item['source_links']:
            cid, did = link['chunk_id'], link['source_document_id']
            if corpus.get(cid) != did:
                raise ValueError('Evidence provenance is outside the frozen corpus')


def _error(exc):
    return f'{type(exc).__name__}: {exc}'


def _summary(rows):
    success = [r for r in rows if r['status'] == 'success']
    def mean(field):
        values = [r[field] for r in success if field in r]
        return statistics.mean(values) if values else None
    latencies = sorted(r['total_query_seconds'] for r in success)
    return {'attempts': len(rows), 'success': len(success), 'failed': len(rows) - len(success),
            'mean_retrieval_seconds': mean('retrieval_seconds'),
            'mean_generation_seconds': mean('generation_seconds'),
            'mean_total_query_seconds': mean('total_query_seconds'),
            'p95_total_query_seconds': latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None,
            'prompt_tokens': sum(r.get('generation_metrics', {}).get('prompt_tokens') or 0 for r in success),
            'completion_tokens': sum(r.get('generation_metrics', {}).get('completion_tokens') or 0 for r in success),
            'answers_at_output_limit': sum(r.get('generation_metrics', {}).get('finish_reason') == 'length' for r in success)}


def export_results(config, split, names, output, questions):
    """Scoring happens after inference; labels are never supplied to generation."""
    from coverage import score_coverage
    output = Path(output)
    bundle = Path(config['bundle'])
    metadata = read_json(Path(config['output']) / 'questions/metadata.json')
    labels = {q['question_id']: q for fold in ['development', 'fold_a', 'fold_b']
              for q in read_jsonl(bundle / 'questions' / f'{fold}.labels.jsonl')}
    exclusions = read_json(bundle / 'questions/coverage_exclusions.json')
    report = {'split': split, 'systems': {}, 'correctness': 'Not graded; human/judge calibration is a separate evaluation task'}
    for name in names:
        records, scores, groups = [], [], defaultdict(list)
        for question in questions:
            qid = question['question_id']
            path = output / 'attempts' / name / f'{qid}.json'
            if not path.exists():
                continue
            row = read_json(path)
            records.append(row)
            if row['status'] == 'success':
                score = score_coverage(qid, labels[qid]['expected_doc_ids'], row['final_source_document_ids'], exclusions)
            else:
                score = {'coverage': None, 'full_coverage': None, 'source_count': None,
                         'coverage_na_reason': 'Failed run; no valid final context'}
            scored = {'question_id': qid, **metadata[qid], 'status': row['status'], **score}
            scores.append(scored)
            groups[metadata[qid]['project_type']].append(scored)
        write_jsonl(output / f'{name}.predictions.jsonl', records)
        coverage_summary = {}
        for kind, group in sorted(groups.items()):
            applicable = [r for r in group if r['coverage'] is not None]
            coverage_summary[kind] = {'attempts': len(group), 'failed': sum(r['status'] == 'failed' for r in group),
                'applicable': len(applicable),
                'mean_coverage': statistics.mean(r['coverage'] for r in applicable) if applicable else None,
                'full_coverage_rate': statistics.mean(r['full_coverage'] for r in applicable) if applicable else None,
                'mean_source_count': statistics.mean(r['source_count'] for r in group if r['source_count'] is not None)
                if any(r['source_count'] is not None for r in group) else None}
        write_json(output / f'{name}.coverage.json', {'questions': scores, 'by_type': coverage_summary})
        report['systems'][name] = {**_summary(records), 'coverage_by_type': coverage_summary}
    report['expected_attempts'] = len(questions) * len(names)
    report['saved_attempts'] = sum(s['attempts'] for s in report['systems'].values())
    report['all_attempts_saved'] = report['saved_attempts'] == report['expected_attempts']
    report['all_answers_successful'] = report['all_attempts_saved'] and all(not s['failed'] for s in report['systems'].values())
    write_json(output / 'summary.json', report)
    return report


def run(config, corpus, frozen, split='development', names=('bm25', 'vector'), retry_errors=False,
        adapter_factory=make_adapter, retriever_factory=load_retriever):
    root = Path(config['output'])
    questions_path = root / 'questions' / f'{split}.queries.jsonl'
    questions = read_jsonl(questions_path)
    if not questions or len({q['question_id'] for q in questions}) != len(questions):
        raise ValueError('Question IDs must be distinct and nonempty')
    if any(set(q) != {'question_id', 'question'} or not re.fullmatch(r'[\w-]+', q['question_id']) for q in questions):
        raise ValueError('Inference input must contain only safe question IDs and text')
    selection = read_json(root / 'questions/selection.json')
    qids = {q['question_id'] for q in questions}
    if split == 'preliminary' and (qids != set(selection['preliminary_question_ids']) or qids & set(selection['development_ids'])):
        raise ValueError('Preliminary question selection changed or overlaps development')
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Choose distinct retrievers')
    out = root / 'runs' / split
    previous_identity = (read_json(out / 'summary.json').get('generator_identity') or {}) if (out / 'summary.json').exists() else {}
    corpus_map = {c['chunk_id']: c['source_document_id'] for c in corpus.chunks}
    started = time.perf_counter()
    with output_lock(out):
        run_config = {'experiment': frozen, 'questions_sha256': file_digest(questions_path),
                      'implementation': implementation(), 'retrievers': list(names),
                      'artifacts': {name: _artifact_identity(config, name) for name in names},
                      'environment': environment(), 'graph_depth': config.get('graph_depth', 1)}
        freeze(out / 'run_config.json', run_config)
        adapter, adapter_error = None, None
        try:
            adapter = adapter_factory(config['generation_settings'])
            if adapter.settings != frozen['generation_settings']:
                raise ValueError('Generator differs from the frozen shared configuration')
        except Exception as exc:
            adapter_error = _error(exc)
        # Retrieve and encode all queries BEFORE loading the answer model.
        for name in names:
            pending = []
            for q in questions:
                path = out / 'attempts' / name / f'{q["question_id"]}.json'
                if not path.exists() or (retry_errors and read_json(path)['status'] == 'failed'):
                    pending.append(q)
            if not pending:
                continue
            retriever, init_error, query_encoding_seconds = None, None, 0.0
            uncached = [q for q in pending if not (out / 'retrieval' / name / f'{q["question_id"]}.json').exists() or retry_errors]
            try:
                if adapter_error:
                    raise RuntimeError(adapter_error)
                if uncached:
                    retriever = retriever_factory(name, config, corpus.identity)
                    if hasattr(retriever, 'prepare_queries'):
                        retriever.prepare_queries([q['question'] for q in uncached])
                        query_encoding_seconds = retriever.query_encoding_seconds
            except Exception as exc:
                init_error = _error(exc)
            try:
                for question in uncached:
                    qid = question['question_id']
                    record = {'question_id': qid, 'status': 'failed'}
                    try:
                        if init_error:
                            raise RuntimeError(init_error)
                        trace = retriever.retrieve_with_trace(question['question'], k=frozen['retrieval_k'])
                        validate_evidence(trace, corpus_map)
                        pack_started = time.perf_counter()
                        packed = pack_context(trace['evidence'], adapter.count_tokens, frozen['context_token_budget'])
                        if not packed['context']:
                            raise ValueError('No evidence fits the shared context budget')
                        encoding_share = query_encoding_seconds / len(uncached)
                        record.update(status='success', retrieval=trace, packed=packed,
                            query_embedding_amortized_seconds=encoding_share,
                            retrieval_seconds=trace['latency_seconds'] + encoding_share,
                            packing_seconds=time.perf_counter() - pack_started)
                    except Exception as exc:
                        record['error'] = _error(exc)
                    write_json(out / 'retrieval' / name / f'{qid}.json', record)
            finally:
                if retriever is not None and hasattr(retriever, 'close'):
                    retriever.close()
        for name in names:
            for question in questions:
                qid = question['question_id']
                path = out / 'attempts' / name / f'{qid}.json'
                if path.exists() and not (retry_errors and read_json(path)['status'] == 'failed'):
                    continue
                row = {'question_id': qid, 'question': question['question'], 'retriever': name,
                       'status': 'failed', 'answer': None, 'context': '', 'chunk_ids': [],
                       'source_document_ids': [], 'document_ids': [], 'final_source_document_ids': []}
                try:
                    cached = read_json(out / 'retrieval' / name / f'{qid}.json')
                    if cached['status'] != 'success':
                        raise RuntimeError(cached['error'])
                    row.update(cached['packed'])
                    row.update(retrieval=cached['retrieval'], retrieval_seconds=cached['retrieval_seconds'],
                        query_embedding_amortized_seconds=cached['query_embedding_amortized_seconds'],
                        packing_seconds=cached['packing_seconds'],
                        final_source_document_ids=row['source_document_ids'], document_ids=row['source_document_ids'])
                    if adapter_error:
                        raise RuntimeError(adapter_error)
                    generation_started = time.perf_counter()
                    generated = adapter.generate(question['question'], row['context'])
                    if not isinstance(generated.get('answer'), str) or not generated['answer'].strip():
                        raise ValueError('Shared generator returned an empty answer')
                    row.update(answer=generated['answer'], generation_metrics=generated.get('metrics', {}),
                        generation_seconds=time.perf_counter() - generation_started, status='success')
                    row['total_query_seconds'] = row['retrieval_seconds'] + row['packing_seconds'] + row['generation_seconds']
                except Exception as exc:
                    row['error'] = _error(exc)
                write_json(path, row)
                write_json(out / 'progress.json', {'split': split, 'retriever': name, 'question_id': qid,
                    'last_status': row['status'], 'elapsed_seconds': time.perf_counter() - started})
                print(f'{split} {name} {qid}: {row["status"]}', flush=True)
        report = export_results(config, split, names, out, questions)
        report['invocation_elapsed_seconds'] = time.perf_counter() - started
        report['generator_identity'] = {**previous_identity, **(getattr(adapter, 'identity', None) or {})}
        report['generator_initialization_error'] = adapter_error
        write_json(out / 'summary.json', report)
        if adapter is not None and hasattr(adapter, 'close'):
            adapter.close()
        return report
