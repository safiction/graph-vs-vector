"""Stratified pilot extrapolation; no implicit compute budget or electricity price."""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from .common import file_digest, read_jsonl, write_json


def measure_cost(extraction, sample, manifest, output, full_index=None, summaries=None, budget_hours=None):
    extraction = Path(extraction)
    config = json.loads((extraction / 'extraction_config.json').read_text())
    sample_data = json.loads(Path(sample).read_text())
    if config['sample_sha256'] != file_digest(sample) or config['source_manifest_sha256'] != file_digest(manifest):
        raise ValueError('Cost sample/manifest does not match extraction')
    sources = {r['source_document_id']: r['path'].split('/')[0] for r in read_jsonl(manifest)}
    doc_counts = Counter(sources.values())
    records = {r['chunk_id']: r for r in
               (json.loads(p.read_text()) for p in sorted((extraction / 'records').glob('*.json')))}
    random_ids = set(sample_data['random_chunk_ids'])
    if random_ids - set(records):
        raise ValueError('Random cost sample not fully attempted')
    groups = defaultdict(list)

    def metrics(record):
        attempts = record['attempts']
        return {'seconds': sum(a['attempt_wall_seconds'] for a in attempts),
                'prompt_tokens': sum(a.get('metrics', {}).get('prompt_eval_count', 0) or 0 for a in attempts),
                'completion_tokens': sum(a.get('metrics', {}).get('eval_count', 0) or 0 for a in attempts)}

    for cid in sorted(random_ids):
        groups[sources[records[cid]['source_document_id']]].append(metrics(records[cid]))
    strata = {}
    for source, population in sample_data['stratum_chunk_counts'].items():
        rows = groups[source]
        if not rows:
            raise ValueError(f'No random cost observations for {source}')
        strata[source] = {'n': len(rows), 'population_chunks': population, 'population_documents': doc_counts[source],
                          'seconds_mean': statistics.mean(r['seconds'] for r in rows),
                          'seconds_min': min(r['seconds'] for r in rows), 'seconds_max': max(r['seconds'] for r in rows),
                          'prompt_tokens_mean': statistics.mean(r['prompt_tokens'] for r in rows),
                          'completion_tokens_mean': statistics.mean(r['completion_tokens'] for r in rows)}

    def project(chunk_populations):
        seconds = sum(chunk_populations[s] * row['seconds_mean'] for s, row in strata.items())
        return {'estimated_chunks': sum(chunk_populations.values()), 'extraction_wall_hours': seconds / 3600,
                'observed_min_max_sensitivity_hours': [sum(chunk_populations[s] * r['seconds_min'] for s, r in strata.items()) / 3600,
                                                       sum(chunk_populations[s] * r['seconds_max'] for s, r in strata.items()) / 3600],
                'prompt_tokens': sum(chunk_populations[s] * r['prompt_tokens_mean'] for s, r in strata.items()),
                'completion_tokens': sum(chunk_populations[s] * r['completion_tokens_mean'] for s, r in strata.items())}

    total_metrics = [metrics(r) for r in records.values()]
    sessions = [json.loads(p.read_text()) for p in sorted((extraction / 'sessions').glob('*.json'))]
    elapsed = sum(s['elapsed_wall_seconds'] for s in sessions)
    request_time = sum(s['sum_request_seconds'] for s in sessions)
    parallelism = request_time / elapsed if elapsed and request_time else 1
    report = {'model': config['model'], 'pilot_chunks_attempted': len(records), 'random_cost_chunks': len(random_ids),
              'actual': {k: sum(r[k] for r in total_metrics) for k in ('seconds', 'prompt_tokens', 'completion_tokens')},
              'failed_chunks': sum(r['status'] != 'ok' for r in records.values()),
              'accepted_entities': sum(len(r.get('entities', [])) for r in records.values()),
              'accepted_relations': sum(len(r.get('relations', [])) for r in records.values()),
              'rejected_items': sum(len(r.get('rejected', [])) for r in records.values()),
              'strata': strata, 'pilot_corpus_projection': project(sample_data['stratum_chunk_counts']),
              'budget_hours': budget_hours, 'budget_status': 'not_provided' if budget_hours is None else 'requires_full_index',
              'limitations': [f'{min(r["n"] for r in strata.values())}–{max(r["n"] for r in strata.values())} random chunks per source; sensitivity range is NOT a confidence interval.',
                 'Full corpus chunk counts are estimated from per-source pilot chunks/documents.',
                 'Elapsed inference wall time includes client/server overhead and any configured offload; it is not isolated GPU kernel time.',
                 'API billing is zero for local inference; electricity and hardware costs were not priced.',
                 'Full extraction estimate excludes corpus preparation, graph assembly, query generation and summary inference.']}
    report['actual']['elapsed_wall_seconds'] = elapsed if sessions else report['actual']['seconds']
    report['parallel_calibration'] = {'workers': config.get('workers', 1), 'measured_effective_parallelism': parallelism,
                                    'sessions': len(sessions), 'sum_request_seconds': request_time}
    if full_index:
        doc_index = json.loads(Path(full_index).read_text())
        full_counts = Counter(path.split('/')[0] for path in doc_index.values())
        if set(full_counts) != set(strata):
            raise ValueError('Pilot lacks a full-corpus source stratum')
        chunk_populations = {s: full_counts[s] * r['population_chunks'] / r['population_documents'] for s, r in strata.items()}
        report['full_corpus_documents'] = len(doc_index)
        report['full_index_sha256'] = file_digest(full_index)
        report['full_corpus_projection'] = project(chunk_populations)
        if budget_hours is not None:
            if budget_hours <= 0:
                raise ValueError('Budget hours must be positive')
            report['budget_status'] = ('extraction_alone_exceeds_budget' if
                report['full_corpus_projection']['extraction_wall_hours'] > budget_hours else
                'extraction_estimate_within_budget_total_cost_not_established')
    if config.get('workers', 1) > 1:
        for key in ('pilot_corpus_projection', 'full_corpus_projection'):
            if key in report:
                projection = report[key]
                projection['sum_request_latency_hours'] = projection['extraction_wall_hours']
                projection['extraction_wall_hours'] /= parallelism
                projection['observed_min_max_sensitivity_hours'] = [h / parallelism for h in projection['observed_min_max_sensitivity_hours']]
        report['limitations'].append('Parallel wall-time projection is calibrated by measured overlapping request time / batch wall time; it is a small-batch scenario, not guaranteed steady-state throughput.')
        if budget_hours is not None and 'full_corpus_projection' in report:
            report['budget_status'] = ('extraction_alone_exceeds_budget' if
                report['full_corpus_projection']['extraction_wall_hours'] > budget_hours else
                'extraction_estimate_within_budget_total_cost_not_established')
    if summaries:
        summary_records = [json.loads(p.read_text()) for p in sorted((Path(summaries) / 'records').glob('*.json'))]
        measured = [metrics(r) for r in summary_records]
        report['actual_summaries'] = {k: sum(r[k] for r in measured) for k in ('seconds', 'prompt_tokens', 'completion_tokens')}
        summary_sessions = [json.loads(p.read_text()) for p in sorted((Path(summaries) / 'sessions').glob('*.json'))]
        report['actual_summaries']['elapsed_wall_seconds'] = (sum(s['elapsed_wall_seconds'] for s in summary_sessions)
                                                            if summary_sessions else report['actual_summaries']['seconds'])
        report['summary_methods'] = dict(Counter(r.get('method', 'unknown') for r in summary_records))
    write_json(output, report)
    return report
