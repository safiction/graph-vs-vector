"""Validate development predictions, export blinded fact reviews, score paired reviews."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from prepare_preliminary import ROOT, read_json, rows, resolve_config, write_json, write_rows
from coverage import score_coverage

SYSTEMS = {'bm25', 'vector', 'graph-local', 'graph-global'}
TYPES = ('single-fact', 'multi-hop', 'corpus-global', 'fuzzy')


def validate_predictions(predictions, questions, source_ids):
    by_id = {q['question_id']: q for q in questions}
    seen, run_ids = set(), {}
    for p in predictions:
        qid, system = p['question_id'], p['retriever']
        if qid not in by_id or system not in SYSTEMS or (qid, system) in seen:
            raise ValueError('Unknown/non-development question, retriever, or duplicate attempt')
        seen.add((qid, system))
        for key in ('run_id', 'configuration_sha256'):
            if not isinstance(p.get(key), str) or not p[key]:
                raise ValueError(f'Missing {key}')
        digest = p['configuration_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('configuration_sha256 must be a lowercase SHA256')
        identity = (p['run_id'], digest)
        if system in run_ids and run_ids[system] != identity:
            raise ValueError('Mixed runs/configurations for one retriever')
        run_ids[system] = identity
        if p['status'] == 'success':
            if not isinstance(p.get('answer'), str) or not p['answer'].strip():
                raise ValueError('Successful answer must be nonempty; abstention must be explicit')
            if not isinstance(p.get('context'), str):
                raise ValueError('Actual packed context is required')
            ids = p.get('final_source_document_ids')
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or not set(ids) <= source_ids:
                raise ValueError('Invalid/missing final source IDs')
            if not p['context'].strip() and ids:
                raise ValueError('Empty context cannot credit sources')
        elif p['status'] == 'failed':
            if not isinstance(p.get('error'), str) or not p['error'].strip():
                raise ValueError('Failure needs an error description')
        else:
            raise ValueError('Use actual success/failed attempts, not retrieval-only or pending rows')
    return {'attempts': len(predictions), 'systems': sorted(run_ids),
            'unrun_by_system': {s: sorted(set(by_id) - {qid for qid, sys in seen if sys == s}) for s in sorted(SYSTEMS)}}


def export(config_path, predictions_path, output):
    config = resolve_config(config_path)
    if output.exists():
        raise FileExistsError('Choose a new export directory; never overwrite reviews')
    package = config['package_root']
    if output == package or package in output.parents:
        raise ValueError('Cannot write into the frozen package')
    questions = rows(config['output_root'] / 'development.labels.jsonl')
    source_ids = set((package / 'corpus/document_ids.txt').read_text(encoding='utf-8').splitlines())
    predictions = rows(predictions_path)
    status = validate_predictions(predictions, questions, source_ids)
    qmap = {q['question_id']: q for q in questions}
    exclusions = read_json(config['coverage_exclusions'])
    coverage = []
    for p in predictions:
        q = qmap[p['question_id']]
        value = score_coverage(p['question_id'], q['expected_doc_ids'], p['final_source_document_ids'], exclusions) if p['status'] == 'success' else {
            'coverage': None, 'full_coverage': None, 'source_count': None, 'coverage_na_reason': 'Failed run'}
        coverage.append({'question_id': p['question_id'], 'retriever': p['retriever'],
                         'project_type': q['project_type'], 'status': p['status'], **value})
    candidates = [p for p in predictions if p['status'] == 'success']
    # Fixed seed and type/retriever round-robin; only available successful attempts.
    rng = random.Random(config['seed'])
    selected = []
    counts = {}
    for kind in TYPES:
        buckets = {s: sorted([p for p in candidates if p['retriever'] == s and
                             qmap[p['question_id']]['project_type'] == kind], key=lambda p: p['question_id']) for s in sorted(SYSTEMS)}
        for bucket in buckets.values():
            rng.shuffle(bucket)
        chosen = []
        while len(chosen) < 10 and any(buckets.values()):
            for s in sorted(buckets):
                if buckets[s] and len(chosen) < 10:
                    chosen.append(buckets[s].pop())
        selected.extend(chosen)
        counts[kind] = len(chosen)
    rng.shuffle(selected)
    forms, key = [], []
    for number, p in enumerate(selected, 1):
        aid = f'answer_{number:04d}'
        q = qmap[p['question_id']]
        if not q['answer_facts'] or any(not isinstance(f, str) for f in q['answer_facts']):
            raise ValueError('Expected nonempty string answer facts')
        key.append({'anonymous_answer_id': aid, 'question_id': p['question_id'], 'retriever': p['retriever'],
                    'project_type': q['project_type'], 'run_id': p['run_id'], 'configuration_sha256': p['configuration_sha256']})
        for i, fact in enumerate(q['answer_facts'], 1):
            forms.append({'anonymous_answer_id': aid, 'fact_id': f'fact_{i:03d}',
                          'question': q['question'], 'reference_answer': q['gold_answer'],
                          'candidate_answer': p['answer'], 'fact': fact, 'label': None, 'notes': ''})
    output.mkdir(parents=True)
    write_rows(output / 'reviewer_1.jsonl', forms)
    write_rows(output / 'reviewer_2.jsonl', forms)
    write_rows(output / 'judge.jsonl', forms)
    write_rows(output / 'private_answer_key.jsonl', key)
    write_rows(output / 'review_template.jsonl', forms)
    write_rows(output / 'final_context_coverage.jsonl', coverage)
    status.update(selected_answers_by_type=counts, fact_rows=len(forms),
                  calibration_40_ready=all(n == 10 for n in counts.values()),
                  predictions_sha256=hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                  note='Partial samples are not the planned balanced 40-answer calibration. Unrun is not failed.')
    write_json(output / 'export_status.json', status)
    print(f'Exported {len(selected)} answers / {len(forms)} facts to {output}')


def keyed_reviews(path, template):
    result = {}
    for row in rows(path):
        key = (row['anonymous_answer_id'], row['fact_id'])
        if key not in template or key in result:
            raise ValueError('Unknown/duplicate review fact')
        for field in ('question', 'reference_answer', 'candidate_answer', 'fact'):
            if row.get(field) != template[key][field]:
                raise ValueError('Review question/answer/fact text was changed')
        label = row.get('label')
        if not (label is None or label == 'needs_review' or type(label) is int and label in (0, 1)):
            raise ValueError('Labels must be 0, 1, needs_review or null')
        result[key] = label
    if result.keys() != template.keys():
        raise ValueError('Missing fact rows')
    return result


def agreement(a, b):
    pairs = [(a[k], b[k]) for k in a if type(a[k]) is int and type(b[k]) is int]
    if not pairs:
        return {'paired_binary_facts': 0, 'excluded_facts': len(a), 'agreement': None, 'kappa': None}
    n = len(pairs)
    observed = sum(x == y for x, y in pairs) / n
    pa, pb = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return {'paired_binary_facts': n, 'excluded_facts': len(a) - n,
            'agreement': observed, 'kappa': (observed - expected) / (1 - expected) if expected < 1 else None}


def score(directory, output, reviewer_1, reviewer_2, judge=None, adjudicated=None):
    if not reviewer_1.strip() or not reviewer_2.strip() or reviewer_1.strip().casefold() == reviewer_2.strip().casefold():
        raise ValueError('Record two different human reviewer names')
    if output.exists():
        raise FileExistsError('Choose a new scoring directory')
    template = {(r['anonymous_answer_id'], r['fact_id']): r for r in rows(directory / 'review_template.jsonl')}
    if not template:
        raise ValueError('No answers were exported for grading')
    a = keyed_reviews(directory / 'reviewer_1.jsonl', template)
    b = keyed_reviews(directory / 'reviewer_2.jsonl', template)
    resolved_labels = {k: a[k] if type(a[k]) is int and type(b[k]) is int and a[k] == b[k] else None for k in template}
    if adjudicated:
        needed = {k: template[k] for k, value in resolved_labels.items() if value is None}
        decisions = keyed_reviews(adjudicated, needed)
        if any(type(value) is not int for value in decisions.values()):
            raise ValueError('All adjudicated labels must be 0 or 1')
        resolved_labels.update(decisions)
    disagreements, answer_scores = [], []
    for key in template:
        if resolved_labels[key] is None:
            disagreements.append({**template[key], 'reviewer_1_label': a[key], 'reviewer_2_label': b[key], 'label': None})
    for aid in sorted({k[0] for k in template}):
        keys = [k for k in template if k[0] == aid]
        resolved = all(resolved_labels[k] is not None for k in keys)
        answer_scores.append({'anonymous_answer_id': aid, 'facts': len(keys),
                              'status': 'resolved' if resolved else 'needs_adjudication',
                              'correctness': sum(resolved_labels[k] for k in keys) / len(keys) if resolved else None})
    report = {'reviewers': [reviewer_1, reviewer_2], 'human_human': agreement(a, b),
              'unresolved_facts': len(disagreements),
              'scope': 'sampled development answers only; not a complete system comparison'}
    if judge:
        j = keyed_reviews(judge, template)
        report['judge_human_1'] = agreement(j, a)
        report['judge_human_2'] = agreement(j, b)
    identities = {r['anonymous_answer_id']: r for r in rows(directory / 'private_answer_key.jsonl')}
    report['sampled_correctness'] = []
    for system in sorted({i['retriever'] for i in identities.values()}):
        for kind in TYPES:
            subset_scores = [r for r in answer_scores if identities[r['anonymous_answer_id']]['retriever'] == system
                             and identities[r['anonymous_answer_id']]['project_type'] == kind]
            resolved = [r['correctness'] for r in subset_scores if r['correctness'] is not None]
            if subset_scores:
                report['sampled_correctness'].append({'retriever': system, 'project_type': kind,
                    'sampled_answers': len(subset_scores), 'resolved_answers': len(resolved),
                    'unresolved_answers': len(subset_scores) - len(resolved),
                    'mean_resolved_correctness': sum(resolved) / len(resolved) if resolved else None})
    report['by_type'] = {}
    for kind in TYPES:
        subset = {k for k in template if identities[k[0]]['project_type'] == kind}
        aa, bb = {k: a[k] for k in subset}, {k: b[k] for k in subset}
        report['by_type'][kind] = {'human_human': agreement(aa, bb)}
        if judge:
            jj = {k: j[k] for k in subset}
            report['by_type'][kind].update(judge_human_1=agreement(jj, aa), judge_human_2=agreement(jj, bb))
    output.mkdir(parents=True)
    write_rows(output / 'adjudication.jsonl', disagreements)
    write_rows(output / 'agreed_answer_scores.jsonl', answer_scores)
    write_rows(output / 'sampled_scores_with_systems.jsonl', [{**identities[r['anonymous_answer_id']], **r} for r in answer_scores])
    write_json(output / 'agreement.json', report)
    print(f'Scored independent reviews; {len(disagreements)} facts need completion/adjudication')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    e = commands.add_parser('export')
    e.add_argument('--config', type=Path, default=ROOT / 'configs/preliminary.json')
    e.add_argument('--predictions', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    s = commands.add_parser('score')
    s.add_argument('--reviews', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--reviewer-1', required=True)
    s.add_argument('--reviewer-2', required=True)
    s.add_argument('--judge', type=Path)
    s.add_argument('--adjudicated', type=Path)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.config.resolve(), args.predictions.resolve(), args.output.resolve())
    else:
        score(args.reviews.resolve(), args.output.resolve(), args.reviewer_1, args.reviewer_2, args.judge, args.adjudicated)
