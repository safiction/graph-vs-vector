"""Prepare development-only handoff without modifying the frozen package."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config import QUESTION_CATEGORIES
from coverage import score_coverage


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, values):
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf-8')


def resolve_config(config_path):
    config = read_json(config_path)
    for key in ('package_root', 'output_root', 'selection_manifest', 'coverage_exclusions'):
        config[key] = (ROOT / config[key]).resolve()
    return config


def prepare(config_path):
    config = resolve_config(config_path)
    package, output = config['package_root'], config['output_root']
    if output == package or package in output.parents:
        raise ValueError('Output must be outside the frozen package')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite preparation or human reviews: {output}')
    corpus, results = package / 'corpus', package / 'results'
    selection = read_json(config['selection_manifest'])
    exclusions = read_json(config['coverage_exclusions'])
    checks = dict((rel, digest) for digest, rel in
                  (line.split('  ', 1) for line in (package / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()))
    key_files = ['package.json', 'results/index.json', 'results/source_audit.json',
                 'results/extraction_status.json', 'results/summaries/summary_status.json',
                 'corpus/bundle.json', 'corpus/document_ids.txt', 'corpus/corpus_manifest.jsonl',
                 'corpus/chunks/chunks.jsonl', 'corpus/chunks/chunk_config.json']
    for split in ('development', 'fold_a', 'fold_b'):
        key_files += [f'corpus/questions/{split}.{kind}.jsonl' for kind in ('queries', 'labels')]
    key_files += [f'results/development/{s}.retrieval.jsonl' for s in ('graph-local', 'graph-global')]
    hashes = {}
    for relative in key_files:
        actual = sha(package / relative)
        if checks.get(relative) != actual:
            raise ValueError(f'Package checksum mismatch: {relative}')
        hashes[relative] = actual
    audit = read_json(results / 'source_audit.json')
    if not audit['passed'] or audit['index_sha256'] != hashes['results/index.json']:
        raise ValueError('Saved audit does not pass or describe this index')
    extraction = read_json(results / 'extraction_status.json')
    summary = read_json(results / 'summaries/summary_status.json')
    if not extraction['complete'] or not summary['complete']:
        raise ValueError('Package reports incomplete indexing')
    docs = {x['source_document_id']: x for x in rows(corpus / 'corpus_manifest.jsonl')}
    mapping = {category: kind for kind, cats in QUESTION_CATEGORIES.items() for category in cats}
    normalized, seen = {}, set()
    for split in ('development', 'fold_a', 'fold_b'):
        questions = rows(corpus / f'questions/{split}.labels.jsonl')
        ids = [q['question_id'] for q in questions]
        expected = set(selection['development_ids'] if split == 'development' else selection['folds'][split])
        if len(ids) != len(set(ids)) or set(ids) != expected or seen.intersection(ids):
            raise ValueError(f'Split mismatch/overlap: {split}')
        seen.update(ids)
        if rows(corpus / f'questions/{split}.queries.jsonl') != [
                {'question_id': q['question_id'], 'question': q['question']} for q in questions]:
            raise ValueError('Query-only file differs from labels')
        for q in questions:
            if not set(q['expected_doc_ids']) <= docs.keys():
                raise ValueError('Missing gold source')
        normalized[split] = [{**q, 'project_type': mapping[q['question_type']], 'split': split} for q in questions]
    development = {q['question_id']: q for q in normalized['development']}
    diagnostics, review_rows, packet = [], [], ['# Development retrieval review', '',
        'These are retrieved candidates, NOT packed contexts or generated answers.',
        'Do not use this review as answer correctness or final-context coverage.', '']
    for system in ('graph-local', 'graph-global'):
        traces = rows(results / f'development/{system}.retrieval.jsonl')
        if len(traces) != len(development) or {t['question_id'] for t in traces} != development.keys():
            raise ValueError('Development trace IDs mismatch')
        for trace in traces:
            qid = trace['question_id']
            q = development[qid]
            sources = sorted({link['source_document_id'] for e in trace['evidence'] for link in e['source_links']})
            if not set(sources) <= docs.keys():
                raise ValueError('Trace has unknown source ID')
            score = score_coverage(qid, q['expected_doc_ids'], sources, exclusions)
            diagnostics.append({'question_id': qid, 'retriever': system, 'project_type': q['project_type'],
                                'metric_stage': 'retrieved_candidates_before_packing',
                                'evidence_items': len(trace['evidence']), **score})
            review_rows.append({'question_id': qid, 'retriever': system, 'reviewer': '',
                                'relevance': None, 'source_support': None,
                                'reference_validity': None, 'notes': ''})
            packet += [f'## {system} / {qid}', '', q['question'], '',
                       f"Type: {q['project_type']}", '', 'Reference answer:', q['gold_answer'], '',
                       'Gold source files:']
            for sid in q['expected_doc_ids']:
                packet.append(f"- {sid}: {corpus / 'dataset/generated_data/sources' / docs[sid]['path']}")
            for number, e in enumerate(trace['evidence'], 1):
                packet += ['', f'### Evidence {number}', '', e['text'], '']
                for link in e['source_links']:
                    packet.append(f"Source {link['source_document_id']}: "
                                  f"{corpus / 'dataset/generated_data/sources' / link['source_path']}")
            packet.append('')
    report = {'status': 'development_preparation_and_retrieval_diagnostics_only',
              'package_config': str(Path(config_path).relative_to(ROOT)),
              'package_commit': read_json(package / 'package.json')['commit'],
              'verified_key_file_hashes': hashes,
              'verification_scope': 'key files only; saved structural audit not rerun; no semantic certification',
              'source_document_count': len(docs), 'extraction': extraction, 'summaries': summary,
              'question_counts': {s: dict(Counter(q['project_type'] for q in qs)) for s, qs in normalized.items()},
              'split_changed': False, 'generated_answers': 0, 'graded_answers': 0,
              'blockers': ['Shared generator and baseline predictions pending',
                           'Development has no corpus-global questions; protocol decision required',
                           'No T4 validation in this delivery'],
              'retrieval_diagnostics': {system: {
                  'questions': len([r for r in diagnostics if r['retriever'] == system]),
                  'mean_candidate_document_coverage': sum(r['coverage'] for r in diagnostics if r['retriever'] == system and r['coverage'] is not None) /
                      sum(r['coverage'] is not None for r in diagnostics if r['retriever'] == system)
              } for system in ('graph-local', 'graph-global')}}
    output.mkdir(parents=True)
    for split, questions in normalized.items():
        write_rows(output / f'{split}.labels.jsonl', questions)
        write_rows(output / f'{split}.queries.jsonl', [{'question_id': q['question_id'], 'question': q['question']} for q in questions])
    write_rows(output / 'retrieval_diagnostics.jsonl', diagnostics)
    write_rows(output / 'retrieval_review.jsonl', review_rows)
    (output / 'retrieval_review_packet.md').write_text('\n'.join(packet), encoding='utf-8')
    write_json(output / 'preparation_report.json', report)
    print(f'Prepared {output}; original 12/100/100 splits preserved')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/preliminary.json')
    prepare(parser.parse_args().config.resolve())
