"""Compute document coverage from the source IDs in the final supplied context."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from config import MANIFEST_ROOT


def score_coverage(question_id, gold_ids, context_ids, exclusions):
    """Return coverage, full coverage and distinct context count; None means N/A."""
    if not isinstance(gold_ids, list) or not isinstance(context_ids, list):
        raise ValueError('Gold and final context source IDs must be lists')
    if any(not isinstance(x, str) or not x for x in gold_ids + context_ids):
        raise ValueError('Source IDs must be nonempty strings')
    gold, context = set(gold_ids), set(context_ids)
    reason = exclusions.get(question_id, {}).get('reason')
    if reason or not gold:
        return {'coverage': None, 'full_coverage': None, 'source_count': len(context),
                'coverage_na_reason': reason or 'No gold document IDs'}
    return {'coverage': len(gold & context) / len(gold), 'full_coverage': gold <= context,
            'source_count': len(context), 'coverage_na_reason': None}


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--contexts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    question_rows = read_jsonl(args.questions)
    questions = {q['question_id']: q for q in question_rows}
    if len(questions) != len(question_rows):
        raise ValueError('Duplicate question IDs in questions file')
    exclusions = json.loads((MANIFEST_ROOT / 'coverage_exclusions.json').read_text(encoding='utf-8'))
    results, seen = [], set()
    for row in read_jsonl(args.contexts):
        qid = row['question_id']
        if qid not in questions or qid in seen:
            raise ValueError(f'Unknown or duplicate question ID: {qid}')
        seen.add(qid)
        question = questions[qid]
        status = row['status']
        if status == 'success':
            score = score_coverage(qid, question['expected_doc_ids'], row['final_source_document_ids'], exclusions)
        elif status == 'failed':
            score = {'coverage': None, 'full_coverage': None, 'source_count': None,
                     'coverage_na_reason': 'Failed run; no valid final context'}
        else:
            raise ValueError(f'Unknown run status: {status}')
        results.append({'question_id': qid, 'project_type': question['project_type'], 'status': status, **score})
    if seen != questions.keys():
        raise ValueError(f'Context records missing for {len(questions.keys() - seen)} questions; add explicit failure records')
    by_type = defaultdict(list)
    for row in results:
        by_type[row['project_type']].append(row)
    summary = {}
    for kind, rows in sorted(by_type.items()):
        applicable = [r for r in rows if r['coverage'] is not None]
        supplied = [r for r in rows if r['source_count'] is not None]
        summary[kind] = {'attempts': len(rows), 'failures': sum(r['status'] == 'failed' for r in rows),
                         'coverage_applicable': len(applicable),
                         'mean_coverage': sum(r['coverage'] for r in applicable) / len(applicable) if applicable else None,
                         'full_coverage_rate': sum(r['full_coverage'] for r in applicable) / len(applicable) if applicable else None,
                         'mean_source_count': sum(r['source_count'] for r in supplied) / len(supplied) if supplied else None}
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'questions': results, 'by_type': summary}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(results)} scores to {args.output}')


if __name__ == '__main__':
    main()
