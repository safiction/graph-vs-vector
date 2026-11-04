"""Audit question references, source files and exact rendered-text duplicates."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from config import DATASET_ROOT, OUTPUT_ROOT


def document_text(document):
    """Render the fields in the benchmark's text-export order."""
    title = str(document[document['title_field_name']])
    fields = document['content_field_names']
    if not isinstance(fields, list) or not fields:
        raise ValueError('content_field_names must be a nonempty list')
    parts = []
    for name in fields:
        value = document[name]
        parts.append('\n'.join(str(item) for item in value) if isinstance(value, list) else str(value))
    return title + '\n\n' + '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET_ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'audit')
    parser.add_argument('--progress-interval', type=float, default=5.0,
                        help='Seconds between progress updates (default: 5)')
    args = parser.parse_args()
    if args.progress_interval <= 0:
        parser.error('--progress-interval must be positive')
    root = args.dataset / 'generated_data' / 'sources'
    if not root.is_dir():
        raise FileNotFoundError(f'Source directory not found: {root}')
    print('Loading question data and document index...', flush=True)
    index = json.loads((args.dataset / 'generated_data' / 'uuid_index.json').read_text(encoding='utf-8'))
    questions = [json.loads(line) for line in (args.dataset / 'questions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    gold = {item for q in questions for item in q['expected_doc_ids']}
    counts = Counter()
    paths_seen, ids = set(), {}
    hashes = defaultdict(list)
    issues = []
    print('Finding source files (the total is not known yet)...', flush=True)
    paths = []
    last_update = time.monotonic()
    for path in root.rglob('*.json'):
        paths.append(path)
        now = time.monotonic()
        if now - last_update >= args.progress_interval:
            print(f'Finding source files: {len(paths):,} found so far', flush=True)
            last_update = now
    total = len(paths)
    print(f'Found {total:,} source JSON files. Starting inspection...', flush=True)
    started = time.monotonic()
    last_update = started
    for processed, path in enumerate(paths, 1):
        relative = path.relative_to(root).as_posix()
        paths_seen.add(relative)
        counts[relative.split('/')[0]] += 1
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
            doc_id = obj['dataset_doc_uuid']
            if not isinstance(doc_id, str) or not doc_id.startswith('dsid_'):
                raise ValueError('invalid dataset_doc_uuid')
            text = document_text(obj)
            if not text.strip():
                raise ValueError('empty rendered document')
            if doc_id in ids:
                issues.append({'issue': 'duplicate_document_id', 'path': relative, 'other': ids[doc_id], 'id': doc_id})
            else:
                ids[doc_id] = relative
            if index.get(doc_id) != relative:
                issues.append({'issue': 'index_mismatch', 'path': relative, 'id': doc_id, 'indexed_path': index.get(doc_id)})
            hashes[hashlib.sha256(text.encode('utf-8')).hexdigest()].append(doc_id)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            issues.append({'issue': 'invalid_document', 'path': relative, 'detail': str(exc)})
        now = time.monotonic()
        if now - last_update >= args.progress_interval or processed == total:
            elapsed = now - started
            rate = processed / elapsed if elapsed else 0
            remaining = (total - processed) / rate if rate else 0
            print(f'Inspected {processed:,}/{total:,} ({processed / total:.1%}) | '
                  f'{rate:,.1f} files/s | elapsed {elapsed / 60:.1f} min | '
                  f'estimated remaining {remaining / 60:.1f} min | '
                  f'issues found {len(issues):,}', flush=True)
            last_update = now
    print('Inspection complete. Checking index references and writing reports...', flush=True)
    for doc_id, path in index.items():
        if path not in paths_seen:
            issues.append({'issue': 'indexed_path_missing', 'path': path, 'id': doc_id})
    for doc_id in sorted(gold - ids.keys()):
        issues.append({'issue': 'gold_id_missing', 'id': doc_id})
    duplicates = [{'sha256': digest, 'document_ids': sorted(values)} for digest, values in hashes.items() if len(values) > 1]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'question_count': len(questions), 'question_ids_unique': len({q['question_id'] for q in questions}),
               'question_types': dict(sorted(Counter(q['question_type'] for q in questions).items())),
               'question_file_sha256': hashlib.sha256((args.dataset / 'questions.jsonl').read_bytes()).hexdigest(),
               'indexed_documents': len(index), 'source_json_files': len(paths_seen), 'source_counts': dict(sorted(counts.items())),
               'unique_gold_ids': len(gold), 'issue_counts': dict(Counter(x['issue'] for x in issues)),
               'exact_rendered_text_duplicate_groups': len(duplicates)}
    for name, value in [('summary.json', summary), ('issues.json', issues), ('duplicate_text_groups.json', duplicates)]:
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
