"""Offline integrity and referential coverage check for a portable corpus bundle."""
import hashlib
import json
from pathlib import Path
import sys


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def verify(root):
    root = Path(root).resolve()
    checked = set()
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or relative in checked:
            raise ValueError(f'Unsafe or duplicate checksum path: {relative}')
        checked.add(relative)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {relative}')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual - {'SHA256SUMS'} != checked:
        raise ValueError('Unlisted or missing bundle files')
    metadata = json.loads((root / 'bundle.json').read_text())
    manifest = list(rows(root / 'corpus_manifest.jsonl'))
    docs = {r['source_document_id'] for r in manifest}
    if len(docs) != len(manifest) or len(docs) != metadata['document_count']:
        raise ValueError('Wrong document count or duplicate IDs')
    if set((root / 'document_ids.txt').read_text().splitlines()) != docs:
        raise ValueError('Document list mismatch')
    rendered = list(rows(root / 'documents.jsonl'))
    if len(rendered) != len(docs) or {r['source_document_id'] for r in rendered} != docs:
        raise ValueError('Rendered documents mismatch')
    for record in manifest:
        path = root / 'dataset/generated_data/sources' / record['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256'] or json.loads(raw)['dataset_doc_uuid'] != record['source_document_id']:
            raise ValueError(f'Source manifest mismatch: {path}')
    chunks, chunk_docs = set(), set()
    for chunk in rows(root / 'chunks/chunks.jsonl'):
        if chunk['source_document_id'] not in docs or chunk['chunk_id'] in chunks or not chunk['text'].strip():
            raise ValueError('Invalid or duplicate chunk')
        chunks.add(chunk['chunk_id'])
        chunk_docs.add(chunk['source_document_id'])
    if chunk_docs != docs or len(chunks) != metadata['chunk_count']:
        raise ValueError('Incomplete chunk coverage')
    seen = set()
    for name in ['development', 'fold_a', 'fold_b']:
        queries = list(rows(root / 'questions' / f'{name}.queries.jsonl'))
        labels = list(rows(root / 'questions' / f'{name}.labels.jsonl'))
        if queries != [{'question_id': q['question_id'], 'question': q['question']} for q in labels]:
            raise ValueError('Queries differ from labels or expose extra fields')
        for q in labels:
            if q['question_id'] in seen or not set(q['expected_doc_ids']) <= docs:
                raise ValueError('Question overlap or missing gold document')
            seen.add(q['question_id'])
    print(f'PASS: {len(checked)} hashes, {len(docs)} documents, {len(chunks)} chunks, {len(seen)} disjoint questions')


if __name__ == '__main__':
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
