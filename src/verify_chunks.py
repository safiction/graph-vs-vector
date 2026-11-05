"""Verify a chunk file against its source manifest and pinned tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
from config import DATASET_ROOT, BGE_MODEL, BGE_REVISION, CHUNK_SIZE
from inspect_dataset import document_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--chunks', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL, revision=BGE_REVISION, local_files_only=True)
    records = {r['source_document_id']: r for r in (json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line.strip())}
    root = DATASET_ROOT / 'generated_data' / 'sources'
    seen, chunk_ids = set(), set()
    current_id, pieces, count = None, [], 0

    def verify_document(doc_id, text_parts):
        if doc_id is None:
            return
        raw = (root / records[doc_id]['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != records[doc_id]['sha256']:
            raise ValueError(f'Source checksum mismatch: {doc_id}')
        if ''.join(text_parts) != document_text(json.loads(raw)):
            raise ValueError(f'Chunks do not reconstruct source: {doc_id}')

    with args.chunks.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            doc_id = row['source_document_id']
            if doc_id != current_id:
                verify_document(current_id, pieces)
                if doc_id in seen or doc_id not in records:
                    raise ValueError(f'Unknown or out-of-order document: {doc_id}')
                seen.add(doc_id)
                current_id, pieces = doc_id, []
            if row['chunk_index'] != len(pieces):
                raise ValueError(f'Incorrect chunk order: {doc_id}')
            if row['chunk_id'] in chunk_ids or not row['chunk_id'].startswith(doc_id + '::'):
                raise ValueError(f'Invalid or duplicate chunk ID: {row["chunk_id"]}')
            chunk_ids.add(row['chunk_id'])
            actual = len(tokenizer(row['text'], add_special_tokens=True, truncation=False)['input_ids'])
            if actual != row['token_count_with_special_tokens'] or actual > CHUNK_SIZE:
                raise ValueError(f'Token count invalid: {row["chunk_id"]}')
            pieces.append(row['text'])
            count += 1
    verify_document(current_id, pieces)
    if seen != records.keys():
        raise ValueError('Chunk file does not cover the manifest exactly')
    print(f'Verified {count} chunks for {len(seen)} documents: text reconstruction and token limits')


if __name__ == '__main__':
    main()
