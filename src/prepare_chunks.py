"""Chunk an explicitly selected corpus with the BGE tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
from config import DATASET_ROOT, BGE_MODEL, BGE_REVISION, CHUNK_SIZE, CHUNK_OVERLAP
from inspect_dataset import document_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='JSONL: source_document_id and source-relative path')
    parser.add_argument('--output-directory', type=Path, required=True, help='Separate output directory for this corpus')
    args = parser.parse_args()
    if not BGE_MODEL or not BGE_REVISION:
        raise ValueError('Choose and configure BGE_MODEL and immutable BGE_REVISION first')
    if CHUNK_OVERLAP != 0:
        raise ValueError('Only no-overlap chunking is implemented')
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install transformers and the chosen tokenizer before chunking') from exc
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL, revision=BGE_REVISION, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('The selected tokenizer must provide fast offset mappings')
    limit = CHUNK_SIZE - tokenizer.num_special_tokens_to_add(pair=False)
    if limit <= 0 or tokenizer.model_max_length < CHUNK_SIZE:
        raise ValueError('Tokenizer/model length is incompatible with the configured chunk size')
    raw_manifest = args.manifest.read_bytes()
    records = [json.loads(line) for line in raw_manifest.decode('utf-8').splitlines() if line.strip()]
    if not records or len({r['source_document_id'] for r in records}) != len(records):
        raise ValueError('Corpus manifest is empty or contains duplicate IDs')
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    output, metadata = out / 'chunks.jsonl', out / 'chunk_config.json'
    if output.exists() or metadata.exists():
        raise FileExistsError('Chunk output exists; preserve it before starting a new run')
    settings = {'source_manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(), 'model': BGE_MODEL,
                'revision': BGE_REVISION, 'max_tokens_with_special_tokens': CHUNK_SIZE, 'overlap': CHUNK_OVERLAP,
                'text_format': 'upstream convert_to_text with title', 'algorithm': 'offset windows; shrink end until retokenized chunk fits',
                'tokenizer_library': __import__('transformers').__version__}
    version = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    root = (DATASET_ROOT / 'generated_data' / 'sources').resolve()
    count = 0
    # An interrupted or failed run leaves a visible partial file and never produces metadata.
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        for record in sorted(records, key=lambda r: r['source_document_id']):
            path = (root / record['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Path outside source root: {path}')
            source_bytes = path.read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != record['sha256']:
                raise ValueError(f'Source differs from manifest: {path}')
            document = json.loads(source_bytes)
            doc_id = record['source_document_id']
            if document['dataset_doc_uuid'] != doc_id:
                raise ValueError(f'Document ID mismatch: {path}')
            text = document_text(document)
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            offsets = encoded['offset_mapping']
            if not offsets:
                raise ValueError(f'No tokens in {doc_id}')
            start = 0
            index = 0
            while start < len(offsets):
                end = min(start + limit, len(offsets))
                char_start = 0 if start == 0 else offsets[start][0]
                while True:
                    char_end = offsets[end][0] if end < len(offsets) else len(text)
                    chunk = text[char_start:char_end]
                    actual = len(tokenizer(chunk, add_special_tokens=True, truncation=False)['input_ids'])
                    if actual <= CHUNK_SIZE:
                        break
                    end -= 1
                    if end <= start:
                        raise ValueError(f'Unable to fit a token from {doc_id} within the chunk limit')
                item = {'chunk_id': f'{doc_id}::{version}::{index:06d}', 'source_document_id': doc_id,
                        'chunk_index': index, 'text': chunk, 'token_count_with_special_tokens': actual}
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
                start = end
                index += 1
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    settings.update({'chunk_version': version, 'document_count': len(records), 'chunk_count': count,
                     'chunks_sha256': digest.hexdigest()})
    metadata.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {count} chunks from {len(records)} documents to {output}')


if __name__ == '__main__':
    main()
