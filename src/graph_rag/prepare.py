"""Materialize manifest-verified sources without modifying the benchmark clone."""
import json
from pathlib import Path
from .common import read_jsonl, safe_source, verified_source, write_json, file_digest


def prepare(manifest, dataset, output, chunk_config):
    from config import BGE_MODEL, BGE_REVISION
    import config
    import prepare_chunks

    output = Path(output)
    snapshot = output / 'source_snapshot'
    root = Path(dataset) / 'generated_data' / 'sources'
    changes = {}
    records = read_jsonl(manifest)
    for record in records:
        original = safe_source(root, record['path'])
        raw, conversion = verified_source(original.read_bytes(), record['sha256'])
        if json.loads(raw)['dataset_doc_uuid'] != record['source_document_id']:
            raise ValueError(f'Document ID mismatch: {original}')
        destination = safe_source(snapshot / 'generated_data' / 'sources', record['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError(f'Snapshot changed: {destination}')
        if not destination.exists():
            destination.write_bytes(raw)
        changes[record['source_document_id']] = conversion
    write_json(output / 'source_snapshot.json', {
        'source_manifest_sha256': file_digest(manifest), 'dataset_root': str(Path(dataset).resolve()),
        'newline_conversions': changes, 'model': BGE_MODEL, 'revision': BGE_REVISION})
    chunks_dir = output / 'shared_chunks'
    if not (chunks_dir / 'chunk_config.json').exists():
        import sys
        old = sys.argv
        prepare_chunks.DATASET_ROOT = snapshot
        sys.argv = ['prepare_chunks', '--manifest', str(manifest), '--output-directory', str(chunks_dir)]
        try:
            prepare_chunks.main()
        finally:
            sys.argv = old
    actual = json.loads((chunks_dir / 'chunk_config.json').read_text())
    expected = json.loads(Path(chunk_config).read_text())
    if actual != expected or file_digest(chunks_dir / 'chunks.jsonl') != expected['chunks_sha256']:
        raise ValueError('Regenerated chunks differ from the agreed chunk manifest')
    print(f'Exact shared chunk checksum verified: {actual["chunks_sha256"]}', flush=True)
