"""Build a manifest for an explicitly chosen list of original document IDs."""
import argparse
import hashlib
import json
from pathlib import Path
from config import DATASET_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', type=Path, required=True, help='Text file with one dataset document ID per line')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ids = [x.strip() for x in args.ids.read_text(encoding='utf-8').splitlines() if x.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('The selected ID list must be nonempty and contain no duplicates')
    index = json.loads((DATASET_ROOT / 'generated_data' / 'uuid_index.json').read_text(encoding='utf-8'))
    root = (DATASET_ROOT / 'generated_data' / 'sources').resolve()
    if args.output.exists():
        raise FileExistsError(f'Manifest already exists: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as out:
        for doc_id in sorted(ids):
            relative = index[doc_id]
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Path outside source root: {relative}')
            raw = path.read_bytes()
            if json.loads(raw)['dataset_doc_uuid'] != doc_id:
                raise ValueError(f'ID mismatch: {relative}')
            out.write(json.dumps({'source_document_id': doc_id, 'path': relative,
                                  'sha256': hashlib.sha256(raw).hexdigest()}) + '\n')
    print(f'Wrote {len(ids)} document records to {args.output}')


if __name__ == '__main__':
    main()
