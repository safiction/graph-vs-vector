"""Select a source-proportional 5,000-document pilot containing dev gold documents."""
from collections import Counter, defaultdict
import json
import random
from config import DATASET_ROOT, OUTPUT_ROOT, MANIFEST_ROOT, RANDOM_SEED


def main():
    index = json.loads((DATASET_ROOT / 'generated_data' / 'uuid_index.json').read_text(encoding='utf-8'))
    dev_path = OUTPUT_ROOT / 'questions' / 'development.jsonl'
    dev = [json.loads(line) for line in dev_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    required = {doc_id for row in dev for doc_id in row['expected_doc_ids']}
    if required - index.keys():
        raise ValueError('Development gold IDs are missing from the dataset index')
    groups = defaultdict(list)
    for doc_id, path in index.items():
        groups[path.split('/')[0]].append(doc_id)
    size = 5000
    total = len(index)
    quotas = {source: size * len(ids) // total for source, ids in groups.items()}
    # Largest remainders allocate every slot; source name resolves equal remainders.
    ranked = sorted(groups, key=lambda source: (-(size * len(groups[source]) % total), source))
    for source in ranked[:size - sum(quotas.values())]:
        quotas[source] += 1
    required_by_source = Counter(index[x].split('/')[0] for x in required)
    rng = random.Random(RANDOM_SEED)
    chosen = set(required)
    for source, ids in sorted(groups.items()):
        available = sorted(set(ids) - required)
        n = quotas[source] - required_by_source[source]
        if n < 0 or n > len(available):
            raise ValueError(f'Development gold IDs exceed pilot quota for {source}')
        chosen.update(rng.sample(available, n))
    if len(chosen) != size:
        raise ValueError('Pilot selection has incorrect size')
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_ROOT / 'pilot_document_ids.txt'
    metadata = MANIFEST_ROOT / 'pilot_selection.json'
    if out.exists() or metadata.exists():
        raise FileExistsError('Pilot selection already exists')
    out.write_text(''.join(x + '\n' for x in sorted(chosen)), encoding='utf-8')
    metadata.write_text(json.dumps({'seed': RANDOM_SEED, 'sample_size': size, 'method': 'proportional source quotas, largest remainder; include development gold IDs then random sample',
                                    'source_quotas': quotas, 'development_gold_count': len(required)}, indent=2) + '\n', encoding='utf-8')
    print(f'Selected {len(chosen)} pilot documents, including {len(required)} development gold documents')


if __name__ == '__main__':
    main()
