"""Select reproducible, disjoint evaluation and development questions."""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from config import DATASET_ROOT, OUTPUT_ROOT, MANIFEST_ROOT, RANDOM_SEED, QUESTION_CATEGORIES, QUESTIONS_BASE, TOTAL_QUESTIONS, DEV_COUNTS


def main():
    if sum(QUESTIONS_BASE.values()) != TOTAL_QUESTIONS:
        raise ValueError(f'Category allocation sums to {sum(QUESTIONS_BASE.values())}, expected {TOTAL_QUESTIONS}; resolve config first')
    type_by_category = {category: kind for kind, cats in QUESTION_CATEGORIES.items() for category in cats}
    if len(type_by_category) != sum(map(len, QUESTION_CATEGORIES.values())):
        raise ValueError('Every category must belong to only one project type')
    totals = Counter()
    for category, count in QUESTIONS_BASE.items():
        totals[type_by_category[category]] += count
    if set(totals) != set(QUESTION_CATEGORIES) or any(n != 50 for n in totals.values()):
        raise ValueError(f'Expected 50 per project type: {dict(totals)}')
    if DEV_COUNTS is None:
        raise ValueError('Set DEV_COUNTS by project type after agreeing the development set')
    question_path = DATASET_ROOT / 'questions.jsonl'
    raw = question_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len({x['question_id'] for x in rows}) != len(rows):
        raise ValueError('Duplicate question IDs')
    by_category = defaultdict(list)
    for row in rows:
        by_category[row['question_type']].append(row)
    rng = random.Random(RANDOM_SEED)
    selected, remaining = [], []
    for category, count in sorted(QUESTIONS_BASE.items()):
        pool = sorted(by_category[category], key=lambda q: q['question_id'])
        if count > len(pool):
            raise ValueError(f'Not enough {category} questions')
        rng.shuffle(pool)
        selected.extend(pool[:count])
        remaining.extend(pool[count:])
    folds = {'fold_a': [], 'fold_b': []}
    for kind in QUESTION_CATEGORIES:
        # Balance project types exactly; category differences between folds are at most one.
        kind_rows = [r for r in selected if type_by_category[r['question_type']] == kind]
        groups = defaultdict(list)
        for row in kind_rows:
            groups[row['question_type']].append(row)
        for category in sorted(groups):
            group = groups[category]
            rng.shuffle(group)
            first = 'fold_a' if len(folds['fold_a']) <= len(folds['fold_b']) else 'fold_b'
            for i, row in enumerate(group):
                target = first if i % 2 == 0 else ('fold_b' if first == 'fold_a' else 'fold_a')
                folds[target].append({**row, 'project_type': kind, 'fold': target})
        assert sum(r['project_type'] == kind for r in folds['fold_a']) == 25
        assert sum(r['project_type'] == kind for r in folds['fold_b']) == 25
    development = []
    if set(DEV_COUNTS) != set(QUESTION_CATEGORIES):
        raise ValueError('DEV_COUNTS must explicitly include every project type, including zero counts')
    for kind, n in DEV_COUNTS.items():
        pool = sorted((r for r in remaining if type_by_category[r['question_type']] == kind), key=lambda q: q['question_id'])
        if not isinstance(n, int) or n < 0 or n > len(pool):
            raise ValueError(f'Invalid or unavailable development count for {kind}: {n}; available {len(pool)}')
        rng.shuffle(pool)
        development.extend({**row, 'project_type': kind, 'fold': 'development'} for row in pool[:n])
    out = OUTPUT_ROOT / 'questions'
    out.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    targets = [out / f'{name}.jsonl' for name in ('fold_a', 'fold_b', 'development')] + [MANIFEST_ROOT / 'questions.json']
    if any(path.exists() for path in targets[:3]):
        raise FileExistsError('Question output files already exist; preserve or explicitly remove them before regeneration')
    payloads = [(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in sorted(data, key=lambda x: x['question_id']))).encode('utf-8')
                for data in [folds['fold_a'], folds['fold_b'], development]]
    manifest = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'seed': RANDOM_SEED,
                'python_version': sys.version.split()[0],
                'sampling_method': 'sort IDs by category, seeded shuffle, take configured counts; balance folds within categories',
                'output_sha256': {path.name: hashlib.sha256(content).hexdigest() for path, content in zip(targets[:3], payloads)},
                'category_allocation': QUESTIONS_BASE,
                'development_counts': DEV_COUNTS, 'folds': {k: sorted(x['question_id'] for x in v) for k, v in folds.items()},
                'development_ids': sorted(x['question_id'] for x in development)}
    if targets[3].exists():
        existing = json.loads(targets[3].read_text(encoding='utf-8'))
        if existing != manifest:
            raise ValueError('Existing selection manifest differs from generated selection or environment')
    for path, content in zip(targets[:3], payloads):
        path.write_bytes(content)
    if not targets[3].exists():
        targets[3].write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote 100 questions per fold and {len(development)} development questions to {out}')


if __name__ == '__main__':
    main()
