"""Build a portable, source-stratified benchmark subset without changing shared inputs."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inspect_dataset import document_text
from graph_rag.common import verified_source


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='\n') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')


def select(index, count, required, seed):
    if not 1 <= count <= len(index) or not set(required) <= index.keys():
        raise ValueError('Invalid count or unknown required document')
    groups = defaultdict(list)
    for doc, path in sorted(index.items()):
        groups[path.split('/')[0]].append(doc)
    base = {s: count * len(rows) // len(index) for s, rows in groups.items()}
    remainder = sorted(groups, key=lambda s: (-(count * len(groups[s]) % len(index)), s))
    for source in remainder[:count - sum(base.values())]:
        base[source] += 1
    mandatory = {s: set(rows) & set(required) for s, rows in groups.items()}
    if len(set(required)) > count:
        raise ValueError('Required documents exceed subset size')
    quotas = {s: max(base[s], len(mandatory[s])) for s in groups}
    # Minimal adjustment to proportional quotas, taking overflow from largest strata.
    while sum(quotas.values()) > count:
        eligible = [s for s in groups if quotas[s] > len(mandatory[s])]
        source = max(eligible, key=lambda s: (quotas[s], s))
        quotas[source] -= 1
    rng, selected = random.Random(seed), set(required)
    for source in sorted(groups):
        pool = sorted(set(groups[source]) - mandatory[source])
        selected.update(rng.sample(pool, quotas[source] - len(mandatory[source])))
    return selected, dict(sorted(quotas.items())), dict(sorted(base.items()))


def build(dataset, output, count=15000, seed=42):
    import prepare_chunks
    from huggingface_hub import snapshot_download
    from config import BGE_MODEL, BGE_REVISION

    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Preserve existing bundle: {output}')
    output.mkdir(parents=True)
    index_path = dataset / 'generated_data/uuid_index.json'
    index = json.loads(index_path.read_text())
    splits = json.loads((ROOT / 'manifests/questions.json').read_text())
    verified_source((dataset / 'questions.jsonl').read_bytes(), splits['source_sha256'])
    question_ids = set(splits['development_ids']) | {q for fold in splits['folds'].values() for q in fold}
    questions = [json.loads(line) for line in (dataset / 'questions.jsonl').read_text().splitlines() if line.strip()]
    questions = sorted((q for q in questions if q['question_id'] in question_ids), key=lambda q: q['question_id'])
    if len(questions) != len(question_ids):
        raise ValueError('Missing or duplicate selected question IDs')
    required = {doc for q in questions for doc in q['expected_doc_ids']}
    selected, quotas, base = select(index, count, required, seed)
    source_root = output / 'dataset/generated_data/sources'
    manifest, documents = [], []
    for doc in sorted(selected):
        relative = index[doc]
        original = (dataset / 'generated_data/sources' / relative).resolve()
        if not original.is_relative_to((dataset / 'generated_data/sources').resolve()):
            raise ValueError('Unsafe source path')
        raw = original.read_bytes()
        obj = json.loads(raw)
        if obj['dataset_doc_uuid'] != doc:
            raise ValueError(f'Wrong document ID: {relative}')
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        manifest.append({'source_document_id': doc, 'path': relative, 'sha256': digest(destination)})
        documents.append({'source_document_id': doc, 'source_type': relative.split('/')[0],
                          'path': relative, 'text': document_text(obj)})
    jsonl(output / 'corpus_manifest.jsonl', manifest)
    jsonl(output / 'documents.jsonl', documents)
    (output / 'document_ids.txt').write_text(''.join(d + '\n' for d in sorted(selected)))
    dump(output / 'dataset/generated_data/uuid_index.json', {d: index[d] for d in sorted(selected)})
    qdir = output / 'questions'
    qdir.mkdir()
    # Indexing/query consumers receive no labels in the query files.
    query = lambda q: {'question_id': q['question_id'], 'question': q['question']}
    for name, ids in [('development', splits['development_ids']), *sorted(splits['folds'].items())]:
        jsonl(qdir / f'{name}.queries.jsonl', [query(q) for q in questions if q['question_id'] in ids])
        jsonl(qdir / f'{name}.labels.jsonl', [q for q in questions if q['question_id'] in ids])
    dump(qdir / 'splits.json', splits)
    shutil.copyfile(ROOT / 'manifests/coverage_exclusions.json', qdir / 'coverage_exclusions.json')
    dump(qdir / 'coverage.json', {'questions': len(questions), 'required_documents': len(required),
         'missing_gold_documents': sorted(required - selected),
         'categories': dict(sorted(Counter(q['question_type'] for q in questions).items())),
         'limitation': 'All listed gold IDs present; this does not prove corpus-global answers remain exhaustive. qst_0413 has an upstream duplicate-ID ambiguity; see coverage_exclusions.json.'})
    print(f'Selected {len(selected)} documents; {len(questions)} questions; preparing chunks', flush=True)
    prepare_chunks.DATASET_ROOT = output / 'dataset'
    old = sys.argv
    sys.argv = ['prepare_chunks', '--manifest', str(output / 'corpus_manifest.jsonl'),
                '--output-directory', str(output / 'chunks')]
    try:
        prepare_chunks.main()
    finally:
        sys.argv = old
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'config.json']
    tokenizer = Path(snapshot_download(BGE_MODEL, revision=BGE_REVISION, local_files_only=True,
                                      allow_patterns=tokenizer_files))
    (output / 'tokenizer').mkdir()
    for name in tokenizer_files:
        shutil.copyfile(tokenizer / name, output / 'tokenizer' / name)
    shutil.copyfile(dataset / 'LICENSE', output / 'UPSTREAM_LICENSE')
    shutil.copyfile(ROOT / 'scripts/verify_corpus_bundle.py', output / 'verify.py')
    provenance = output / 'provenance'
    provenance.mkdir()
    for relative in ['scripts/package_corpus_subset.py', 'src/prepare_chunks.py', 'src/inspect_dataset.py', 'src/config.py']:
        shutil.copyfile(ROOT / relative, provenance / Path(relative).name)
    chunk_config = json.loads((output / 'chunks/chunk_config.json').read_text())
    metadata = {'document_count': count, 'chunk_count': chunk_config['chunk_count'], 'seed': seed,
                'selection': 'source-proportional largest remainder; all project question gold documents mandatory; overflow removed from largest available strata; seeded random remainder',
                'source_document_counts': quotas, 'unconstrained_source_quotas': base,
                'population_source_counts': dict(sorted(Counter(p.split('/')[0] for p in index.values()).items())),
                'population_documents': len(index), 'required_gold_documents': len(required),
                'upstream_index_sha256': digest(index_path), 'upstream_questions_sha256': digest(dataset / 'questions.jsonl'),
                'upstream_commit': subprocess.check_output(['git', '-C', str(dataset), 'rev-parse', 'HEAD'], text=True).strip(),
                'upstream_remote': subprocess.check_output(['git', '-C', str(dataset), 'remote', 'get-url', 'origin'], text=True).strip(),
                'question_selection': 'Existing project fold_a (100), fold_b (100), development (12); no scores used. Evaluation gold IDs used only to ensure source inclusion, never for retrieval/routing.',
                'status': 'standalone 15k comparison candidate; existing shared project manifests unchanged'}
    dump(output / 'bundle.json', metadata)
    (output / 'README.md').write_text('''# Portable 15,000-document corpus

Copy this entire directory or unpack its .tar.gz on any computer. No GPU,
network connection, original benchmark checkout or Python packages are needed
to verify or consume it. Python 3.10+ is sufficient for verification:

```sh
python3 verify.py .
```

For identical comparison inputs, index `chunks/chunks.jsonl` directly. Every row
has chunk_id, source_document_id, text and token count. Both systems must index
all rows. Preserve these IDs in citations. Alternatively index the full rendered
documents in `documents.jsonl`; report that chunking differs. Raw source JSON is
under `dataset/generated_data/sources/`; paths and SHA256 are in corpus_manifest.jsonl.
Tokenizer files and immutable revision/chunk settings are bundled. The 512-token,
zero-overlap chunks were made with the project's existing BGE chunker, unmodified.

Use `questions/*.queries.jsonl` for inference; these contain only question ID
and text. Labels are separate `*.labels.jsonl`. Keep development separate from
fold_a/fold_b. Fit routing/settings only with development scores. No fitted router
or generated answers are included. The original project splits are preserved.
All listed expected_doc_ids are present, but corpus-global gold answers may no
longer be exhaustive on a subset. qst_0413 retains its known duplicate-ID coverage
exception. This package does not resolve that upstream ambiguity.

bundle.json records quotas, seed, population hashes and upstream Git revision.
The sample is conditioned on including the 423 gold documents for the 212 project
questions. Confluence exceeds its unconstrained quota slightly; this is recorded.
SHA256SUMS covers every payload file (including raw sources, scripts and tokenizer).

To recreate selection/chunks from the full benchmark using this project checkout:

```sh
data/graph/.venv/bin/python scripts/package_corpus_subset.py --dataset ../EnterpriseRAG-Bench --output data/exports/corpus_15000 --documents 15000 --seed 42
```

Use the recorded upstream revision and Python 3.12 / transformers 5.22.1 with
the pinned tokenizer cached (see project requirements.txt). Source file newline
bytes must match the hashes to recreate identical byte checksums; the archive
already carries the exact source bytes. Frozen document IDs are document_ids.txt.
The provenance folder contains the build/chunk/render code used for this bundle.
Do not feed question labels into index construction. See UPSTREAM_LICENSE.
''', encoding='utf-8')
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(output).as_posix()}\n' for p in files))
    subprocess.run([sys.executable, str(output / 'verify.py'), str(output)], check=True)
    archive = output.with_suffix('.tar.gz')
    with archive.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    info = tar.gettarinfo(str(path), arcname=f'{output.name}/{path.relative_to(output).as_posix()}')
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    with path.open('rb') as stream:
                        tar.addfile(info, stream)
    archive.with_name(archive.name + '.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive), **metadata}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'EnterpriseRAG-Bench')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/exports/corpus_15000')
    parser.add_argument('--documents', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build(args.dataset, args.output, args.documents, args.seed)
