"""Validate shared chunks and freeze question selections without leaking labels."""
from collections import Counter
from pathlib import Path
from config import QUESTION_CATEGORIES
from .common import digest, file_digest, freeze, read_json, read_jsonl, write_json, write_jsonl

TYPES = ('single-fact', 'multi-hop', 'corpus-global', 'fuzzy')
TYPE_MAP = {category: kind for kind, categories in QUESTION_CATEGORIES.items() for category in categories}


class Corpus:
    def __init__(self, bundle):
        self.root = Path(bundle)
        self.chunk_path = self.root / 'chunks/chunks.jsonl'
        self.manifest_path = self.root / 'corpus_manifest.jsonl'
        self.config = read_json(self.root / 'chunks/chunk_config.json')
        self.bundle = read_json(self.root / 'bundle.json')
        for path, expected in [(self.chunk_path, self.config['chunks_sha256']),
                               (self.manifest_path, self.config['source_manifest_sha256'])]:
            if file_digest(path) != expected:
                raise ValueError(f'Corpus checksum mismatch: {path}')
        records = read_jsonl(self.manifest_path)
        self.sources = {row['source_document_id']: row for row in records}
        if len(self.sources) != len(records) or len(records) != self.bundle['document_count']:
            raise ValueError('Duplicate document IDs or wrong manifest size')
        self.chunks = read_jsonl(self.chunk_path)
        chunk_ids = [c['chunk_id'] for c in self.chunks]
        if (len(set(chunk_ids)) != len(chunk_ids) or len(chunk_ids) != self.bundle['chunk_count'] or
                {c['source_document_id'] for c in self.chunks} != set(self.sources)):
            raise ValueError('Invalid chunk count, duplicate IDs or incomplete document coverage')
        if any(not c['text'].strip() or not 0 < c['token_count_with_special_tokens'] <= 512 for c in self.chunks):
            raise ValueError('Empty or oversized shared chunk')
        self.identity = {'chunks_sha256': self.config['chunks_sha256'],
                         'source_manifest_sha256': self.config['source_manifest_sha256'],
                         'document_count': len(self.sources), 'chunk_count': len(self.chunks),
                         'chunk_ids_sha256': digest(chunk_ids)}

    def audit(self):
        missing = [s['path'] for s in self.sources.values() if not
                   (self.root / 'dataset/generated_data/sources' / s['path']).is_file()]
        return {**self.identity, 'chunks_verified': True, 'missing_raw_source_files': missing,
                'raw_bundle_complete': not missing,
                'index_input': 'Verified prebuilt shared chunks; no re-chunking or source exclusion'}


def prepare_questions(bundle, output, seed=42):
    """Select 5 questions per type per evaluation fold by seeded hash ordering.

    Only category/ID metadata affects selection. Gold answers and facts never enter
    query files or retrievers. The existing 200 evaluation and 12 dev IDs are kept.
    """
    bundle, output = Path(bundle), Path(output)
    split_manifest = read_json(bundle / 'questions/splits.json')
    query_sets, metadata = {}, {}
    seen = set()
    input_hashes = {}
    for split in ['development', 'fold_a', 'fold_b']:
        qpath = bundle / 'questions' / f'{split}.queries.jsonl'
        lpath = bundle / 'questions' / f'{split}.labels.jsonl'
        queries, labels = read_jsonl(qpath), read_jsonl(lpath)
        if queries != [{'question_id': q['question_id'], 'question': q['question']} for q in labels]:
            raise ValueError('Query/label mismatch or extra fields in query input')
        ids = [q['question_id'] for q in queries]
        if len(ids) != (12 if split == 'development' else 100):
            raise ValueError('Expected 12 development and 100 questions per evaluation fold')
        expected = split_manifest['development_ids'] if split == 'development' else split_manifest['folds'][split]
        if len(set(ids)) != len(ids) or set(ids) & seen or set(ids) != set(expected):
            raise ValueError('Question split differs from frozen selection or overlaps another split')
        seen.update(ids)
        input_hashes[qpath.name] = file_digest(qpath)
        input_hashes[lpath.name] = file_digest(lpath)
        query_sets[split] = queries
        for label in labels:
            qid = label['question_id']
            metadata[qid] = {'project_type': TYPE_MAP[label['question_type']], 'fold': split}
        if split != 'development' and Counter(metadata[qid]['project_type'] for qid in ids) != Counter(dict.fromkeys(TYPES, 25)):
            raise ValueError('Evaluation fold must have 25 questions per project type')
    preliminary = []
    for split in ['fold_a', 'fold_b']:
        for kind in TYPES:
            candidates = [q for q in query_sets[split] if metadata[q['question_id']]['project_type'] == kind]
            candidates.sort(key=lambda q: digest([seed, split, kind, q['question_id']]))
            preliminary.extend(candidates[:5])
    preliminary.sort(key=lambda q: q['question_id'])
    selection = {'seed': seed, 'method': 'seeded SHA256 order; 5 per type per evaluation fold',
                 'preliminary_question_ids': [q['question_id'] for q in preliminary],
                 'development_ids': [q['question_id'] for q in query_sets['development']],
                 'input_sha256': input_hashes, 'selection_sha256': file_digest(bundle / 'questions/splits.json')}
    freeze(output / 'selection.json', selection)
    for name, rows in {**query_sets, 'preliminary': preliminary}.items():
        path = output / f'{name}.queries.jsonl'
        if path.exists() and read_jsonl(path) != rows:
            raise ValueError(f'Question input changed: {path}')
        write_jsonl(path, rows)
    freeze(output / 'metadata.json', metadata)
    return selection


def prepare(config):
    corpus = Corpus(config['bundle'])
    output = Path(config['output'])
    selection = prepare_questions(corpus.root, output / 'questions', config['seed'])
    frozen = {'corpus': corpus.identity, 'preliminary_question_ids': selection['preliminary_question_ids'],
              'development_ids': selection['development_ids'], 'retrieval_k': config['retrieval_k'],
              'context_token_budget': config['context_token_budget'],
              'embedding': config['embedding'], 'generation_settings': config['generation_settings']}
    if frozen['retrieval_k'] < 1 or frozen['context_token_budget'] < 1:
        raise ValueError('Positive retrieval k and context token budget required')
    freeze(output / 'experiment.json', frozen)
    audit = corpus.audit()
    write_json(output / 'corpus_audit.json', audit)
    return corpus, frozen, audit
