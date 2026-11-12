"""Persistent SQLite BM25 and exact cosine search over normalized BGE vectors."""
from contextlib import closing
import gc
from pathlib import Path
import re
import sqlite3
import time
from .common import environment, file_digest, freeze, read_json, write_json


def evidence(row, score, retriever):
    chunk_id, doc_id, text, source_path, source_sha = row
    return {'id': chunk_id, 'chunk_id': chunk_id, 'source_document_id': doc_id,
            'source_document_ids': [doc_id], 'text': text, 'score': float(score),
            'retriever': retriever, 'source_links': [{'chunk_id': chunk_id,
            'source_document_id': doc_id, 'source_path': source_path,
            'source_uri': 'enterprise-rag://' + source_path, 'source_sha256': source_sha}]}


def build_bm25(corpus, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = {'corpus': corpus.identity, 'engine': 'sqlite-fts5-bm25',
                'sqlite_version': sqlite3.sqlite_version, 'tokenizer': 'unicode61 remove_diacritics 2',
                'k1': 1.2, 'b': 0.75, 'query': 'OR over Unicode alphanumeric terms; no stemming'}
    path = output / 'index.sqlite'
    metadata_path = output / 'metadata.json'
    if metadata_path.exists():
        freeze(output / 'settings.json', settings)
        meta = read_json(metadata_path)
        if meta['index_sha256'] != file_digest(path):
            raise ValueError('BM25 index checksum mismatch')
        return meta
    # No complete index exists; rebuild the disposable partial database.
    write_json(output / 'settings.json', settings)
    started = time.perf_counter()
    temp = output / 'index.partial.sqlite'
    # SQLite transaction makes a partially built database safe to rebuild.
    with closing(sqlite3.connect(temp)) as db:
        db.execute('DROP TABLE IF EXISTS search')
        db.execute('DROP TABLE IF EXISTS chunks')
        db.execute('CREATE TABLE chunks(chunk_id TEXT UNIQUE NOT NULL, source_document_id TEXT NOT NULL, '
                   'text TEXT NOT NULL, source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL)')
        db.executemany('INSERT INTO chunks VALUES(?,?,?,?,?)', [
            (c['chunk_id'], c['source_document_id'], c['text'],
             corpus.sources[c['source_document_id']]['path'], corpus.sources[c['source_document_id']]['sha256'])
            for c in corpus.chunks])
        db.execute("CREATE VIRTUAL TABLE search USING fts5(text, content='chunks', content_rowid='rowid', "
                   "tokenize='unicode61 remove_diacritics 2')")
        db.execute("INSERT INTO search(search) VALUES('rebuild')")
        db.execute("INSERT INTO search(search) VALUES('optimize')")
        db.commit()
    temp.replace(path)
    meta = {**settings, 'indexed_chunks': len(corpus.chunks), 'elapsed_seconds': time.perf_counter() - started,
            'index_sha256': file_digest(path), 'bytes': path.stat().st_size}
    write_json(metadata_path, meta)
    return meta


class BM25Retriever:
    name = 'bm25'

    def __init__(self, directory, expected_corpus=None):
        directory = Path(directory)
        self.metadata = read_json(directory / 'metadata.json')
        if expected_corpus is not None and self.metadata['corpus'] != expected_corpus:
            raise ValueError('BM25 index belongs to a different corpus')
        self.index_path = directory / 'index.sqlite'
        if file_digest(self.index_path) != self.metadata['index_sha256']:
            raise ValueError('BM25 index checksum mismatch')
        self.db = sqlite3.connect(self.index_path.resolve().as_uri() + '?mode=ro', uri=True)
        self.identity = {'index_sha256': self.metadata['index_sha256'], 'corpus': self.metadata['corpus']}

    def retrieve_with_trace(self, query, k=10):
        if k < 1:
            raise ValueError('k must be positive')
        started = time.perf_counter()
        tokens = list(dict.fromkeys(re.findall(r'[^\W_]+', query.casefold(), re.UNICODE)))
        records = []
        if tokens:
            expression = ' OR '.join('"' + token + '"' for token in tokens)
            rows = self.db.execute('SELECT c.chunk_id,c.source_document_id,c.text,c.source_path,c.source_sha256,'
                                   '-bm25(search) AS score FROM search JOIN chunks c ON c.rowid=search.rowid '
                                   'WHERE search MATCH ? ORDER BY bm25(search),c.chunk_id LIMIT ?', (expression, k))
            records = [evidence(row[:5], row[5], self.name) for row in rows]
        return {'retriever': self.name, 'evidence': records, 'latency_seconds': time.perf_counter() - started,
                'settings': {'k': k, 'engine': 'sqlite-fts5-bm25'}, 'corpus_scope': 'frozen_15000'}

    def retrieve(self, query, k=10):
        return self.retrieve_with_trace(query, k)['evidence']

    def close(self):
        self.db.close()


class BGEEncoder:
    def __init__(self, settings):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.settings, self.torch = settings, torch
        if settings['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; install CUDA PyTorch or explicitly configure CPU')
        if not re.fullmatch('[0-9a-f]{40}', settings['revision']):
            raise ValueError('Use an immutable embedding model revision')
        self.device = settings['device']
        self.tokenizer = AutoTokenizer.from_pretrained(settings['model'], revision=settings['revision'],
                                                      cache_dir=settings['cache_dir'], trust_remote_code=False)
        dtype = getattr(torch, settings['dtype'])
        self.model = AutoModel.from_pretrained(settings['model'], revision=settings['revision'],
            cache_dir=settings['cache_dir'], trust_remote_code=False, dtype=dtype,
            attn_implementation='sdpa').to(self.device).eval()
        if self.model.config.hidden_size != settings['dimensions']:
            raise ValueError('Embedding dimensions differ from configuration')

    def encode(self, texts, query=False):
        import numpy as np
        if not texts:
            return np.empty((0, self.settings['dimensions']), dtype=np.float32)
        result = []
        for start in range(0, len(texts), self.settings['batch_size']):
            batch = texts[start:start + self.settings['batch_size']]
            if query:
                batch = [self.settings['query_instruction'] + text for text in batch]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.settings['max_length'], return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                vectors = self.model(**encoded).last_hidden_state[:, 0].float()
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
            result.append(vectors.cpu().numpy())
        return np.concatenate(result)

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def build_vector(corpus, output, settings, encoder_factory=BGEEncoder):
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frozen = {'corpus': corpus.identity, 'embedding': settings, 'storage_dtype': 'float32',
              'normalization': 'L2', 'pooling': 'CLS', 'similarity': 'exact inner product',
              'environment': environment()}
    freeze(output / 'settings.json', frozen)
    target, partial, progress_path = output / 'embeddings.npy', output / 'embeddings.partial.npy', output / 'progress.json'
    metadata_path = output / 'metadata.json'
    if metadata_path.exists():
        meta = read_json(metadata_path)
        if meta['embeddings_sha256'] != file_digest(target):
            raise ValueError('Vector index checksum mismatch')
        return meta
    shape = (len(corpus.chunks), settings['dimensions'])
    progress = read_json(progress_path) if progress_path.exists() else {'completed': 0, 'embedding_seconds': 0.0}
    # A crash between final rename and metadata write is also resumable.
    if target.exists() and progress['completed'] == shape[0]:
        vectors = np.load(target, mmap_mode='r')
    else:
        if progress['completed'] and not partial.exists():
            raise ValueError('Embedding checkpoint is missing its data file')
        vectors = np.lib.format.open_memmap(partial, mode='r+' if partial.exists() else 'w+',
                                          dtype=np.float32, shape=shape)
        if vectors.shape != shape or not 0 <= progress['completed'] <= shape[0]:
            raise ValueError('Invalid embedding checkpoint')
        started = time.perf_counter()
        encoder = encoder_factory(settings)
        try:
            for start in range(progress['completed'], shape[0], settings['batch_size']):
                end = min(start + settings['batch_size'], shape[0])
                batch = encoder.encode([c['text'] for c in corpus.chunks[start:end]])
                if batch.shape != (end - start, shape[1]) or not np.isfinite(batch).all():
                    raise ValueError('Invalid embedding batch')
                if not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=0.002):
                    raise ValueError('Embeddings must be L2 normalized')
                vectors[start:end] = batch
                vectors.flush()
                elapsed = progress['embedding_seconds'] + time.perf_counter() - started
                write_json(progress_path, {'completed': end, 'embedding_seconds': elapsed, 'total': shape[0],
                                          'chunks_per_second': (end - progress['completed']) / max(time.perf_counter() - started, 1e-9)})
                if end % (settings['batch_size'] * 20) == 0 or end == shape[0]:
                    print(f'BGE embeddings {end}/{shape[0]} ({elapsed:.1f}s active)', flush=True)
        finally:
            encoder.close()
        del vectors
        gc.collect()
        partial.replace(target)
        vectors = np.load(target, mmap_mode='r')
    if vectors.shape != shape or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.002):
        raise ValueError('Final vector index is invalid')
    meta = {**frozen, 'indexed_chunks': shape[0], 'elapsed_seconds': read_json(progress_path)['embedding_seconds'],
            'input_tokens_with_special_tokens': sum(c['token_count_with_special_tokens'] for c in corpus.chunks),
            'embeddings_sha256': file_digest(target), 'bytes': target.stat().st_size}
    write_json(metadata_path, meta)
    return meta


class VectorRetriever:
    name = 'vector'

    def __init__(self, directory, bm25_directory, expected_corpus=None, encoder_factory=BGEEncoder):
        import numpy as np
        directory = Path(directory)
        self.metadata = read_json(directory / 'metadata.json')
        if expected_corpus is not None and self.metadata['corpus'] != expected_corpus:
            raise ValueError('Vector index belongs to a different corpus')
        path = directory / 'embeddings.npy'
        if file_digest(path) != self.metadata['embeddings_sha256']:
            raise ValueError('Vector index checksum mismatch')
        self.vectors = np.load(path, mmap_mode='r')
        self.storage = BM25Retriever(bm25_directory, self.metadata['corpus'])
        count = self.storage.db.execute('SELECT count(*) FROM chunks').fetchone()[0]
        if self.vectors.shape != (count, self.metadata['embedding']['dimensions']):
            raise ValueError('Vector rows do not match chunk storage')
        self.factory, self.encoder, self.query_cache = encoder_factory, None, {}
        self.query_encoding_seconds = 0.0
        self.identity = {'embeddings_sha256': self.metadata['embeddings_sha256'],
                         'chunk_storage_sha256': self.storage.metadata['index_sha256'],
                         'corpus': self.metadata['corpus'], 'embedding': self.metadata['embedding']}

    def prepare_queries(self, queries):
        """Encode queries before loading the generator, freeing GPU memory afterwards."""
        texts = list(dict.fromkeys(queries))
        if not texts:
            return
        started = time.perf_counter()
        encoder = self.factory(self.metadata['embedding'])
        try:
            encoded = encoder.encode(texts, query=True)
            self.query_cache.update(zip(texts, encoded))
        finally:
            encoder.close()
        self.query_encoding_seconds += time.perf_counter() - started

    def retrieve_with_trace(self, query, k=10):
        import numpy as np
        if k < 1:
            raise ValueError('k must be positive')
        started = time.perf_counter()
        cached = query in self.query_cache
        if not cached:
            self.prepare_queries([query])
        vector = self.query_cache[query]
        scores = self.vectors @ vector
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite vector scores')
        # Stable row order is the frozen shared chunk order.
        indices = np.lexsort((np.arange(len(scores)), -scores))[:k]
        records = []
        for index in indices:
            row = self.storage.db.execute('SELECT chunk_id,source_document_id,text,source_path,source_sha256 '
                                          'FROM chunks WHERE rowid=?', (int(index) + 1,)).fetchone()
            records.append(evidence(row, scores[index], self.name))
        return {'retriever': self.name, 'evidence': records, 'latency_seconds': time.perf_counter() - started,
                'query_embedding_cached': cached, 'settings': {'k': k, 'similarity': 'cosine'},
                'corpus_scope': 'frozen_15000'}

    def retrieve(self, query, k=10):
        return self.retrieve_with_trace(query, k)['evidence']

    def close(self):
        self.storage.close()
        del self.vectors


def load_retriever(name, config, corpus_identity):
    root = Path(config['output']) / 'indexes'
    if name == 'bm25':
        return BM25Retriever(root / 'bm25', corpus_identity)
    if name == 'vector':
        return VectorRetriever(root / 'vector', root / 'bm25', corpus_identity)
    from graph_rag.retrieve import GraphLocalRetriever, GraphGlobalRetriever
    if name == 'graph-local':
        retriever = GraphLocalRetriever(config['graph_index'], depth=config.get('graph_depth', 1))
    elif name == 'graph-global':
        retriever = GraphGlobalRetriever(config['graph_index'], config['graph_summaries'])
    else:
        raise ValueError(f'Unknown retriever: {name}')
    metadata = retriever.index['metadata']
    if (metadata['chunks_sha256'] != corpus_identity['chunks_sha256'] or
            metadata['source_manifest_sha256'] != corpus_identity['source_manifest_sha256'] or
            metadata['indexed_chunks'] != corpus_identity['chunk_count'] or
            metadata['total_corpus_chunks'] != corpus_identity['chunk_count'] or
            metadata['corpus_scope'] == 'pilot_sample'):
        raise ValueError('Graph index must cover the exact complete frozen comparison corpus')
    retriever.identity = {'index_sha256': file_digest(config['graph_index']),
                          'summaries_sha256': getattr(retriever, 'summaries_sha256', None),
                          'depth': config.get('graph_depth', 1), 'corpus': corpus_identity}
    return retriever
