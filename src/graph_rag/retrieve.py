"""Retriever interface: retrieve(query, k) returns source-backed evidence records.

The shared generator owns packing. pack_context requires its real token counter.
"""
import json
import math
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from .common import digest, file_digest, normalize


STOPWORDS = set('a an and are as at be by can do does for from how i in is it of on or that the their this to was were what when where which who why with'.split())


def terms(text):
    return [w for w in re.findall(r'\w+', text.casefold()) if len(w) > 1 and w not in STOPWORDS]


class LexicalRanker:
    """Fixed TF-IDF cosine ranking of graph evidence, with deterministic ties."""
    def __init__(self, texts):
        self.counts = {key: Counter(terms(text)) for key, text in texts.items()}
        df = Counter(t for counts in self.counts.values() for t in counts)
        self.idf = {t: math.log((1 + len(texts)) / (1 + n)) + 1 for t, n in df.items()}
        self.vectors = {key: {t: (1 + math.log(n)) * self.idf[t] for t, n in counts.items()}
                        for key, counts in self.counts.items()}
        self.norms = {key: math.sqrt(sum(v*v for v in vector.values())) for key, vector in self.vectors.items()}

    def scores(self, query):
        q = {t: (1 + math.log(n)) * self.idf.get(t, 0) for t, n in Counter(terms(query)).items()}
        norm = math.sqrt(sum(v*v for v in q.values()))
        return {key: sum(value * q.get(t, 0) for t, value in vector.items()) / (norm * self.norms[key])
                if norm and self.norms[key] else 0 for key, vector in self.vectors.items()}


def source_link(chunk):
    return {'chunk_id': chunk['chunk_id'], 'source_document_id': chunk['source_document_id'],
            'source_path': chunk['source_path'], 'source_uri': chunk['source_uri'],
            'source_sha256': chunk['source_sha256']}


class GraphLocalRetriever:
    name = 'graph-local'

    def __init__(self, index, depth=1, max_nodes=100, max_seeds=5):
        if depth not in (1, 2) or max_nodes < 1 or max_seeds < 1:
            raise ValueError('Use depth 1 or 2 and positive graph limits')
        self.index_path = Path(index)
        self.index = json.loads(self.index_path.read_text())
        self.depth, self.max_nodes, self.max_seeds = depth, max_nodes, max_seeds
        self.adjacency = defaultdict(set)
        for edge in self.index['edges']:
            self.adjacency[edge['source']].add(edge['target'])
            self.adjacency[edge['target']].add(edge['source'])
        self.ranker = LexicalRanker({cid: c['text'] for cid, c in self.index['chunks'].items()})

    def link_entities(self, query):
        normalized = ' ' + normalize(query) + ' '
        query_terms = set(terms(query))
        candidates = []
        for nid, node in self.index['nodes'].items():
            exact = any(' ' + normalize(alias) + ' ' in normalized for alias in node['aliases'])
            # Prefer exact mentions; multiword approximate matches require strong overlap.
            score = 0
            if exact:
                score = 1 + min(len(node['normalized'].split()), 8) / 10
            else:
                nt = set(terms(node['name']))
                overlap = len(nt & query_terms) / len(nt) if nt else 0
                if len(nt) >= 2 and overlap >= 0.75:
                    score = overlap * 0.8
            if score:
                candidates.append({'node_id': nid, 'name': node['name'], 'score': score,
                                   'method': 'exact_normalized_alias' if exact else 'token_overlap'})
        return sorted(candidates, key=lambda r: (-r['score'], r['node_id']))[:self.max_seeds]

    def retrieve_with_trace(self, query, k=10):
        if k < 1:
            raise ValueError('k must be positive')
        started = time.monotonic()
        seeds = self.link_entities(query)
        depths = {s['node_id']: 0 for s in seeds[:self.max_nodes]}
        queue = deque(depths)
        while queue:
            node = queue.popleft()
            if depths[node] >= self.depth:
                continue
            for neighbour in sorted(self.adjacency[node]):
                if neighbour not in depths and len(depths) < self.max_nodes:
                    depths[neighbour] = depths[node] + 1
                    queue.append(neighbour)
        support = defaultdict(float)
        for nid, depth in depths.items():
            for mention in self.index['nodes'][nid]['mentions']:
                support[mention['chunk_id']] += 1 / (depth + 1)
        lexical = self.ranker.scores(query)
        candidates = []
        for cid, graph_score in support.items():
            chunk = self.index['chunks'][cid]
            score = lexical[cid] + 0.1 * math.log1p(graph_score)
            candidates.append({'id': cid, 'chunk_id': cid, 'source_document_id': chunk['source_document_id'],
                               'source_document_ids': [chunk['source_document_id']], 'text': chunk['text'],
                               'score': score, 'source_links': [source_link(chunk)], 'retriever': self.name})
        evidence = sorted(candidates, key=lambda c: (-c['score'], c['id']))[:k]
        return {'retriever': self.name, 'linked_entities': seeds, 'expanded_nodes': depths,
                'evidence': evidence, 'latency_seconds': time.monotonic() - started,
                'settings': {'depth': self.depth, 'max_nodes': self.max_nodes, 'max_seeds': self.max_seeds, 'k': k},
                'corpus_scope': self.index['metadata']['corpus_scope']}

    def retrieve(self, query, k=10):
        return self.retrieve_with_trace(query, k)['evidence']


class GraphGlobalRetriever:
    name = 'graph-global'

    def __init__(self, index, summaries):
        self.index_path = Path(index)
        self.index = json.loads(self.index_path.read_text())
        summaries = Path(summaries)
        status = json.loads((summaries / 'summary_status.json').read_text())
        if not status['complete'] or status['index_sha256'] != file_digest(index):
            raise ValueError('Summaries are incomplete or belong to another index')
        if status['summary_config_sha256'] != file_digest(summaries / 'summary_config.json'):
            raise ValueError('Summary settings changed')
        self.summaries = {r['community_id']: r for r in
                          (json.loads(p.read_text()) for p in sorted((summaries / 'records').glob('*.json')))}
        self.summaries_sha256 = digest(self.summaries)
        if len(self.summaries) != status['communities'] or any(s['status'] != 'ok' for s in self.summaries.values()):
            raise ValueError('Summary checkpoint set changed')
        communities = {c['id']: c for c in self.index['communities']}
        if set(self.summaries) != set(communities):
            raise ValueError('Summary communities differ from index')
        for cid, summary in self.summaries.items():
            for claim in summary['claims']:
                if not claim['quote'] or not any(
                    claim['chunk_id'] == e['chunk_id'] and claim['quote'] in e['quote']
                    for e in communities[cid]['evidence']):
                    raise ValueError('Summary references unsupported community evidence')
        self.ranker = LexicalRanker({cid: s['title'] + ' ' + ' '.join(c['text'] for c in s['claims'])
                                    for cid, s in self.summaries.items()})

    def retrieve_with_trace(self, query, k=10):
        if k < 1:
            raise ValueError('k must be positive')
        started = time.monotonic()
        scores = self.ranker.scores(query)
        ranked = sorted((cid for cid in scores if scores[cid] > 0), key=lambda cid: (-scores[cid], cid))[:k]
        evidence = []
        for cid in ranked:
            for i, claim in enumerate(self.summaries[cid]['claims']):
                chunk = self.index['chunks'][claim['chunk_id']]
                evidence.append({'id': f'{cid}::claim::{i}', 'community_id': cid,
                    'text': claim['text'] + '\nSupporting quotation: ' + claim['quote'],
                    'score': scores[cid], 'chunk_id': claim['chunk_id'],
                    'source_document_id': chunk['source_document_id'],
                    'source_document_ids': [chunk['source_document_id']], 'source_links': [source_link(chunk)],
                    'retriever': self.name})
        return {'retriever': self.name, 'ranked_communities': ranked, 'evidence': evidence,
                'latency_seconds': time.monotonic() - started, 'settings': {'k_communities': k},
                'corpus_scope': self.index['metadata']['corpus_scope']}

    def retrieve(self, query, k=10):
        return self.retrieve_with_trace(query, k)['evidence']


def pack_context(evidence, count_tokens, max_tokens):
    """Use the SAME callback and budget as shared generation, including labels.

    Omit whole evidence items if they do not fit. Never credit an unsupplied source.
    This is optional: the shared runner can use its own identical packing routine.
    """
    if max_tokens < 1:
        raise ValueError('Context budget must be positive')
    blocks, supplied = [], []
    for item in evidence:
        labels = ', '.join(link['chunk_id'] for link in item['source_links'])
        block = f'[{labels}]\n{item["text"]}'
        candidate = '\n\n'.join(blocks + [block])
        if count_tokens(candidate) <= max_tokens:
            blocks.append(block)
            supplied.append(item)
    context = '\n\n'.join(blocks)
    return {'context': context, 'context_tokens': count_tokens(context), 'evidence': supplied,
            'chunk_ids': sorted({s['chunk_id'] for e in supplied for s in e['source_links']}),
            'source_document_ids': sorted({s['source_document_id'] for e in supplied for s in e['source_links']})}
