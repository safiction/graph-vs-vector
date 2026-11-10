"""Exhaustive structural provenance audit (semantic entailment still needs review)."""
import json
from pathlib import Path
from .common import read_jsonl, file_digest, safe_source, verified_source, write_json, node_id


def audit(index_path, manifest, dataset, output, summaries=None):
    from inspect_dataset import document_text
    index = json.loads(Path(index_path).read_text())
    if file_digest(manifest) != index['metadata']['source_manifest_sha256']:
        raise ValueError('Source manifest differs from index')
    documents = {r['source_document_id']: r for r in read_jsonl(manifest)}
    errors, counts, source_texts = [], {'source_documents': 0, 'chunks': 0, 'entity_mentions': 0,
                                      'relation_supports': 0, 'summary_claims': 0}, {}
    root = Path(dataset) / 'generated_data' / 'sources'
    for cid, chunk in index['chunks'].items():
        try:
            docid = chunk['source_document_id']
            source = documents[docid]
            path = safe_source(root, source['path'])
            if chunk['source_uri'] != path.as_uri() or chunk['source_path'] != source['path'] or chunk['source_sha256'] != source['sha256']:
                raise ValueError('Source link metadata mismatch')
            if docid not in source_texts:
                raw, _ = verified_source(path.read_bytes(), source['sha256'])
                document = json.loads(raw)
                if document['dataset_doc_uuid'] != docid:
                    raise ValueError('Source document ID differs')
                source_texts[docid] = document_text(document)
                counts['source_documents'] += 1
            if chunk['text'] not in source_texts[docid] or cid != chunk['chunk_id']:
                raise ValueError('Chunk is not in its source document')
            counts['chunks'] += 1
        except (ValueError, KeyError, OSError) as exc:
            errors.append({'kind': 'chunk', 'id': cid, 'error': str(exc)})

    def check_support(support):
        chunk = index['chunks'][support['chunk_id']]
        if support['source_document_id'] != chunk['source_document_id']:
            raise ValueError('Support document ID mismatch')
        if chunk['text'][support['start']:support['end']] != support['quote'] or not support['quote']:
            raise ValueError('Support offsets do not reconstruct exact quotation')

    for nid, node in index['nodes'].items():
        for mention in node['mentions']:
            try:
                check_support(mention)
                if node_id(mention['quote']) != nid:
                    raise ValueError('Entity mention maps to a different node')
                counts['entity_mentions'] += 1
            except (ValueError, KeyError) as exc:
                errors.append({'kind': 'entity', 'id': nid, 'error': str(exc)})
    for edge in index['edges']:
        for support in edge['supports']:
            try:
                check_support(support)
                for nid in (edge['source'], edge['target']):
                    if not any(alias in support['quote'] for alias in index['nodes'][nid]['aliases']):
                        raise ValueError('Relation quotation does not mention endpoint')
                counts['relation_supports'] += 1
            except (ValueError, KeyError) as exc:
                errors.append({'kind': 'relation', 'id': edge['id'], 'error': str(exc)})
    if summaries:
        status = json.loads((Path(summaries) / 'summary_status.json').read_text())
        if not status['complete'] or status['index_sha256'] != file_digest(index_path):
            errors.append({'kind': 'summaries', 'error': 'Incomplete summaries or wrong index checksum'})
        community_map = {c['id']: c for c in index['communities']}
        records = [json.loads(p.read_text()) for p in sorted((Path(summaries) / 'records').glob('*.json'))]
        if {r['community_id'] for r in records} != set(community_map):
            errors.append({'kind': 'summaries', 'error': 'Community checkpoint set differs from index'})
        for record in records:
            for claim in record.get('claims', []):
                try:
                    if record['status'] != 'ok':
                        raise ValueError('Failed summary')
                    community = community_map[record['community_id']]
                    if not any(claim['chunk_id'] == s['chunk_id'] and claim['quote'] in s['quote']
                               for s in community['evidence']):
                        raise ValueError('Summary claim is not linked to community evidence')
                    chunk = index['chunks'][claim['chunk_id']]
                    if not claim['quote'] or claim['quote'] not in chunk['text']:
                        raise ValueError('Summary quote differs from original chunk')
                    counts['summary_claims'] += 1
                except (ValueError, KeyError) as exc:
                    errors.append({'kind': 'summary', 'id': record['community_id'], 'error': str(exc)})
    report = {'passed': not errors, 'counts': counts, 'errors': errors,
              'index_sha256': file_digest(index_path),
              'limitations': ['Structural provenance checks do not establish semantic entailment of model predicates/claims.',
                             'Named-entity identity collisions require manual review.']}
    write_json(output, report)
    return report
