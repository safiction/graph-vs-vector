"""Graph contract checks: provenance, neighbourhoods, packing, resumption and leakage."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from graph_rag.common import digest, file_digest, write_json, node_id, verified_source
from graph_rag.extract import extract, validate_extraction
from graph_rag.index import build_index
from graph_rag.retrieve import GraphLocalRetriever, GraphGlobalRetriever, pack_context
from graph_rag.audit import audit
from graph_rag.routing import train_router, route_question, SYSTEMS
from graph_rag.run import run_preliminary


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class FakeModel:
    identity = {'model': 'fixture', 'digest': 'fixed-test-weights', 'server_version': 'fixture'}

    def __init__(self):
        self.calls = 0

    def chat(self, system, text, schema, max_tokens=1100):
        self.calls += 1
        source, target = text.split(' supports ')[0].split('\n')[-1], text.split(' supports ')[1].rstrip('.')
        return json.dumps({'entities': [{'name': source, 'type': 'project'}, {'name': target, 'type': 'project'}],
                           'relations': [{'source': source, 'target': target, 'predicate': 'supports',
                                          'quote': f'{source} supports {target}.'}]}), {
                                              'done_reason': 'stop', 'prompt_eval_count': 20, 'eval_count': 10}


class GraphContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / 'dataset'
        self.manifest, self.chunks = self.root / 'manifest.jsonl', self.root / 'chunks.jsonl'
        self.extraction, self.index_path = self.root / 'extraction', self.root / 'index.json'
        manifests, chunks = [], []
        for i, (source, target) in enumerate([('Alpha', 'Beta'), ('Beta', 'Gamma'), ('Gamma', 'Delta')]):
            docid = f'dsid_{i}'
            document = {'dataset_doc_uuid': docid, 'title_field_name': 'title', 'title': f'Note {i}',
                        'content_field_names': ['body'], 'body': f'{source} supports {target}.'}
            relative = f'github/note{i}.json'
            path = self.dataset / 'generated_data' / 'sources' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document))
            manifests.append({'source_document_id': docid, 'path': relative, 'sha256': file_digest(path)})
            chunks.append({'chunk_id': f'{docid}::fixed::000000', 'chunk_index': 0,
                           'source_document_id': docid, 'text': f'Note {i}\n\n{source} supports {target}.'})
        jsonl(self.manifest, manifests)
        jsonl(self.chunks, chunks)
        self.model = FakeModel()
        extract(self.chunks, self.manifest, self.extraction, self.model)
        self.index = build_index(self.chunks, self.manifest, self.extraction, self.dataset, self.index_path)

    def tearDown(self):
        self.tmp.cleanup()

    def make_summaries(self):
        output = self.root / 'summaries'
        write_json(output / 'summary_config.json', {'fixture': True})
        for community in self.index['communities']:
            e = community['evidence'][0]
            write_json(output / 'records' / (community['id'] + '.json'), {
                'community_id': community['id'], 'status': 'ok', 'title': 'Support relationships',
                'claims': [{'text': e['quote'], 'chunk_id': e['chunk_id'], 'quote': e['quote']}]})
        write_json(output / 'summary_status.json', {'complete': True, 'index_sha256': file_digest(self.index_path),
            'summary_config_sha256': file_digest(output / 'summary_config.json'), 'communities': len(self.index['communities'])})
        return output

    def test_completed_extraction_resumes_without_model_calls(self):
        self.assertEqual(self.model.calls, 3)
        status = extract(self.chunks, self.manifest, self.extraction, self.model)
        self.assertTrue(status['complete'])
        self.assertEqual(self.model.calls, 3)

    def test_checkpoint_rejects_changed_corpus(self):
        with self.chunks.open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            extract(self.chunks, self.manifest, self.extraction, self.model)

    def test_hallucinated_names_quotes_and_missing_endpoints_are_rejected(self):
        value = {'entities': [{'name': 'Alpha', 'type': 'project'}, {'name': 'Beta', 'type': 'project'},
                              {'name': 'Imaginary', 'type': 'project'}],
                 'relations': [{'source': 'Alpha', 'target': 'Beta', 'predicate': 'supports', 'quote': 'Alpha...Beta'},
                               {'source': 'Alpha', 'target': 'Imaginary', 'predicate': 'supports', 'quote': 'Alpha supports Beta.'}]}
        result = validate_extraction(value, 'Alpha supports Beta.')
        self.assertEqual(len(result['entities']), 2)
        self.assertEqual(result['relations'], [])
        self.assertEqual(len(result['rejected']), 3)

    def test_neighbourhood_depth_and_unknown_entities(self):
        one = GraphLocalRetriever(self.index_path, depth=1).retrieve_with_trace('Tell me about ALPHA')
        two = GraphLocalRetriever(self.index_path, depth=2).retrieve_with_trace('Tell me about Alpha')
        self.assertIn(node_id('Beta'), one['expanded_nodes'])
        self.assertNotIn(node_id('Gamma'), one['expanded_nodes'])
        self.assertIn(node_id('Gamma'), two['expanded_nodes'])
        self.assertEqual(one['linked_entities'][0]['name'], 'Alpha')
        self.assertEqual(GraphLocalRetriever(self.index_path).retrieve('unmentioned zebra'), [])
        for evidence in one['evidence']:
            self.assertTrue(evidence['source_links'][0]['source_uri'].endswith('.json'))

    def test_context_budget_counts_citations_and_credits_only_supplied_sources(self):
        evidence = GraphLocalRetriever(self.index_path).retrieve('Alpha')
        first = evidence[0]
        exact = len(f'[{first["chunk_id"]}]\n{first["text"]}')
        packed = pack_context(evidence, len, exact)
        self.assertEqual(packed['context_tokens'], exact)
        self.assertEqual(packed['source_document_ids'], [first['source_document_id']])
        self.assertEqual(pack_context(evidence, len, 1)['source_document_ids'], [])

    def test_global_retrieval_returns_claim_specific_sources(self):
        output = self.make_summaries()
        evidence = GraphGlobalRetriever(self.index_path, output).retrieve('Alpha Beta Gamma support', k=10)
        self.assertTrue(evidence)
        for item in evidence:
            self.assertEqual(item['source_document_ids'], [item['source_document_id']])
            self.assertEqual(len(item['source_links']), 1)
        self.assertTrue(audit(self.index_path, self.manifest, self.dataset, self.root / 'audit.json', output)['passed'])

    def test_global_retrieval_rejects_unsupported_summary_quote(self):
        output = self.make_summaries()
        path = next((output / 'records').glob('*.json'))
        value = json.loads(path.read_text())
        value['claims'][0]['quote'] = 'invented quotation'
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            GraphGlobalRetriever(self.index_path, output)

    def test_audit_detects_wrong_document_ids_offsets_and_changed_sources(self):
        self.assertTrue(audit(self.index_path, self.manifest, self.dataset, self.root / 'audit.json')['passed'])
        nid = next(iter(self.index['nodes']))
        self.index['nodes'][nid]['mentions'][0]['source_document_id'] = 'dsid_wrong'
        write_json(self.index_path, self.index)
        self.assertFalse(audit(self.index_path, self.manifest, self.dataset, self.root / 'audit.json')['passed'])
        source = self.dataset / 'generated_data' / 'sources' / 'github/note0.json'
        source.write_text('{}')
        report = audit(self.index_path, self.manifest, self.dataset, self.root / 'audit.json')
        self.assertTrue(any('content differs' in e['error'] for e in report['errors']))

    def test_newline_recovery_requires_exact_original_checksum(self):
        original = b'{\r\n"body": "Alpha"\r\n}'
        recovered, conversion = verified_source(original.replace(b'\r\n', b'\n'), digest(original))
        self.assertEqual(recovered, original)
        self.assertEqual(conversion, 'CRLF')
        with self.assertRaises(ValueError):
            verified_source(b'{"body":"Beta"}', digest(original))

    def test_preliminary_requires_full_graph_and_uses_shared_generator_without_gold(self):
        summary_dir = self.make_summaries()
        retrievers = [GraphLocalRetriever(self.index_path), GraphGlobalRetriever(self.index_path, summary_dir)]
        questions = [{'question_id': f'q{i}', 'question': 'Alpha fails' if i == 0 else 'Alpha supports Beta',
                      'project_type': ('single-fact', 'multi-hop', 'corpus-global', 'fuzzy')[i // 10],
                      'gold_answer': 'MUST NEVER REACH THE GENERATOR'} for i in range(40)]
        qpath, selection, config = self.root / 'questions.jsonl', self.root / 'selection.json', self.root / 'comparison.json'
        jsonl(qpath, questions)
        write_json(selection, {'development_ids': ['dev'], 'folds': {'a': [q['question_id'] for q in questions]}})
        write_json(config, {'preliminary_question_ids': [q['question_id'] for q in questions],
                           'generation_settings': {'prompt': 'shared-fixture', 'tokenizer': 'fixture'},
                           'context_token_budget': 1000, 'retrieval_k': 10,
                           'shared_chunks_sha256': file_digest(self.chunks),
                           'frozen_source_manifest_sha256': file_digest(self.manifest)})

        class SharedAdapter:
            settings = {'prompt': 'shared-fixture', 'tokenizer': 'fixture'}
            count_tokens = staticmethod(len)
            calls = 0

            def generate(self, question, context):
                assert 'MUST NEVER' not in question + context
                self.calls += 1
                if 'fails' in question:
                    raise RuntimeError('fixture generator failure')
                return {'answer': 'Fixture answer', 'metrics': {'fixture': True}}

        adapter = SharedAdapter()
        output = self.root / 'run'
        report = run_preliminary(retrievers, qpath, selection, config, adapter, output)
        self.assertEqual(report['attempts']['graph-local'], {'failed': 1, 'success': 39})
        self.assertEqual(report['attempts']['graph-global'], {'failed': 1, 'success': 39})
        saved = [json.loads(s) for s in (output / 'graph-local.preliminary.jsonl').read_text().splitlines()]
        self.assertEqual(len(saved), 40)
        self.assertTrue(all(r['final_source_document_ids'] == r['source_document_ids'] for r in saved))
        run_preliminary(retrievers, qpath, selection, config, adapter, output)
        self.assertEqual(adapter.calls, 80)
        retrievers[0].index['metadata']['corpus_scope'] = 'pilot_sample'
        with self.assertRaisesRegex(ValueError, 'sampled pilot'):
            run_preliminary(retrievers, qpath, selection, config, adapter, self.root / 'bad-run')


class RoutingContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.questions, self.selection, self.scores = [self.root / n for n in ('dev.jsonl', 'selection.json', 'scores.jsonl')]
        jsonl(self.questions, [{'question_id': 'd1', 'project_type': 'single-fact', 'question': 'What is Alpha?'},
                               {'question_id': 'd2', 'project_type': 'fuzzy', 'question': 'Find similar Beta ideas'}])
        write_json(self.selection, {'development_ids': ['d1', 'd2'], 'folds': {'a': ['eval1']}})
        self.rows = [{'question_id': q, 'retriever': s, 'score': 1.0 if s == 'vector' else 0.0}
                     for q in ('d1', 'd2') for s in SYSTEMS]
        jsonl(self.scores, self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def train(self):
        return train_router(self.scores, self.questions, self.selection, self.root / 'router.json')

    def test_routes_use_development_scores_and_label_absent_global_type(self):
        router = self.train()
        self.assertEqual(route_question(router, 'What is Alpha?')['retriever'], 'vector')
        self.assertTrue(route_question(router, 'broad', known_type='corpus-global')['used_fallback'])
        self.assertIn('corpus-global', router['unlearned_types'])
        self.assertEqual(router['trained_question_ids'], ['d1', 'd2'])

    def test_evaluated_scores_cannot_tune_routing(self):
        jsonl(self.scores, self.rows + [{'question_id': 'eval1', 'retriever': 'bm25', 'score': 1}])
        with self.assertRaisesRegex(ValueError, 'prohibited'):
            self.train()

    def test_unpaired_duplicate_and_nonfinite_scores_are_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]],
                     [{**self.rows[0], 'score': float('nan')}] + self.rows[1:]):
            jsonl(self.scores, rows)
            with self.assertRaises(ValueError):
                self.train()


if __name__ == '__main__':
    unittest.main()
