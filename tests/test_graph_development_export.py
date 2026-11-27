import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_graph_development import export_development
from baseline_rag.common import read_json, read_jsonl, write_json, write_jsonl


class GraphDevelopmentExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle, self.run = self.root / 'bundle', self.root / 'run'
        self.config = self.root / 'config.json'
        write_json(self.config, {'bundle': str(self.bundle), 'output': str(self.run), 'seed': 42})
        write_json(self.bundle / 'questions/splits.json', {'development_ids': ['dev'], 'folds': {'fold_a': ['eval']}})
        write_json(self.bundle / 'questions/coverage_exclusions.json', {})
        (self.bundle / 'document_ids.txt').write_text('doc\n', encoding='utf-8')
        write_jsonl(self.bundle / 'questions/development.labels.jsonl', [{'question_id': 'dev', 'question': 'Question?',
            'gold_answer': 'Reference', 'answer_facts': ['Required fact'], 'expected_doc_ids': ['doc']}])
        write_json(self.run / 'questions/metadata.json', {'dev': {'project_type': 'single-fact', 'fold': 'development'}})
        write_json(self.run / 'runs/development/run_config.json', {'fixture_identity': 'frozen'})
        self.prediction = {'question_id': 'dev', 'question': 'Question?', 'status': 'success',
                           'answer': 'Candidate', 'context': '[cid]\nEvidence', 'final_source_document_ids': ['doc']}
        for name in ['graph-local', 'graph-global']:
            write_jsonl(self.run / f'runs/development/{name}.predictions.jsonl', [{**self.prediction, 'retriever': name}])
        self.output = self.root / 'reviews'

    def tearDown(self):
        self.temp.cleanup()

    def test_exports_all_graph_answers_blinded_with_frozen_run_identity(self):
        result = export_development(self.config, self.output)
        self.assertEqual(result['attempts'], 2)
        forms = read_jsonl(self.output / 'reviewer_1.jsonl')
        self.assertEqual(len(forms), 2)
        self.assertTrue(all(r['label'] is None and 'retriever' not in r for r in forms))
        key = read_jsonl(self.output / 'private_answer_key.jsonl')
        self.assertEqual({r['retriever'] for r in key}, {'graph-local', 'graph-global'})
        self.assertTrue(all(len(r['configuration_sha256']) == 64 for r in key))
        self.assertFalse(read_json(self.output / 'export_status.json')['calibration_40_ready'])

    def test_refuses_writing_inside_frozen_bundle(self):
        with self.assertRaisesRegex(ValueError, 'frozen bundle'):
            export_development(self.config, self.bundle / 'reviews')

    def test_refuses_overwriting_existing_human_reviews(self):
        export_development(self.config, self.output)
        with self.assertRaises(FileExistsError):
            export_development(self.config, self.output)

    def test_rejects_heldout_or_incomplete_prediction_pairing(self):
        path = self.run / 'runs/development/graph-local.predictions.jsonl'
        write_jsonl(path, [{**self.prediction, 'question_id': 'eval', 'retriever': 'graph-local'}])
        with self.assertRaisesRegex(ValueError, 'original development'):
            export_development(self.config, self.output)

    def test_failed_attempts_remain_explicit_without_fake_fact_grades(self):
        path = self.run / 'runs/development/graph-local.predictions.jsonl'
        write_jsonl(path, [{**self.prediction, 'retriever': 'graph-local', 'status': 'failed',
                            'answer': None, 'error': 'Model unavailable', 'final_source_document_ids': []}])
        result = export_development(self.config, self.output)
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(len(read_jsonl(self.output / 'reviewer_1.jsonl')), 1)
        coverage = read_jsonl(self.output / 'final_context_coverage.jsonl')
        failed = next(r for r in coverage if r['status'] == 'failed')
        self.assertIsNone(failed['coverage'])


if __name__ == '__main__':
    unittest.main()
