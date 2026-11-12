import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from preliminary_grading import agreement, validate_predictions, export, score
from prepare_preliminary import ROOT, write_json, write_rows, read_json, rows


class PreliminaryContracts(unittest.TestCase):
    def setUp(self):
        self.questions = [{'question_id': 'dev'}]
        self.prediction = {'question_id': 'dev', 'retriever': 'bm25', 'run_id': 'test',
                           'configuration_sha256': 'a' * 64, 'status': 'success',
                           'answer': 'An answer', 'context': 'Evidence', 'final_source_document_ids': ['source']}

    def test_evaluation_ids_and_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_predictions([{**self.prediction, 'question_id': 'heldout'}], self.questions, {'source'})
        with self.assertRaises(ValueError):
            validate_predictions([self.prediction, self.prediction], self.questions, {'source'})

    def test_missing_context_and_unknown_sources_are_rejected(self):
        for change in ({'context': None}, {'context': ''}, {'final_source_document_ids': ['invented']}):
            with self.assertRaises(ValueError):
                validate_predictions([{**self.prediction, **change}], self.questions, {'source'})

    def test_unrun_is_not_failed_and_failure_needs_reason(self):
        result = validate_predictions([], self.questions, {'source'})
        self.assertEqual(result['attempts'], 0)
        self.assertEqual(result['unrun_by_system']['vector'], ['dev'])
        with self.assertRaises(ValueError):
            validate_predictions([{**self.prediction, 'status': 'failed'}], self.questions, {'source'})

    def test_retrieval_only_and_mixed_runs_rejected(self):
        with self.assertRaises(ValueError):
            validate_predictions([{**self.prediction, 'status': 'retrieval_only_pending_shared_generator'}], self.questions, {'source'})
        with self.assertRaises(ValueError):
            validate_predictions([self.prediction, {**self.prediction, 'question_id': 'dev2', 'run_id': 'other'}],
                                 self.questions + [{'question_id': 'dev2'}], {'source'})

    def test_agreement_excludes_unresolved_and_handles_degenerate_kappa(self):
        result = agreement({'a': 1, 'b': 0, 'c': None}, {'a': 1, 'b': 1, 'c': 0})
        self.assertEqual(result['paired_binary_facts'], 2)
        self.assertEqual(result['excluded_facts'], 1)
        self.assertEqual(result['agreement'], 0.5)
        self.assertEqual(result['kappa'], 0)
        self.assertIsNone(agreement({'a': 1}, {'a': 1})['kappa'])

    def test_export_blinding_unresolved_grades_and_adjudication(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as tmp:
            root = Path(tmp)
            (root / 'corpus').mkdir()
            (root / 'corpus/document_ids.txt').write_text('source\n', encoding='utf-8')
            write_json(root / 'exclusions.json', {})
            write_json(root / 'config.json', {'package_root': str(root), 'output_root': str(root),
                       'selection_manifest': str(root / 'unused.json'),
                       'coverage_exclusions': str(root / 'exclusions.json'), 'seed': 42})
            write_rows(root / 'development.labels.jsonl', [{'question_id': 'dev', 'question': 'Synthetic?',
                       'gold_answer': 'Synthetic reference', 'answer_facts': ['Fact one', 'Fact two'],
                       'expected_doc_ids': ['source'], 'project_type': 'single-fact'}])
            write_rows(root / 'predictions.jsonl', [self.prediction])
            # Export outside the mock package to exercise immutable-package protection.
            (root / 'package').mkdir()
            (root / 'corpus').rename(root / 'package/corpus')
            config = read_json(root / 'config.json')
            config['package_root'] = str(root / 'package')
            write_json(root / 'config.json', config)
            output = root / 'reviews'
            export(root / 'config.json', root / 'predictions.jsonl', output)
            self.assertFalse(read_json(output / 'export_status.json')['calibration_40_ready'])
            a = rows(output / 'reviewer_1.jsonl')
            self.assertNotIn('retriever', a[0])
            for row in a:
                row['label'] = 1
            write_rows(output / 'reviewer_1.jsonl', a)
            a[0]['label'] = 0
            write_rows(output / 'reviewer_2.jsonl', a)
            score(output, root / 'scores', 'Alice', 'Bob')
            self.assertIsNone(rows(root / 'scores/agreed_answer_scores.jsonl')[0]['correctness'])
            decisions = rows(root / 'scores/adjudication.jsonl')
            decisions[0]['label'] = 1
            write_rows(root / 'adjudicated.jsonl', decisions)
            score(output, root / 'resolved', 'Alice', 'Bob', adjudicated=root / 'adjudicated.jsonl')
            self.assertEqual(rows(root / 'resolved/agreed_answer_scores.jsonl')[0]['correctness'], 1)


if __name__ == '__main__':
    unittest.main()
