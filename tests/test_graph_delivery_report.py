"""Delivery report public seam: build_report and CLI, using only temporary artifacts.

Catches saved retry accounting, integrity, budgets/provenance and pending runs.
Does not establish semantic correctness, tokenizer accuracy or real hardware costs.
"""
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_graph_delivery import build_report


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class DeliveryReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle = self.root / 'bundle'
        self.graph = self.root / 'graph'
        self.output = self.root / 'answers'
        self.config_path = self.root / 'config.json'
        self.chunks = [{'chunk_id': f'c{i}', 'source_document_id': f'd{i}',
                        'text': text, 'token_count_with_special_tokens': 8}
                       for i, text in enumerate(['Alpha supports Beta.', 'Gamma supports Delta.'])]
        self.sources = [{'source_document_id': f'd{i}', 'path': f'notes/{i}.json',
                         'sha256': str(i) * 64} for i in range(2)]
        save_lines(self.bundle / 'chunks/chunks.jsonl', self.chunks)
        save_lines(self.bundle / 'corpus_manifest.jsonl', self.sources)
        self.corpus = {'document_count': 2, 'chunk_count': 2,
                       'chunks_sha256': file_hash(self.bundle / 'chunks/chunks.jsonl'),
                       'source_manifest_sha256': file_hash(self.bundle / 'corpus_manifest.jsonl'),
                       'chunk_ids_sha256': digest(['c0', 'c1'])}
        save(self.bundle / 'bundle.json', {'document_count': 2, 'chunk_count': 2,
                                         'selection': 'question-conditioned fixture'})
        save(self.bundle / 'chunks/chunk_config.json', self.corpus)
        self.config = {'bundle': str(self.bundle), 'output': str(self.output), 'seed': 42,
                       'retrieval_k': 10, 'context_token_budget': 2560, 'embedding': {'model': 'fixture'},
                       'generation_settings': {'model': 'answer-fixture', 'max_input_tokens': 4096,
                                               'max_new_tokens': 384},
                       'graph_index': str(self.graph / 'index.json'),
                       'graph_summaries': str(self.graph / 'summaries'), 'graph_depth': 1}
        save(self.config_path, self.config)
        self.questions = {'development': [f'dev{i}' for i in range(12)],
                          'fold_a': [f'a{i}' for i in range(100)],
                          'fold_b': [f'b{i}' for i in range(100)]}
        self.questions['preliminary'] = self.questions['fold_a'][:20] + self.questions['fold_b'][:20]
        for split, ids in self.questions.items():
            save_lines(self.output / 'questions' / f'{split}.queries.jsonl',
                       [{'question_id': qid, 'question': 'What supports Beta?'} for qid in ids])
        self.frozen = {key: self.config[key] for key in
                       ('retrieval_k', 'context_token_budget', 'embedding', 'generation_settings')}
        self.frozen.update(corpus=self.corpus, development_ids=self.questions['development'],
                           preliminary_question_ids=self.questions['preliminary'])
        save(self.output / 'experiment.json', self.frozen)
        save(self.output / 'corpus_audit.json', {**self.corpus, 'chunks_verified': True,
                                               'raw_bundle_complete': True, 'missing_raw_source_files': []})
        save(self.output / 'questions/selection.json',
             {'development_ids': self.questions['development'],
              'preliminary_question_ids': self.questions['preliminary'],
               'input_sha256': {f'{split}.queries.jsonl': file_hash(self.output / 'questions' / f'{split}.queries.jsonl')
                                for split in ('development', 'fold_a', 'fold_b')}})
        self.extraction_config = {key: self.corpus[key] for key in
                                  ('chunks_sha256', 'source_manifest_sha256')}
        self.extraction_config.update(sample_sha256=None, model={'model': 'extraction-fixture'})
        save(self.graph / 'extraction_config.json', self.extraction_config)
        self.records = []
        for i, chunk in enumerate(self.chunks):
            attempts = [self.attempt(10, 1, 7)] if i == 0 else [self.attempt(40, 4, 10)]
            if i == 0:
                for prompt, completion, wall in [(20, 2, 8), (30, 3, 9)]:
                    attempts.append({**self.attempt(prompt, completion, wall),
                                     'recovery_manifest': 'recovery/fix/manifest.json',
                                     'recovery_profile': {'max_tokens': 2048}})
            row = {**{key: chunk[key] for key in ('chunk_id', 'source_document_id')},
                   'text_sha256': digest(chunk['text']), 'status': 'ok', 'attempts': attempts,
                   'entities': [], 'relations': [], 'rejected': []}
            self.records.append(row)
            save(self.graph / 'records' / f'{digest(chunk["chunk_id"])}.json', row)
        save(self.graph / 'recovery/fix/original_records/backup.json',
             {**self.records[0], 'status': 'error', 'attempts': self.records[0]['attempts'][:1]})
        save(self.graph / 'recovery/fix/manifest.json',
             {'primary_extraction_config_sha256': file_hash(self.graph / 'extraction_config.json'),
              'model': {'model': 'extraction-fixture'}, 'profile': {'max_tokens': 2048}})
        save(self.graph / 'extraction_status.json',
             {'complete': True, 'selected_chunks': 2, 'completed_chunks': 2,
              'error_chunks': 0, 'pending_chunks': 0,
              'extraction_config_sha256': file_hash(self.graph / 'extraction_config.json')})
        save(self.graph / 'sessions/one.json', {'attempted_chunks': 1, 'workers': 2,
             'elapsed_wall_seconds': 3, 'sum_request_seconds': 7})
        index_chunks = {c['chunk_id']: {**c, 'source_path': s['path'], 'source_sha256': s['sha256'],
                        'source_uri': (self.bundle / 'dataset/generated_data/sources' / s['path']).as_uri()}
                        for c, s in zip(self.chunks, self.sources)}
        self.evidence = {'chunk_id': 'c0', 'source_document_id': 'd0', 'start': 0,
                         'end': 20, 'quote': 'Alpha supports Beta.'}
        self.index = {'metadata': {**self.corpus, 'total_corpus_chunks': 2, 'indexed_chunks': 2,
                                  'corpus_scope': 'all_supplied_chunks',
                                  'extraction_config_sha256': file_hash(self.graph / 'extraction_config.json'),
                                  'checkpoint_sha256': self.checkpoint_hash()},
                      'nodes': {'n0': {'name': 'Alpha'}, 'n1': {'name': 'Beta'}}, 'edges': [],
                      'chunks': index_chunks,
                      'communities': [{'id': 'com_one', 'node_ids': ['n0', 'n1'], 'evidence': [self.evidence]}]}
        save(self.graph / 'index.json', self.index)
        save(self.graph / 'index.cost.json', {'wall_seconds': 1.5, 'model_calls': 0,
                                            'nodes': 2, 'relations': 0, 'communities': 1})
        save(self.graph / 'summaries/summary_config.json',
             {'index_sha256': file_hash(self.graph / 'index.json'), 'model': {'model': 'summary-fixture'},
              'max_evidence_items': 12})
        self.summary = {'community_id': 'com_one', 'status': 'ok', 'method': 'extractive_fallback',
                        'supplied_evidence': [self.evidence], 'title': 'Support',
                        'claims': [{'text': 'Alpha supports Beta.', 'quote': 'Alpha supports Beta.', 'chunk_id': 'c0'}],
                        'attempts': [self.attempt(5, 1, 2), self.attempt(6, 2, 3)]}
        save(self.graph / 'summaries/records/com_one.json', self.summary)
        save(self.graph / 'summaries/summary_status.json',
             {'complete': True, 'index_sha256': file_hash(self.graph / 'index.json'),
              'summary_config_sha256': file_hash(self.graph / 'summaries/summary_config.json'),
              'communities': 1, 'completed': 1, 'errors': 0, 'extractive_fallbacks': 1})
        save(self.graph / 'summaries/sessions/one.json', {'attempted_communities': 1,
             'elapsed_wall_seconds': 4, 'sum_request_seconds': 2, 'workers': 2})
        save(self.graph / 'source_audit.json', {'passed': True, 'errors': [],
             'index_sha256': file_hash(self.graph / 'index.json'),
             'counts': {'source_documents': 2, 'chunks': 2, 'summary_claims': 1}})

    @staticmethod
    def attempt(prompt, completion, wall):
        return {'attempt_wall_seconds': wall,
                'metrics': {'prompt_eval_count': prompt, 'eval_count': completion,
                            'wall_seconds': wall - 1, 'done_reason': 'stop'}}

    def checkpoint_hash(self):
        return digest(sorted(self.records, key=lambda row: digest(row['chunk_id'])))

    def report(self):
        return build_report(self.config_path, self.graph)

    def make_run(self, split='development'):
        root = self.output / 'runs' / split
        identities = {'graph-local': {str(self.graph / 'index.json'): file_hash(self.graph / 'index.json')},
                      'graph-global': {str(self.graph / name): file_hash(self.graph / name) for name in
                                       ('index.json', 'summaries/summary_config.json', 'summaries/summary_status.json')}}
        save(root / 'run_config.json', {'experiment': self.frozen,
             'questions_sha256': file_hash(self.output / 'questions' / f'{split}.queries.jsonl'),
             'retrievers': ['graph-local', 'graph-global'], 'artifacts': identities, 'graph_depth': 1,
             'implementation': {'commit': 'fixture'}, 'environment': {'python': 'fixture'}})
        summary = {'split': split, 'systems': {}, 'expected_attempts': len(self.questions[split]) * 2,
                   'saved_attempts': len(self.questions[split]) * 2, 'all_attempts_saved': True,
                   'all_answers_successful': False, 'invocation_elapsed_seconds': 99,
                   'generator_identity': {'model': 'answer-fixture',
                                          'settings_sha256': digest(self.config['generation_settings'])}}
        for system in ('graph-local', 'graph-global'):
            rows = []
            for i, qid in enumerate(self.questions[split]):
                chunk = self.index['chunks']['c0']
                link = {key: chunk[key] for key in
                        ('chunk_id', 'source_document_id', 'source_path', 'source_uri', 'source_sha256')}
                row = {'question_id': qid, 'question': 'What supports Beta?', 'retriever': system,
                       'status': 'success', 'answer': 'Alpha [c0]', 'context': '[c0]\nAlpha supports Beta.',
                       'context_tokens': 8, 'chunk_ids': ['c0'], 'source_document_ids': ['d0'],
                       'final_source_document_ids': ['d0'],
                       'evidence': [{'text': 'Alpha supports Beta.', 'source_links': [link]}],
                       'generation_metrics': {'prompt_tokens': 100, 'local_prompt_tokens': 100,
                                              'completion_tokens': 10},
                       'retrieval_seconds': 1, 'packing_seconds': 2, 'generation_seconds': 3,
                       'total_query_seconds': 6}
                if system == 'graph-global' and i == 0:
                    row = {'question_id': qid, 'question': row['question'], 'retriever': system,
                           'status': 'failed', 'error': 'Fixture failure', 'answer': None,
                           'context': '', 'chunk_ids': [], 'source_document_ids': [],
                           'final_source_document_ids': [],
                           'generation_metrics': {'prompt_tokens': 7, 'completion_tokens': 2}}
                rows.append(row)
                save(root / 'attempts' / system / f'{qid}.json', row)
            save_lines(root / f'{system}.predictions.jsonl', rows)
            count = len(rows) - (system == 'graph-global')
            summary['systems'][system] = {'attempts': len(rows), 'success': count,
                'failed': len(rows) - count, 'prompt_tokens': count * 100, 'completion_tokens': count * 10}
        save(root / 'summary.json', summary)
        return root

    def test_counts_all_current_retry_attempts_once_not_recovery_backups(self):
        report = self.report()
        actual = report.get('extraction', {}).get('cost', {})
        self.assertEqual({key: actual.get(key) for key in
                          ('records', 'attempts', 'retry_attempts', 'recovery_attempts',
                           'prompt_tokens', 'completion_tokens', 'sum_attempt_wall_seconds',
                           'sum_request_wall_seconds')},
                         {'records': 2, 'attempts': 4, 'retry_attempts': 2, 'recovery_attempts': 2,
                          'prompt_tokens': 100, 'completion_tokens': 10, 'sum_attempt_wall_seconds': 34,
                          'sum_request_wall_seconds': 30})

    def test_reports_session_wall_separately_and_flags_interrupted_history(self):
        report = self.report()
        sessions = report['extraction'].get('sessions', {})
        self.assertEqual({key: sessions.get(key) for key in
                          ('saved_session_wall_seconds', 'saved_session_request_seconds',
                           'attempted_records', 'historical_wall_complete')},
                         {'saved_session_wall_seconds': 3, 'saved_session_request_seconds': 7,
                          'attempted_records': 1, 'historical_wall_complete': False})

    def test_rejects_errors_or_changed_hashes_counts_and_provenance(self):
        cases = [('extraction_status.json', 'complete', False),
                 ('extraction_status.json', 'completed_chunks', 1),
                 ('extraction_status.json', 'extraction_config_sha256', 'wrong'),
                 (f'records/{digest("c0")}.json', 'status', 'error'),
                 (f'records/{digest("c0")}.json', 'text_sha256', 'wrong'),
                 (f'records/{digest("c0")}.json', 'attempts', []),
                 ('index.cost.json', 'communities', 2),
                 ('summaries/summary_status.json', 'complete', False),
                 ('summaries/summary_status.json', 'completed', 0),
                 ('summaries/summary_status.json', 'summary_config_sha256', 'wrong'),
                 ('summaries/records/com_one.json', 'status', 'error'),
                 ('summaries/records/com_one.json', 'claims',
                  [{'text': 'Unsupported', 'quote': 'invented quotation', 'chunk_id': 'c0'}]),
                 ('source_audit.json', 'passed', False),
                 ('source_audit.json', 'index_sha256', 'wrong')]
        for relative, key, value in cases:
            with self.subTest(artifact=relative, field=key):
                path = self.graph / relative
                original = path.read_bytes()
                row = json.loads(original)
                row[key] = value
                save(path, row)
                try:
                    with self.assertRaises(ValueError):
                        self.report()
                finally:
                    path.write_bytes(original)

    def test_absent_optional_splits_are_pending_not_failed_or_scored(self):
        report = self.report()
        runs = report.get('runs', {})
        self.assertEqual({split: {key: runs.get(split, {}).get(key) for key in
                                  ('status', 'present', 'expected_attempts', 'saved_attempts')}
                          for split in ('development', 'preliminary', 'fold_a', 'fold_b')},
                         {split: {'status': 'pending' if split in ('development', 'preliminary') else 'deferred_not_requested', 'present': False,
                                  'expected_attempts': expected, 'saved_attempts': 0}
                          for split, expected in [('development', 24), ('preliminary', 80),
                                                  ('fold_a', 200), ('fold_b', 200)]})
        self.assertEqual(report.get('correctness'), 'pending_external_grading')
        self.assertEqual(report.get('routing'), 'deferred_until_paired_correctness')

    def test_exports_actual_runner_costs_identity_and_validation_including_failure_usage(self):
        root = self.make_run()
        run = self.report().get('runs', {}).get('development', {})
        cost = run.get('cost', {})
        self.assertEqual({key: cost.get(key) for key in
                          ('attempts', 'success', 'failed', 'prompt_tokens', 'completion_tokens',
                           'sum_total_query_seconds', 'attempts_without_total_query_seconds')},
                         {'attempts': 24, 'success': 23, 'failed': 1, 'prompt_tokens': 2307,
                          'completion_tokens': 232, 'sum_total_query_seconds': 138,
                          'attempts_without_total_query_seconds': 1})
        self.assertEqual(run.get('run_config_sha256'), file_hash(root / 'run_config.json'))
        self.assertTrue(run.get('budget_validation', {}).get('recorded_counts_within_limits'))
        self.assertTrue(run.get('provenance_validation', {}).get('passed'))
        self.assertEqual(run.get('last_invocation_wall_seconds'), 99)

    def test_rejects_success_without_context_count_or_with_inconsistent_latency(self):
        root = self.make_run()
        path = root / 'attempts/graph-local/dev0.json'
        prediction = root / 'graph-local.predictions.jsonl'
        original, original_export = path.read_bytes(), prediction.read_bytes()
        for field, value in [('context_tokens', None), ('total_query_seconds', 999), ('generation_seconds', None)]:
            with self.subTest(field=field):
                row = json.loads(original)
                if value is None:
                    row.pop(field)
                else:
                    row[field] = value
                save(path, row)
                rows = [json.loads(line) for line in original_export.splitlines()]
                rows[0] = row
                save_lines(prediction, rows)
                try:
                    with self.assertRaises(ValueError):
                        self.report()
                finally:
                    path.write_bytes(original)
                    prediction.write_bytes(original_export)

    def test_cli_writes_only_after_validation_and_missing_summary_cannot_replace_report(self):
        target = self.root / 'report/completion.json'
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/report_graph_delivery.py'),
                   '--config', str(self.config_path), '--extraction', str(self.graph), '--output', str(target)]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(target.exists(), 'CLI did not save its validated report')
        original = target.read_bytes()
        self.assertTrue(json.loads(original)['graph_artifacts_validated'])
        (self.graph / 'summaries/records/com_one.json').unlink()
        failed = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(target.read_bytes(), original)

    def test_combines_summary_retries_with_extraction_without_inventing_historical_wall(self):
        totals = self.report().get('preprocessing', {})
        cost = totals.get('cost', {})
        self.assertEqual({key: cost.get(key) for key in
                          ('attempts', 'retry_attempts', 'recovery_attempts', 'model_calls',
                           'prompt_tokens', 'completion_tokens', 'sum_attempt_wall_seconds', 'sum_request_wall_seconds')},
                         {'attempts': 6, 'retry_attempts': 3, 'recovery_attempts': 2, 'model_calls': 6,
                          'prompt_tokens': 111, 'completion_tokens': 13, 'sum_attempt_wall_seconds': 39,
                          'sum_request_wall_seconds': 33})
        self.assertEqual(totals.get('index_wall_seconds'), 1.5)
        self.assertIsNone(totals.get('historical_wall_seconds'))

    def test_cli_cannot_overwrite_a_current_checkpoint(self):
        target = self.graph / 'records' / f'{digest("c0")}.json'
        original = target.read_bytes()
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/report_graph_delivery.py'),
                   '--config', str(self.config_path), '--extraction', str(self.graph), '--output', str(target)]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(target.read_bytes(), original)

    def test_grading_handoff_exposes_saved_development_exports_not_invented_grades(self):
        root = self.make_run()
        report = self.report()
        handoff = report.get('development_grading_handoff', {})
        self.assertEqual(handoff.get('status'), 'pending_external_grading')
        self.assertEqual({item['path'] for item in handoff.get('prediction_exports', [])},
                         {str(root / f'{system}.predictions.jsonl') for system in ('graph-local', 'graph-global')})
        self.assertEqual(len(report.get('external_blockers', [])), 2)
        self.assertFalse(any('T4' in item for item in report['external_blockers']))
        self.assertEqual(report['T4_validation'], 'unverified_no_actual_T4_measurements')
        self.assertIsNone(report['gpu_hours'])
        self.assertIsNone(report['electricity_cost'])

    def test_paired_scope_finishes_without_core_folds_or_routing_training(self):
        self.make_run('development')
        self.make_run('preliminary')
        report = self.report()
        self.assertTrue(report['all_graph_attempts_saved'])
        self.assertEqual(report['routing'], 'deferred_until_paired_correctness')
        self.assertEqual(report['runs']['fold_a']['status'], 'deferred_not_requested')
        self.assertFalse(report['runs']['fold_b']['required_for_current_scope'])
        self.assertFalse(any('corpus-global' in item for item in report['external_blockers']))

    def test_collected_t4_results_validate_content_hashes_across_mount_paths(self):
        root = self.make_run()
        path = root / 'run_config.json'
        config = json.loads(path.read_text())
        for system, artifacts in config['artifacts'].items():
            config['artifacts'][system] = {'/kaggle/working/paired_graph/' + Path(p).name: h
                                            for p, h in artifacts.items()}
        save(path, config)
        report = self.report()
        self.assertEqual(report['runs']['development']['saved_attempts'], 24)
        self.assertEqual(report['runs']['development']['run_config'], config)

    def test_partial_unrequested_core_directory_cannot_block_paired_report(self):
        self.make_run('development')
        self.make_run('preliminary')
        (self.output / 'runs/fold_a').mkdir()
        try:
            report = self.report()
        except Exception as exc:
            self.fail(f'Unrequested core directory blocked paired reporting: {exc}')
        self.assertTrue(report['all_graph_attempts_saved'])
        self.assertEqual(report['runs']['fold_a']['status'], 'deferred_not_requested')
        self.assertTrue(report['runs']['fold_a']['present'])
        self.assertIn('warning', report['runs']['fold_a'])

    def test_packing_cost_is_separate_from_retrieval_and_generation(self):
        self.make_run()
        report = self.report()
        self.assertEqual(report['runs']['development']['cost'].get('sum_packing_seconds'), 46)

    def test_missing_original_query_hash_cannot_claim_immutable_text(self):
        path = self.output / 'questions/selection.json'
        selection = json.loads(path.read_text())
        selection['input_sha256'] = {}
        save(path, selection)
        with self.assertRaisesRegex(ValueError, 'question hash'):
            self.report()


if __name__ == '__main__':
    unittest.main()
