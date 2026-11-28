"""Exercise final-run completeness, immutable inputs and corrupt saved evidence."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_rag.common import read_json, read_jsonl, write_json, write_jsonl
from baseline_rag.corpus import prepare
from baseline_rag.final import audit, verify_inputs
from baseline_rag.generation import SharedGenerator
from baseline_rag.retrieval import build_bm25, build_vector, evidence
from baseline_rag.run import _artifact_identity
from graph_rag.retrieve import pack_context
import test_baseline_rag as fixtures


class TokenizerFixture:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m['content'].split()) for m in messages)))


class FinalBaselineContracts(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BaselineContracts()
        self.fixture.setUp()
        self.root = self.fixture.root
        old = self.fixture.output
        settings = {'system_prompt': 'Use the evidence.', 'user_template': '{context} {question}',
                    'max_input_tokens': 200, 'max_new_tokens': 10}
        original = {**self.fixture.config, 'embedding': self.fixture.embedding,
                    'seed': 42, 'retrieval_k': 10, 'context_token_budget': 100,
                    'generation_settings': settings}
        corpus, reference, _ = prepare(original)
        build_bm25(corpus, old / 'indexes/bm25')
        build_vector(corpus, old / 'indexes/vector', original['embedding'], fixtures.TinyEncoder)
        self.config = {**original, 'output': str(self.root / 'final'),
                       'index_root': str(old / 'indexes'), 'reference_experiment': str(old / 'experiment.json'),
                       'question_manifest': str(corpus.root / 'questions/splits.json')}
        prepare(self.config)
        self.adapter = SharedGenerator.__new__(SharedGenerator)
        self.adapter.settings = settings
        self.adapter.count_tokens = lambda text: len(text.split())
        self.adapter.tokenizer = TokenizerFixture()
        chunk, source = corpus.chunks[0], corpus.sources['d0']
        packed = pack_context([evidence((chunk['chunk_id'], 'd0', chunk['text'],
                                       source['path'], source['sha256']), 1, 'bm25')],
                              self.adapter.count_tokens, 100)
        for split in ('fold_a', 'fold_b'):
            queries = read_jsonl(corpus.root / 'questions' / f'{split}.queries.jsonl')
            out = Path(self.config['output']) / 'runs' / split
            from baseline_rag.common import file_digest
            qpath = Path(self.config['output']) / 'questions' / f'{split}.queries.jsonl'
            write_json(out / 'run_config.json', {'experiment': reference, 'retrievers': ['bm25', 'vector'],
                'questions_sha256': file_digest(qpath), 'implementation': {'fixture': 'test'},
                'environment': {'fixture': True},
                'artifacts': {s: _artifact_identity(self.config, s) for s in ('bm25', 'vector')}})
            for system in ('bm25', 'vector'):
                rows = []
                for q in queries:
                    row = {**q, **deepcopy(packed), 'retriever': system, 'status': 'success', 'answer': 'Alpha',
                        'document_ids': ['d0'], 'final_source_document_ids': ['d0'],
                        'retrieval_seconds': 1., 'packing_seconds': .1, 'generation_seconds': 2.,
                        'total_query_seconds': 3.1, 'generation_metrics': {
                            'prompt_tokens': len(self.adapter.tokenizer.apply_chat_template(
                                self.adapter.messages(q['question'], packed['context']))),
                            'completion_tokens': 10, 'finish_reason': 'length'}}
                    rows.append(row)
                    write_json(out / 'attempts' / system / f'{q["question_id"]}.json', row)
                write_jsonl(out / f'{system}.predictions.jsonl', rows)
        self.path = Path(self.config['output']) / 'runs/fold_a/bm25.predictions.jsonl'

    def tearDown(self):
        self.fixture.tearDown()

    def save_rows(self, rows, checkpoints=True):
        write_jsonl(self.path, rows)
        if checkpoints:
            for row in rows:
                write_json(self.path.parent / 'attempts/bm25' / f'{row["question_id"]}.json', row)

    def test_complete_run_preserves_cap_flags_and_leaves_correctness_ungraded(self):
        report, rows = audit(self.config, self.adapter)
        self.assertTrue(report['all_attempts_saved'], report['audit_problems'])
        self.assertTrue(report['all_answers_successful'])
        self.assertEqual(report['validated_attempts'], 400)
        self.assertTrue(all(r['correctness'] is None and r['output_limit_hit'] for r in rows))

    def test_explicit_failure_is_complete_but_not_successful(self):
        rows = read_jsonl(self.path)
        rows[0] = {k: rows[0][k] for k in ('question_id', 'question', 'retriever')}
        rows[0].update(status='failed', error='RuntimeError: GPU unavailable')
        self.save_rows(rows)
        report, measured = audit(self.config, self.adapter)
        self.assertTrue(report['all_attempts_saved'], report['audit_problems'])
        self.assertFalse(report['all_answers_successful'])
        self.assertEqual(report['failed_attempts'], 1)
        failure = next(r for r in measured if r['status'] == 'failed')
        self.assertIsNone(failure['coverage'])
        self.assertIsNone(failure['correctness'])

    def test_rejects_corrupted_evidence_budgets_flags_timings_and_exports(self):
        original = read_jsonl(self.path)
        for corruption in ('duplicate', 'missing', 'question', 'source', 'context', 'prompt', 'cap', 'timing', 'checkpoint'):
            with self.subTest(corruption=corruption):
                rows = deepcopy(original)
                if corruption == 'duplicate':
                    rows[-1] = deepcopy(rows[0])
                elif corruption == 'missing':
                    rows.pop()
                elif corruption == 'question':
                    rows[0]['question'] = 'Changed'
                elif corruption == 'source':
                    rows[0]['evidence'][0]['source_links'][0]['source_document_id'] = 'd1'
                elif corruption == 'context':
                    rows[0]['context'] += ' invented evidence'
                elif corruption == 'prompt':
                    rows[0]['generation_metrics']['prompt_tokens'] = 201
                elif corruption == 'cap':
                    rows[0]['generation_metrics']['finish_reason'] = 'stop'
                elif corruption == 'timing':
                    rows[0]['generation_seconds'] = -1
                else:
                    rows[0]['answer'] = 'Export altered after checkpoint'
                self.save_rows(rows, checkpoints=corruption != 'checkpoint')
                report, _ = audit(self.config, self.adapter)
                self.assertFalse(report['all_attempts_saved'])
                self.assertTrue(report['audit_problems'])
                self.save_rows(original)

    def test_preflight_rejects_settings_index_and_question_tampering(self):
        for field, value in (('context_token_budget', 99), ('retrieval_k', 3),
                             ('output', str(self.fixture.output / 'overwrite'))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_inputs({**self.config, field: value})
        qpath = self.fixture.bundle / 'questions/fold_a.queries.jsonl'
        saved = qpath.read_bytes()
        qpath.write_bytes(saved + b'\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verify_inputs(self.config)
        qpath.write_bytes(saved)
        index = Path(self.config['index_root']) / 'vector/embeddings.npy'
        with index.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verify_inputs(self.config)

    def test_preflight_cli_does_not_create_final_output(self):
        config = {**self.config, 'output': str(self.root / 'not-started')}
        path = self.root / 'final-config.json'
        write_json(path, config)
        script = Path(__file__).resolve().parents[1] / 'scripts/report_final_baselines.py'
        result = subprocess.run([sys.executable, str(script), '--config', str(path), '--preflight'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(Path(config['output']).exists())


if __name__ == '__main__':
    unittest.main()
