"""Baseline contracts: real lexical search, cosine ranking, resumption and leakage."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_rag.common import digest, file_digest, read_json, read_jsonl, write_json, write_jsonl
from baseline_rag.corpus import Corpus, prepare_questions
from baseline_rag.retrieval import BM25Retriever, VectorRetriever, build_bm25, build_vector, load_retriever
from baseline_rag.run import run
from baseline_rag.generation import SharedGenerator


class TinyEncoder:
    calls = []

    def __init__(self, settings):
        pass

    def encode(self, texts, query=False):
        import numpy as np
        self.calls.extend(texts)
        return np.array([[1., 0.] if 'alpha' in t.lower() else [0., 1.] for t in texts], dtype=np.float32)

    def close(self):
        pass


class BaselineContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'
        self.output = self.root / 'output'
        chunks = [{'chunk_id': f'd{i}::v::0', 'source_document_id': f'd{i}',
                   'text': text, 'token_count_with_special_tokens': 12} for i, text in enumerate([
            'alpha deployment staging', 'beta billing invoice', 'alpha rollout rollback', 'beta payment retry'])]
        sources = [{'source_document_id': f'd{i}', 'path': f'github/doc{i}.json', 'sha256': '0' * 64} for i in range(4)]
        write_jsonl(self.bundle / 'chunks/chunks.jsonl', chunks)
        write_jsonl(self.bundle / 'corpus_manifest.jsonl', sources)
        write_json(self.bundle / 'bundle.json', {'document_count': 4, 'chunk_count': 4})
        write_json(self.bundle / 'chunks/chunk_config.json', {
            'chunks_sha256': file_digest(self.bundle / 'chunks/chunks.jsonl'),
            'source_manifest_sha256': file_digest(self.bundle / 'corpus_manifest.jsonl')})
        self.corpus = Corpus(self.bundle)
        self.embedding = {'dimensions': 2, 'batch_size': 2}
        self.config = {'bundle': str(self.bundle), 'output': str(self.output),
                       'generation_settings': {'model': 'fixture'}, 'graph_index': str(self.root / 'graph.json'),
                       'graph_summaries': str(self.root / 'summaries')}
        self.make_questions()

    def tearDown(self):
        self.temp.cleanup()

    def make_questions(self):
        categories = ['basic', 'intra_document_reasoning', 'completeness', 'semantic']
        all_ids, dev_ids = {}, []
        for split in ['development', 'fold_a', 'fold_b']:
            labels = []
            for index in range(12 if split == 'development' else 100):
                qid = f'{split}_{index:03d}'
                label = {'question_id': qid, 'question': f'alpha deployment {index}',
                    'question_type': categories[index % 4], 'expected_doc_ids': ['d0'],
                    'gold_answer': 'GOLD_SECRET_MUST_NEVER_REACH_MODEL', 'answer_facts': ['GOLD_SECRET_FACT']}
                labels.append(label)
            write_jsonl(self.bundle / 'questions' / f'{split}.labels.jsonl', labels)
            write_jsonl(self.bundle / 'questions' / f'{split}.queries.jsonl',
                        [{'question_id': q['question_id'], 'question': q['question']} for q in labels])
            ids = [q['question_id'] for q in labels]
            if split == 'development':
                dev_ids = ids
            else:
                all_ids[split] = ids
        write_json(self.bundle / 'questions/splits.json', {'development_ids': dev_ids, 'folds': all_ids})
        write_json(self.bundle / 'questions/coverage_exclusions.json', {})
        prepare_questions(self.bundle, self.output / 'questions')

    def test_checksum_and_document_coverage_are_required(self):
        self.assertEqual(self.corpus.identity['document_count'], 4)
        with (self.bundle / 'chunks/chunks.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            Corpus(self.bundle)

    def test_bm25_ranks_real_text_and_preserves_source_links(self):
        build_bm25(self.corpus, self.output / 'indexes/bm25')
        retriever = BM25Retriever(self.output / 'indexes/bm25', self.corpus.identity)
        try:
            result = retriever.retrieve('invoice', 2)
            self.assertEqual(result[0]['source_document_id'], 'd1')
            self.assertGreater(result[0]['score'], 0)
            self.assertEqual(result[0]['source_links'][0]['source_path'], 'github/doc1.json')
            self.assertEqual(retriever.retrieve('!!!', 2), [])
            retriever.retrieve('" OR 1=1; DROP TABLE chunks; --', 2)
            self.assertEqual(retriever.db.execute('SELECT count(*) FROM chunks').fetchone()[0], 4)
        finally:
            retriever.close()

    def test_vector_cosine_search_and_corpus_mismatch(self):
        build_bm25(self.corpus, self.output / 'indexes/bm25')
        build_vector(self.corpus, self.output / 'indexes/vector', self.embedding, TinyEncoder)
        retriever = VectorRetriever(self.output / 'indexes/vector', self.output / 'indexes/bm25',
                                    self.corpus.identity, TinyEncoder)
        try:
            result = retriever.retrieve('alpha', 2)
            self.assertEqual([r['source_document_id'] for r in result], ['d0', 'd2'])
            self.assertAlmostEqual(result[0]['score'], 1)
        finally:
            retriever.close()
        with self.assertRaisesRegex(ValueError, 'different corpus'):
            VectorRetriever(self.output / 'indexes/vector', self.output / 'indexes/bm25', {'wrong': 'corpus'}, TinyEncoder)

    def test_embedding_crash_resumes_after_last_complete_batch(self):
        class Interrupted(TinyEncoder):
            def encode(self, texts, query=False):
                if 'rollout' in texts[0]:
                    raise RuntimeError('power loss')
                return super().encode(texts, query)
        target = self.output / 'vector'
        with self.assertRaisesRegex(RuntimeError, 'power loss'):
            build_vector(self.corpus, target, self.embedding, Interrupted)
        self.assertEqual(read_json(target / 'progress.json')['completed'], 2)
        TinyEncoder.calls = []
        result = build_vector(self.corpus, target, self.embedding, TinyEncoder)
        self.assertEqual(result['indexed_chunks'], 4)
        self.assertEqual(len(TinyEncoder.calls), 2)

    def test_preliminary_is_balanced_disjoint_and_contains_no_labels(self):
        selected = read_jsonl(self.output / 'questions/preliminary.queries.jsonl')
        meta = read_json(self.output / 'questions/metadata.json')
        dev = {q['question_id'] for q in read_jsonl(self.output / 'questions/development.queries.jsonl')}
        self.assertEqual(len(selected), 40)
        self.assertFalse(dev & {q['question_id'] for q in selected})
        counts = Counter((meta[q['question_id']]['fold'], meta[q['question_id']]['project_type']) for q in selected)
        self.assertEqual(set(counts.values()), {5})
        self.assertTrue(all(set(q) == {'question_id', 'question'} for q in selected))

    def test_all_four_retrievers_use_same_generator_and_resume_without_leakage(self):
        calls = []
        settings = self.config['generation_settings']
        class Adapter:
            def __init__(self, supplied):
                self.settings = supplied
                self.identity = {'fixture': True}
            def count_tokens(self, text):
                return len(text.split())
            def generate(self, question, context):
                self_outer.assertNotIn('GOLD_SECRET', question + context)
                self_outer.assertLessEqual(self.count_tokens(context), 8)
                calls.append((question, context))
                return {'answer': 'Alpha uses staging [d0::v::0]', 'metrics': {'prompt_tokens': 8, 'completion_tokens': 5}}
        self_outer = self
        class Retriever:
            def retrieve_with_trace(self, query, k=10):
                return {'latency_seconds': .01, 'evidence': [
                    {'id': 'e1', 'text': 'alpha staging', 'score': 1,
                     'source_links': [{'chunk_id': 'd0::v::0', 'source_document_id': 'd0'}]},
                    {'id': 'e2', 'text': 'beta billing ' * 20, 'score': .5,
                     'source_links': [{'chunk_id': 'd1::v::0', 'source_document_id': 'd1'}]}]}
        frozen = {'generation_settings': settings, 'context_token_budget': 8, 'retrieval_k': 10}
        names = ['bm25', 'vector', 'graph-local', 'graph-global']
        with patch('builtins.print'):
            report = run(self.config, self.corpus, frozen, 'preliminary', names,
                         adapter_factory=Adapter, retriever_factory=lambda *args: Retriever())
            self.assertEqual(report['saved_attempts'], 160)
            self.assertTrue(report['all_answers_successful'])
            run(self.config, self.corpus, frozen, 'preliminary', names,
                adapter_factory=Adapter, retriever_factory=lambda *args: Retriever())
        self.assertEqual(len(calls), 160)
        rows = read_jsonl(self.output / 'runs/preliminary/vector.predictions.jsonl')
        self.assertTrue(all(r['final_source_document_ids'] == ['d0'] for r in rows))
        frozen['context_token_budget'] = 9
        with self.assertRaisesRegex(ValueError, 'Configuration changed'):
            run(self.config, self.corpus, frozen, 'preliminary', names, adapter_factory=Adapter)

    def test_initialization_failures_are_saved_for_every_question(self):
        def unavailable(_):
            raise RuntimeError('model not downloaded')
        frozen = {'generation_settings': {}, 'context_token_budget': 8, 'retrieval_k': 10}
        with patch('builtins.print'):
            report = run(self.config, self.corpus, frozen, 'development', ['bm25'], adapter_factory=unavailable)
        self.assertEqual(report['systems']['bm25']['failed'], 12)
        rows = read_jsonl(self.output / 'runs/development/bm25.predictions.jsonl')
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r['answer'] is None and 'not downloaded' in r['error'] for r in rows))

    def test_sampled_graph_is_rejected_for_frozen_comparison(self):
        metadata = {'chunks_sha256': self.corpus.identity['chunks_sha256'],
                    'source_manifest_sha256': self.corpus.identity['source_manifest_sha256'],
                    'indexed_chunks': 1, 'total_corpus_chunks': 4, 'corpus_scope': 'pilot_sample'}
        with patch('graph_rag.retrieve.GraphLocalRetriever', return_value=SimpleNamespace(index={'metadata': metadata})):
            with self.assertRaisesRegex(ValueError, 'complete frozen'):
                load_retriever('graph-local', self.config, self.corpus.identity)

    def test_real_tokenizer_chat_output_and_full_prompt_cap(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        raw = Tokenizer(models.WordLevel({'[UNK]': 0, 'alpha': 1, 'beta': 2}, unk_token='[UNK]'))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]')
        tokenizer.chat_template = "{{ messages[0]['content'] }} {{ messages[1]['content'] }}"
        adapter = SharedGenerator.__new__(SharedGenerator)
        adapter.tokenizer = tokenizer
        adapter.settings = {'backend': 'openai-compatible', 'system_prompt': 'alpha',
                            'user_template': '{context} {question}', 'max_input_tokens': 4}
        adapter._remote = lambda messages, count: {'answer': 'ok', 'metrics': {'prompt_tokens': count}}
        result = adapter.generate('beta', 'alpha alpha')
        self.assertEqual(result['metrics']['prompt_tokens'], 4)
        self.assertEqual(adapter.count_tokens('alpha beta'), 2)
        with self.assertRaisesRegex(ValueError, 'Full prompt exceeds'):
            adapter.generate('beta', 'alpha alpha alpha alpha')

    def test_atomic_progress_update_tolerates_an_open_reader(self):
        from concurrent.futures import ThreadPoolExecutor
        import time
        path = self.output / 'progress.json'
        write_json(path, {'completed': 1})
        with ThreadPoolExecutor(max_workers=1) as pool:
            with path.open('r', encoding='utf-8') as reader:
                future = pool.submit(write_json, path, {'completed': 2})
                time.sleep(.1)
                self.assertEqual(json.load(reader)['completed'], 1)
            future.result(timeout=5)
        self.assertEqual(read_json(path)['completed'], 2)

    def test_ollama_rejects_a_changed_model_digest(self):
        settings = {'backend': 'ollama', 'model': 'fixture', 'revision': 'a' * 40,
            'cache_dir': str(self.root / 'cache'), 'temperature': 0, 'served_model': 'fixture:3b',
            'server_version': '0.18.0', 'server_model_digest': 'expected', 'server_quantization': 'Q4_K_M'}
        tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: 'fixed'), chat_template='fixed')
        responses = [{'version': '0.18.0'}, {'models': [{'name': 'fixture:3b', 'digest': 'changed',
                                                       'details': {'quantization_level': 'Q4_K_M'}}]}]
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
            with patch.object(SharedGenerator, '_request', side_effect=responses):
                with self.assertRaisesRegex(ValueError, 'digest differs'):
                    SharedGenerator(settings)

    def test_ollama_gpu_requirement_does_not_accept_cpu_fallback(self):
        adapter = SharedGenerator.__new__(SharedGenerator)
        adapter.settings = {'backend': 'ollama', 'served_model': 'fixture:3b', 'model': 'fixture',
            'seed': 42, 'max_new_tokens': 8, 'max_input_tokens': 128, 'require_gpu': True}
        adapter._request = lambda path, payload=None: ({'models': []} if path == '/api/ps' else {
            'message': {'content': 'CPU answer'}, 'eval_count': 2, 'prompt_eval_count': 4})
        with self.assertRaisesRegex(RuntimeError, 'no GPU residency'):
            adapter._remote([{'role': 'user', 'content': 'fixture'}], 4)


if __name__ == '__main__':
    unittest.main()
