"""Read-only final baseline validation; never generate answers or invent labels."""
from collections import Counter
import math
from pathlib import Path

from coverage import score_coverage
from graph_rag.retrieve import pack_context
from .common import file_digest, index_root, read_json, read_jsonl
from .corpus import Corpus, TYPES, TYPE_MAP


SYSTEMS = ('bm25', 'vector')
SPLITS = ('fold_a', 'fold_b')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_inputs(config):
    """Check the original experiment and reusable indexes without writing them."""
    corpus = Corpus(config['bundle'])
    reference_path = Path(config['reference_experiment'])
    reference = read_json(reference_path)
    require(corpus.identity == reference['corpus'], 'Corpus differs from delivered baseline')
    for key in ('embedding', 'generation_settings', 'retrieval_k', 'context_token_budget'):
        require(config[key] == reference[key], f'{key} differs from delivered baseline')
    root = Path(config['output']).resolve()
    for protected in (reference_path.parent, corpus.root, index_root(config)):
        protected = protected.resolve()
        require(root != protected and protected not in root.parents and root not in protected.parents,
                f'Final output must be separate from {protected}')
    manifest = read_json(config['question_manifest'])
    bundle_splits = read_json(corpus.root / 'questions/splits.json')
    require(bundle_splits['folds'] == manifest['folds'] and
            bundle_splits['development_ids'] == manifest['development_ids'],
            'Bundle question splits differ from repository manifest')
    selection = read_json(reference_path.parent / 'questions/selection.json')
    require(selection['development_ids'] == reference['development_ids'] and
            selection['preliminary_question_ids'] == reference['preliminary_question_ids'],
            'Delivered selection differs from reference experiment')
    questions, labels, seen = {}, {}, set(manifest['development_ids'])
    for split in SPLITS:
        qpath = corpus.root / 'questions' / f'{split}.queries.jsonl'
        lpath = corpus.root / 'questions' / f'{split}.labels.jsonl'
        for path in (qpath, lpath):
            require(file_digest(path) == selection['input_sha256'][path.name],
                    f'Question input checksum mismatch: {path}')
        queries, gold = read_jsonl(qpath), read_jsonl(lpath)
        ids = [q['question_id'] for q in queries]
        require(len(ids) == len(set(ids)) == 100 and set(ids) == set(manifest['folds'][split])
                and not seen.intersection(ids), f'{split}: wrong, duplicate or overlapping IDs')
        require(queries == [{'question_id': q['question_id'], 'question': q['question']} for q in gold],
                f'{split}: query/label mismatch or label leakage')
        require(Counter(TYPE_MAP[q['question_type']] for q in gold) == Counter(dict.fromkeys(TYPES, 25)),
                f'{split}: question types are not balanced')
        seen.update(ids)
        questions[split] = queries
        labels.update({q['question_id']: q for q in gold})
    indexes = {}
    for system, filename, hash_key in (('bm25', 'index.sqlite', 'index_sha256'),
                                       ('vector', 'embeddings.npy', 'embeddings_sha256')):
        directory = index_root(config) / system
        meta = read_json(directory / 'metadata.json')
        require(meta['corpus'] == corpus.identity and meta['indexed_chunks'] == len(corpus.chunks),
                f'{system}: index corpus mismatch')
        require(file_digest(directory / filename) == meta[hash_key], f'{system}: index checksum mismatch')
        if system == 'vector':
            require(meta['embedding'] == config['embedding'], 'Vector embedding settings mismatch')
        indexes[system] = meta
    return corpus, reference, questions, labels, indexes


def audit_success(row, question, system, corpus_map, sources, config, tokenizer):
    require(row['question'] == question['question'] and row['retriever'] == system,
            'Question text or retriever mismatch')
    require(isinstance(row['answer'], str) and bool(row['answer'].strip()), 'Empty successful answer')
    evidence = row['evidence']
    require(bool(evidence), 'Missing supplied evidence')
    for item in evidence:
        require(bool(item['source_links']), 'Missing provenance')
        for link in item['source_links']:
            chunk = corpus_map[link['chunk_id']]
            source = sources[chunk['source_document_id']]
            require(chunk['text'] == item['text'] and chunk['source_document_id'] == link['source_document_id']
                    and source['path'] == link['source_path'] and source['sha256'] == link['source_sha256'],
                    'Source text or provenance mismatch')
    packed = pack_context(evidence, tokenizer.count_tokens, config['context_token_budget'])
    for key in ('context', 'context_tokens', 'chunk_ids', 'source_document_ids'):
        require(row[key] == packed[key], f'Packed {key} mismatch')
    require(row['final_source_document_ids'] == packed['source_document_ids'] == row['document_ids'],
            'Final source IDs mismatch')
    prompt_tokens = len(tokenizer.tokenizer.apply_chat_template(
        tokenizer.messages(question['question'], row['context']), tokenize=True, add_generation_prompt=True))
    metrics = row['generation_metrics']
    require(prompt_tokens <= config['generation_settings']['max_input_tokens'] and
            metrics['prompt_tokens'] == prompt_tokens, 'Full prompt token count/budget mismatch')
    completion = metrics['completion_tokens']
    require(type(completion) is int and 0 <= completion <= config['generation_settings']['max_new_tokens'],
            'Invalid completion token count')
    require(metrics['finish_reason'] in ('stop', 'length'), 'Missing or unknown finish reason')
    require(completion != config['generation_settings']['max_new_tokens'] or metrics['finish_reason'] == 'length',
            'Output cap flag missing')
    settings = config['generation_settings']
    if settings.get('backend'):
        require(metrics.get('backend') == settings['backend'], 'Generation backend mismatch')
    if settings.get('require_gpu') and settings.get('backend') == 'ollama':
        residency = metrics.get('gpu_residency') or {}
        require(residency.get('digest') == settings['server_model_digest'] and
                residency.get('size_vram', 0) > 0, 'Missing pinned model GPU residency')
    for key in ('retrieval_seconds', 'packing_seconds', 'generation_seconds', 'total_query_seconds'):
        require(type(row[key]) in (float, int) and math.isfinite(row[key]) and row[key] >= 0,
                f'Invalid {key}')
    require(math.isclose(row['total_query_seconds'], sum(row[k] for k in
            ('retrieval_seconds', 'packing_seconds', 'generation_seconds')), rel_tol=1e-6, abs_tol=1e-6),
            'Total query time differs from component timings')


def audit(config, tokenizer):
    corpus, reference, questions, labels, indexes = verify_inputs(config)
    root = Path(config['output'])
    require(read_json(root / 'experiment.json') == reference, 'Final experiment differs from delivered baseline')
    corpus_map = {c['chunk_id']: c for c in corpus.chunks}
    exclusions = read_json(corpus.root / 'questions/coverage_exclusions.json')
    problems, measurements, artifacts, runs = [], [], {}, {}
    for split in SPLITS:
        qpath = root / 'questions' / f'{split}.queries.jsonl'
        require(read_jsonl(qpath) == questions[split], f'{split}: prepared question inputs changed')
        out = root / 'runs' / split
        rc = read_json(out / 'run_config.json')
        require(rc['experiment'] == reference and rc['questions_sha256'] == file_digest(qpath)
                and rc['retrievers'] == list(SYSTEMS), f'{split}: run identity mismatch')
        require(bool(rc['implementation']) and bool(rc['environment']), f'{split}: missing run provenance')
        for system in SYSTEMS:
            expected_artifacts = {str(index_root(config) / system / 'metadata.json'):
                                  file_digest(index_root(config) / system / 'metadata.json')}
            if system == 'vector':
                p = index_root(config) / 'bm25/metadata.json'
                expected_artifacts[str(p)] = file_digest(p)
            require(rc['artifacts'][system] == expected_artifacts, f'{split}/{system}: index identity mismatch')
        artifacts[str(out / 'run_config.json')] = file_digest(out / 'run_config.json')
        qmap = {q['question_id']: q for q in questions[split]}
        for system in SYSTEMS:
            key = f'{split}/{system}'
            path = out / f'{system}.predictions.jsonl'
            if not path.exists():
                problems.append(f'{key}: predictions missing')
                continue
            rows = read_jsonl(path)
            ids = [r['question_id'] for r in rows]
            if len(ids) != len(set(ids)) or set(ids) != set(qmap):
                problems.append(f'{key}: missing, extra or duplicate attempts')
            checkpoint_ids = {p.stem for p in (out / 'attempts' / system).glob('*.json')}
            if checkpoint_ids != set(qmap):
                problems.append(f'{key}: missing or extra checkpoints')
            artifacts[str(path)] = file_digest(path)
            runs[key] = dict(Counter(r.get('status') for r in rows))
            for row in rows:
                qid = row['question_id']
                try:
                    require(qid in qmap and row['question'] == qmap[qid]['question'] and
                            row['retriever'] == system, 'Unknown question or retriever')
                    require(read_json(out / 'attempts' / system / f'{qid}.json') == row,
                            'Export differs from saved checkpoint')
                    if row['status'] == 'success':
                        audit_success(row, qmap[qid], system, corpus_map, corpus.sources, config, tokenizer)
                        coverage = score_coverage(qid, labels[qid]['expected_doc_ids'],
                                                  row['final_source_document_ids'], exclusions)
                    else:
                        require(row['status'] == 'failed' and isinstance(row.get('error'), str)
                                and bool(row['error'].strip()), 'Failure needs explicit error')
                        coverage = {'coverage': None, 'coverage_na_reason': 'Failed run'}
                    metrics = row.get('generation_metrics', {})
                    measurements.append({'split': split, 'question_id': qid, 'retriever': system,
                        'project_type': TYPE_MAP[labels[qid]['question_type']], 'status': row['status'],
                        'error': row.get('error'), 'coverage': coverage['coverage'],
                        'coverage_na_reason': coverage.get('coverage_na_reason'), 'correctness': None,
                        'context_tokens': row.get('context_tokens'), 'prompt_tokens': metrics.get('prompt_tokens'),
                        'completion_tokens': metrics.get('completion_tokens'),
                        'output_limit_hit': metrics.get('finish_reason') == 'length',
                        **{k: row.get(k) for k in ('retrieval_seconds', 'packing_seconds',
                                                  'generation_seconds', 'total_query_seconds')}})
                except (KeyError, ValueError, TypeError, OSError) as exc:
                    problems.append(f'{key}/{qid}: {exc}')
    failed = sum(r['status'] == 'failed' for r in measurements)
    by_type = {}
    for system in SYSTEMS:
        by_type[system] = {}
        for kind in TYPES:
            group = [r for r in measurements if r['retriever'] == system and r['project_type'] == kind]
            applicable = [r['coverage'] for r in group if r['coverage'] is not None]
            successes = [r for r in group if r['status'] == 'success']
            by_type[system][kind] = {'attempts': len(group),
                'failed': sum(r['status'] == 'failed' for r in group),
                'coverage_applicable': len(applicable),
                'mean_coverage': sum(applicable) / len(applicable) if applicable else None,
                'mean_context_tokens': sum(r['context_tokens'] for r in successes) / len(successes) if successes else None,
                'mean_query_seconds': sum(r['total_query_seconds'] for r in successes) / len(successes) if successes else None,
                'output_limit_hits': sum(r['output_limit_hit'] for r in group), 'correctness': None}
    return {'scope': '200 evaluation questions per baseline; development excluded',
            'expected_attempts': 400, 'validated_attempts': len(measurements), 'failed_attempts': failed,
            'all_attempts_saved': len(measurements) == 400 and not problems,
            'all_answers_successful': len(measurements) == 400 and not problems and not failed,
            'audit_problems': problems, 'runs': runs, 'by_type': by_type, 'artifacts': artifacts, 'indexing': indexes,
            'correctness': 'Pending independent grading; never inferred from coverage',
            'measurement_notes': 'Indexing receipts describe reused indexes, not new build costs. '
                'Query timings describe the saved runs; cross-hardware comparisons require qualifications.'}, measurements
