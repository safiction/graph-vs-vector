"""Graph-side adapter to an external, shared answer generator."""
import json
from pathlib import Path
from .common import digest, file_digest, read_jsonl, write_json, append_jsonl, verified_source
from .retrieve import pack_context


def development_questions(selection_path, dataset):
    selection = json.loads(Path(selection_path).read_text())
    path = Path(dataset) / 'questions.jsonl'
    verified_source(path.read_bytes(), selection['source_sha256'])
    ids = set(selection['development_ids'])
    from config import QUESTION_CATEGORIES
    type_map = {category: kind for kind, cats in QUESTION_CATEGORIES.items() for category in cats}
    questions = [{'question_id': q['question_id'], 'question': q['question'],
                  'project_type': type_map[q['question_type']], 'fold': 'development'}
                 for q in read_jsonl(path) if q['question_id'] in ids]
    if {q['question_id'] for q in questions} != ids:
        raise ValueError('Development selection is missing')
    return sorted(questions, key=lambda q: q['question_id'])


def retrieval_demo(retrievers, selection_path, dataset, output, k=10):
    output = Path(output)
    questions = development_questions(selection_path, dataset)
    for retriever in retrievers:
        results = []
        for question in questions:
            trace = retriever.retrieve_with_trace(question['question'], k=k)
            results.append({**question, **trace, 'status': 'retrieval_only_pending_shared_generator',
                            'answer': None, 'purpose': 'development_pilot_only'})
        path = output / f'{retriever.name}.development.retrieval.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results), encoding='utf-8')
    path = output / 'development.questions.jsonl'
    path.write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in questions), encoding='utf-8')
    return {'questions_per_system': len(questions), 'systems': [r.name for r in retrievers],
            'status': 'retrieval_only_pending_shared_generator'}


def run_preliminary(retrievers, questions_path, selection_path, config_path, adapter, output):
    """adapter exposes settings, count_tokens(text), generate(question, context).

    generate returns {'answer': str, 'metrics': {...}} using the team's exact shared
    prompt/model/decoding settings. This module does not choose those settings.
    """
    from collections import Counter
    config = json.loads(Path(config_path).read_text())
    selection = json.loads(Path(selection_path).read_text())
    questions = read_jsonl(questions_path)
    qids = {q['question_id'] for q in questions}
    counts = Counter(q['project_type'] for q in questions)
    if len(qids) != 40 or len(questions) != 40 or counts != Counter({
        'single-fact': 10, 'multi-hop': 10, 'corpus-global': 10, 'fuzzy': 10}):
        raise ValueError('Preliminary set must contain 40 distinct questions, 10 per type')
    if qids & set(selection['development_ids']):
        raise ValueError('Preliminary set overlaps development')
    evaluation_ids = {q for fold in selection['folds'].values() for q in fold}
    if not qids <= evaluation_ids or sorted(qids) != sorted(config['preliminary_question_ids']):
        raise ValueError('Preliminary IDs differ from the supplied shared selection')
    if adapter.settings != config['generation_settings']:
        raise ValueError('Generator settings differ from frozen comparison settings')
    if config['context_token_budget'] < 1 or config['retrieval_k'] < 1:
        raise ValueError('Positive shared context budget and retrieval k required')
    for retriever in retrievers:
        metadata = retriever.index['metadata']
        if metadata['corpus_scope'] == 'pilot_sample' or metadata['indexed_chunks'] != metadata['total_corpus_chunks']:
            raise ValueError('A sampled pilot graph cannot be used for frozen-corpus preliminary evaluation')
        if (metadata['chunks_sha256'] != config['shared_chunks_sha256'] or
                metadata['source_manifest_sha256'] != config['frozen_source_manifest_sha256']):
            raise ValueError('Graph index does not match supplied frozen corpus')
    output = Path(output)
    run_config = {'comparison_config_sha256': file_digest(config_path), 'questions_sha256': file_digest(questions_path),
                  'selection_sha256': file_digest(selection_path), 'generation_settings': adapter.settings,
                  'graph_implementation_sha256': {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                  'retrievers': {r.name: {'index_sha256': file_digest(r.index_path),
                                'summaries_sha256': getattr(r, 'summaries_sha256', None),
                                'depth': getattr(r, 'depth', None), 'max_nodes': getattr(r, 'max_nodes', None),
                                'max_seeds': getattr(r, 'max_seeds', None)} for r in retrievers}}
    config_target = output / 'run_config.json'
    if config_target.exists() and json.loads(config_target.read_text()) != run_config:
        raise ValueError('Run configuration changed; use a separate output directory')
    write_json(config_target, run_config)
    for retriever in retrievers:
        path = output / (retriever.name + '.preliminary.jsonl')
        previous = read_jsonl(path) if path.exists() else []
        if len({r['question_id'] for r in previous}) != len(previous):
            raise ValueError('Duplicate saved attempts')
        done = {r['question_id'] for r in previous}
        if done - qids:
            raise ValueError('Saved attempts contain unexpected questions')
        for question in questions:
            if question['question_id'] in done:
                continue
            row = {'question_id': question['question_id'], 'project_type': question['project_type'],
                   'retriever': retriever.name, 'answer': None, 'document_ids': [], 'status': 'failed',
                   'context': '', 'chunk_ids': [], 'source_document_ids': [], 'final_source_document_ids': []}
            try:
                trace = retriever.retrieve_with_trace(question['question'], k=config['retrieval_k'])
                row['retrieval'] = trace
                packed = pack_context(trace['evidence'], adapter.count_tokens, config['context_token_budget'])
                row.update(packed)
                row['document_ids'] = packed['source_document_ids']
                row['final_source_document_ids'] = packed['source_document_ids']
                if not packed['context']:
                    raise ValueError('No source evidence fits the shared context budget')
                generated = adapter.generate(question['question'], packed['context'])
                if not isinstance(generated.get('answer'), str) or not generated['answer'].strip():
                    raise ValueError('Shared generator returned no answer')
                row.update(answer=generated['answer'], generation_metrics=generated.get('metrics', {}), status='success')
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            append_jsonl(path, row)
    report = {'question_ids': sorted(qids), 'attempts': {}, 'run_config_sha256': digest(run_config)}
    for retriever in retrievers:
        rows = read_jsonl(output / (retriever.name + '.preliminary.jsonl'))
        report['attempts'][retriever.name] = dict(Counter(r['status'] for r in rows))
    write_json(output / 'run_status.json', report)
    return report
