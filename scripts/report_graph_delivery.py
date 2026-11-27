"""Consolidate saved graph delivery artifacts with stdlib only; never run inference.

Checkpoint contents and the large index are streamed, not collected into lists.
Token totals are observed usage, including retries; missing usage stays explicit.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


SYSTEMS = ('graph-local', 'graph-global')
SPLITS = ('development', 'preliminary', 'fold_a', 'fold_b')
REQUIRED_SPLITS = ('development', 'preliminary')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identity(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def _number(value):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and value >= 0, 'Invalid nonnegative cost/count')
    return value


class _JSONStream:
    """Decode one index entry at a time, including minified JSON."""
    def __init__(self, stream):
        self.stream, self.buffer, self.eof = stream, '', False
        self.decoder = json.JSONDecoder()

    def fill(self):
        block = self.stream.read(65536)
        self.buffer += block
        self.eof = not block

    def peek(self):
        self.buffer = self.buffer.lstrip()
        while not self.buffer and not self.eof:
            self.fill()
            self.buffer = self.buffer.lstrip()
        return self.buffer[:1]

    def take(self, expected):
        _require(self.peek() == expected, f'Invalid index JSON: expected {expected!r}')
        self.buffer = self.buffer[1:]

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError('Invalid or truncated index JSON') from None
                self.fill()

    def entries(self, opening, closing):
        self.take(opening)
        while self.peek() != closing:
            key = None
            if opening == '{':
                key = self.value()
                self.take(':')
            yield key, self.value()
            if self.peek() == closing:
                break
            self.take(',')
            _require(self.peek() != closing, 'Trailing comma in index JSON')
        self.take(closing)


def _index_entries(path):
    with Path(path).open(encoding='utf-8') as stream:
        reader = _JSONStream(stream)
        reader.take('{')
        seen = set()
        while reader.peek() != '}':
            section = reader.value()
            _require(section not in seen, 'Duplicate index section')
            seen.add(section)
            reader.take(':')
            if section == 'metadata':
                yield section, None, reader.value()
            else:
                _require(section in {'nodes', 'edges', 'chunks', 'communities'}, 'Unknown index section')
                brackets = ('{', '}') if section in {'nodes', 'chunks'} else ('[', ']')
                for key, row in reader.entries(*brackets):
                    yield section, key, row
            if reader.peek() == '}':
                break
            reader.take(',')
            _require(reader.peek() != '}', 'Trailing comma in index JSON')
        reader.take('}')
        _require(not reader.peek() and seen == {'metadata', 'nodes', 'edges', 'chunks', 'communities'},
                 'Incomplete index or trailing data')


def _new_cost():
    return dict.fromkeys(('records', 'attempts', 'retry_attempts', 'recovery_attempts', 'model_calls',
                          'prompt_tokens', 'completion_tokens', 'attempts_without_token_metrics',
                          'sum_attempt_wall_seconds', 'sum_request_wall_seconds',
                          'attempts_without_request_wall'), 0)


def _add_attempts(cost, row):
    attempts = row['attempts']
    _require(isinstance(attempts, list) and attempts, 'Missing checkpoint attempts')
    cost['records'] += 1
    cost['retry_attempts'] += len(attempts) - 1
    for attempt in attempts:
        metrics = attempt.get('metrics') or {}
        cost['attempts'] += 1
        cost['model_calls'] += metrics.get('done_reason') != 'extractive'
        cost['recovery_attempts'] += bool(attempt.get('recovery_manifest') or attempt.get('recovery_profile'))
        cost['attempts_without_token_metrics'] += any(metrics.get(k) is None for k in ('prompt_eval_count', 'eval_count'))
        cost['prompt_tokens'] += _number(metrics.get('prompt_eval_count') or 0)
        cost['completion_tokens'] += _number(metrics.get('eval_count') or 0)
        cost['sum_attempt_wall_seconds'] += _number(attempt['attempt_wall_seconds'])
        cost['attempts_without_request_wall'] += metrics.get('wall_seconds') is None
        cost['sum_request_wall_seconds'] += _number(metrics.get('wall_seconds') or 0)


def _sessions(root, count_key, cost, interrupted=False):
    result = {'count': 0, 'attempted_records': 0, 'saved_session_wall_seconds': 0,
              'saved_session_request_seconds': 0, 'artifacts': []}
    for path in sorted((Path(root) / 'sessions').glob('*.json')):
        row = _read(path)
        result['count'] += 1
        result['artifacts'].append(_identity(path))
        result['attempted_records'] += _number(row[count_key])
        result['saved_session_wall_seconds'] += _number(row['elapsed_wall_seconds'])
        result['saved_session_request_seconds'] += _number(row['sum_request_seconds'])
    result['matches_current_attempt_totals'] = (
        result['count'] > 0 and result['attempted_records'] == cost['attempts']
        and math.isclose(result['saved_session_request_seconds'], cost['sum_attempt_wall_seconds'], abs_tol=1e-5))
    result['historical_wall_complete'] = False if interrupted else None
    result['limitation'] = (
        'Extraction was interrupted; saved sessions may cover fewer chunks than current records and omit recovery. '
        'Full historical wall time is unavailable.' if interrupted else
        'Session counters have no attempt IDs; matching totals are a consistency check, not proof of full history.')
    return result


def _questions(output, frozen, selection):
    questions, identities = {}, {}
    for split, expected in zip(SPLITS, (12, 40, 100, 100)):
        path = output / 'questions' / f'{split}.queries.jsonl'
        rows = {}
        for row in _jsonl(path):
            qid = row['question_id']
            _require(set(row) == {'question_id', 'question'} and re.fullmatch(r'[\w-]+', qid)
                     and qid not in rows, 'Invalid/duplicate query ID')
            rows[qid] = row['question']
        _require(len(rows) == expected, f'Wrong {split} question count')
        questions[split], identities[split] = rows, _identity(path)
        recorded = selection.get('input_sha256', {}).get(path.name)
        _require(recorded is None or recorded == identities[split]['sha256'], 'Frozen question hash changed')
    dev, pre, a, b = (set(questions[s]) for s in SPLITS)
    _require(dev == set(frozen['development_ids']) == set(selection['development_ids'])
             and pre == set(frozen['preliminary_question_ids']) == set(selection['preliminary_question_ids'])
             and not (dev & (a | b) or a & b) and pre <= a | b,
             'Question selection changed or development/evaluation overlap')
    _require(all(questions['preliminary'][qid] == {**questions['fold_a'], **questions['fold_b']}[qid] for qid in pre),
             'Preliminary query text differs from its core question')
    return questions, identities


def _run_report(config, frozen, split, questions, query_identity, chunks, artifacts):
    root = Path(config['output']) / 'runs' / split
    expected = len(questions) * len(SYSTEMS)
    if not root.exists():
        return {'status': 'pending', 'present': False, 'expected_attempts': expected, 'saved_attempts': 0,
                'questions': query_identity}
    run_config, summary = _read(root / 'run_config.json'), _read(root / 'summary.json')
    _require(run_config['experiment'] == frozen and set(run_config['retrievers']) == set(SYSTEMS)
             and len(run_config['retrievers']) == 2 and run_config['questions_sha256'] == query_identity['sha256']
             and run_config['graph_depth'] == config.get('graph_depth', 1), 'Run configuration differs from frozen inputs')
    for system in SYSTEMS:
        expected_artifacts = {'index'} | ({'summary_config', 'summary_status'} if system == 'graph-global' else set())
        saved_artifacts = run_config['artifacts'][system]
        identities = {Path(p).name: h for p, h in saved_artifacts.items()}
        expected_hashes = {Path(artifacts[k]['path']).name: artifacts[k]['sha256'] for k in expected_artifacts}
        # Kaggle and collected local runs mount identical immutable artifacts at different paths.
        _require(len(identities) == len(saved_artifacts) and identities == expected_hashes,
                 'Run artifact hashes changed')
    identity = summary.get('generator_identity') or {}
    settings = frozen['generation_settings']
    _require(not identity or identity.get('settings_sha256') == _digest(settings), 'Generator settings identity changed')
    for key in ('model', 'revision', 'backend', 'server_version', 'server_model_digest'):
        _require(not identity or key not in settings or identity.get(key) == settings[key], 'Generator model identity changed')
    totals = dict.fromkeys(('attempts', 'success', 'failed', 'prompt_tokens', 'completion_tokens',
                           'attempts_without_token_metrics', 'sum_total_query_seconds',
                           'attempts_without_total_query_seconds', 'sum_retrieval_seconds', 'sum_generation_seconds'), 0)
    systems, run_artifacts, fingerprint = {}, {}, hashlib.sha256()
    max_context, checked_links = 0, 0
    _require(set(summary['systems']) == set(SYSTEMS) and summary['split'] == split, 'Wrong saved systems/split')
    _require(not (root / 'attempts').exists() or
             {p.name for p in (root / 'attempts').iterdir() if p.is_dir()} <= set(SYSTEMS), 'Unexpected attempt system')
    for system in SYSTEMS:
        cost = dict.fromkeys(totals, 0)
        paths = {p.stem: p for p in (root / 'attempts' / system).glob('*.json')}
        _require(set(paths) <= set(questions), 'Unknown saved attempt ID')
        exported = set()
        successful_usage = Counter()
        predictions_path = root / f'{system}.predictions.jsonl'
        for row in _jsonl(predictions_path):
            qid = row['question_id']
            _require(qid in paths and qid not in exported and row == _read(paths[qid]),
                     'Predictions differ from current saved attempts')
            exported.add(qid)
            _require(row['retriever'] == system and row['question'] == questions[qid]
                     and row['status'] in {'success', 'failed'}, 'Invalid saved attempt identity/status')
            success = row['status'] == 'success'
            _require(success or bool(row.get('error')), 'Failed answer has no explicit reason')
            if success:
                _require(identity and isinstance(row.get('answer'), str) and row['answer'].strip()
                         and row.get('context') and row.get('evidence') and row.get('context_tokens'),
                         'Successful answer lacks content/identity/token count')
                fields = ('retrieval_seconds', 'packing_seconds', 'generation_seconds', 'total_query_seconds')
                _require(all(row.get(field) is not None for field in fields)
                         and math.isclose(_number(row['total_query_seconds']),
                             sum(_number(row[field]) for field in fields[:3]), abs_tol=1e-6),
                         'Successful answer has missing/inconsistent latency')
            context = _number(row.get('context_tokens', 0))
            _require(context <= frozen['context_token_budget'], 'Context exceeds shared budget')
            max_context = max(max_context, context)
            links = []
            for evidence in row.get('evidence', []):
                _require(evidence.get('text') and evidence['text'] in row['context'] and evidence.get('source_links'),
                         'Packed evidence lacks text/source links')
                for link in evidence['source_links']:
                    chunk = chunks.get(link['chunk_id'])
                    _require(chunk and all(link.get(k) == chunk[k] for k in
                             ('source_document_id', 'source_path', 'source_uri', 'source_sha256')),
                             'Answer provenance outside frozen source artifacts')
                    links.append(link)
            if success or links:
                _require(links and set(row['chunk_ids']) == {link['chunk_id'] for link in links}
                         and set(row['source_document_ids']) == set(row['final_source_document_ids'])
                         == {link['source_document_id'] for link in links}, 'Packed source IDs differ from links')
            checked_links += len(links)
            metrics = row.get('generation_metrics') or {}
            for key, maximum in (('local_prompt_tokens', settings['max_input_tokens']),
                                 ('completion_tokens', settings['max_new_tokens'])):
                _require(metrics.get(key) is None or _number(metrics[key]) <= maximum, 'Generation token budget exceeded')
            cost['attempts'] += 1
            cost['success' if success else 'failed'] += 1
            cost['attempts_without_token_metrics'] += any(metrics.get(k) is None for k in ('prompt_tokens', 'completion_tokens'))
            for key in ('prompt_tokens', 'completion_tokens'):
                value = _number(metrics.get(key) or 0)
                cost[key] += value
                if success:
                    successful_usage[key] += value
            for field in ('total_query_seconds', 'retrieval_seconds', 'generation_seconds'):
                cost['sum_' + field] += _number(row.get(field) or 0)
            cost['attempts_without_total_query_seconds'] += row.get('total_query_seconds') is None
            fingerprint.update(_canonical(row) + b'\n')
        _require(exported == set(paths), 'Prediction export missing current attempts')
        saved = summary['systems'][system]
        _require(all(saved[k] == cost[k] for k in ('attempts', 'success', 'failed'))
                 and all(saved[k] == successful_usage[k] for k in ('prompt_tokens', 'completion_tokens')),
                 'Runner summary differs from current attempts')
        systems[system] = cost
        run_artifacts[system + '_predictions'] = _identity(predictions_path)
        for key, value in cost.items():
            totals[key] += value
    all_saved = totals['attempts'] == expected
    _require(summary['expected_attempts'] == expected and summary['saved_attempts'] == totals['attempts']
             and summary['all_attempts_saved'] == all_saved
             and summary['all_answers_successful'] == (all_saved and totals['failed'] == 0),
             'Run summary completion flags/counts differ')
    run_artifacts['config'], run_artifacts['summary'] = _identity(root / 'run_config.json'), _identity(root / 'summary.json')
    return {'status': 'all_attempts_saved' if all_saved else 'incomplete', 'present': True,
            'expected_attempts': expected, 'saved_attempts': totals['attempts'], 'cost': totals, 'systems': systems,
            'all_answers_successful': all_saved and totals['failed'] == 0, 'artifacts': run_artifacts,
            'run_config_sha256': run_artifacts['config']['sha256'], 'run_config': run_config, 'summary': summary,
            'generator_identity': identity, 'current_attempts_sha256_export_order': fingerprint.hexdigest(),
            'last_invocation_wall_seconds': _number(summary['invocation_elapsed_seconds']),
            'historical_run_wall_complete': False,
            'budget_validation': {'recorded_counts_within_limits': True, 'tokenizer_recounted': False,
                                  'max_observed_context_tokens': max_context,
                                  'context_token_cap': frozen['context_token_budget'],
                                  'max_input_tokens': settings['max_input_tokens'], 'max_new_tokens': settings['max_new_tokens']},
            'provenance_validation': {'passed': True, 'source_links_checked': checked_links},
            'limitations': ['Current runner overwrites failed retries and last-invocation wall; unsaved prior calls cannot be reconstructed.',
                            'Saved token counts are validated against budgets, not independently retokenized.',
                            'Structural provenance does not establish answer correctness.']}


def build_report(config_path, extraction_path):
    """Validate artifacts and return observed costs, without writing a report.

    Relative config locations follow the shared runner: resolved from the cwd.
    """
    config, extraction = _read(config_path), Path(extraction_path).resolve()
    bundle, output = Path(config['bundle']), Path(config['output'])
    index_path, summaries = Path(config['graph_index']), Path(config['graph_summaries'])
    paths = {'config': Path(config_path), 'bundle': bundle / 'bundle.json',
             'chunk_config': bundle / 'chunks/chunk_config.json', 'chunks': bundle / 'chunks/chunks.jsonl',
             'source_manifest': bundle / 'corpus_manifest.jsonl', 'experiment': output / 'experiment.json',
             'corpus_audit': output / 'corpus_audit.json', 'selection': output / 'questions/selection.json',
             'extraction_config': extraction / 'extraction_config.json',
             'extraction_status': extraction / 'extraction_status.json', 'index': index_path,
             'index_cost': index_path.with_suffix('.cost.json'), 'source_audit': extraction / 'source_audit.json',
             'summary_config': summaries / 'summary_config.json', 'summary_status': summaries / 'summary_status.json'}
    artifacts = {key: _identity(path) for key, path in paths.items()}
    data = {key: _read(path) for key, path in paths.items() if key not in {'chunks', 'source_manifest', 'index'}}
    sha = {key: value['sha256'] for key, value in artifacts.items()}
    frozen, corpus = data['experiment'], data['experiment']['corpus']
    for key in ('retrieval_k', 'context_token_budget', 'embedding', 'generation_settings'):
        _require(frozen[key] == config[key], f'Frozen {key} changed')
    for key, artifact in (('chunks_sha256', 'chunks'), ('source_manifest_sha256', 'source_manifest')):
        _require(corpus[key] == data['chunk_config'][key] == data['extraction_config'][key] == sha[artifact],
                 f'Corpus/extraction {key} changed')
    _require(data['extraction_config']['sample_sha256'] is None, 'Extraction is a sample, not the frozen corpus')
    sources = {}
    for source in _jsonl(paths['source_manifest']):
        did = source['source_document_id']
        _require(did not in sources, 'Duplicate source document ID')
        sources[did] = source
    chunks = {}
    for chunk in _jsonl(paths['chunks']):
        cid = chunk['chunk_id']
        _require(cid not in chunks and chunk['source_document_id'] in sources and chunk['text'].strip(),
                 'Duplicate/invalid corpus chunk')
        chunks[cid] = chunk
    _require(len(sources) == corpus['document_count'] == data['bundle']['document_count']
             and len(chunks) == corpus['chunk_count'] == data['bundle']['chunk_count']
             and {c['source_document_id'] for c in chunks.values()} == set(sources)
             and _digest(list(chunks)) == corpus['chunk_ids_sha256'], 'Frozen corpus count/identity mismatch')
    _require(all(data['corpus_audit'].get(k) == v for k, v in corpus.items())
             and data['corpus_audit']['chunks_verified'], 'Corpus audit changed')
    exstatus = data['extraction_status']
    _require(exstatus['complete'] is True and exstatus['selected_chunks'] == len(chunks)
             and exstatus['completed_chunks'] == len(chunks) and exstatus['error_chunks'] == exstatus['pending_chunks'] == 0
             and exstatus['extraction_config_sha256'] == sha['extraction_config'], 'Extraction incomplete/changed')
    extraction_cost, record_hash, seen_chunks = _new_cost(), hashlib.sha256(b'['), set()
    recovery_paths = set()
    extraction_counts = Counter()
    for path in sorted((extraction / 'records').glob('*.json')):
        row = _read(path)
        cid = row['chunk_id']
        _require(cid in chunks and cid not in seen_chunks and path.name == _digest(cid) + '.json',
                 'Unexpected/duplicate extraction record')
        seen_chunks.add(cid)
        _require(row['status'] == 'ok' and row['source_document_id'] == chunks[cid]['source_document_id']
                 and row['text_sha256'] == _digest(chunks[cid]['text']), 'Extraction error/source mismatch')
        if extraction_cost['records']:
            record_hash.update(b', ')
        record_hash.update(_canonical(row))
        _add_attempts(extraction_cost, row)
        for key in ('entities', 'relations', 'rejected'):
            extraction_counts[key] += len(row[key])
        for attempt in row['attempts']:
            if attempt.get('recovery_manifest'):
                recovery = (extraction / attempt['recovery_manifest']).resolve()
                _require(recovery.is_relative_to(extraction / 'recovery'), 'Recovery manifest escapes extraction')
                recovery_paths.add(recovery)
    record_hash.update(b']')
    _require(seen_chunks == set(chunks), 'Missing extraction records')
    sstatus, sconfig = data['summary_status'], data['summary_config']
    _require(sstatus['complete'] is True and sstatus['errors'] == 0
             and sstatus['index_sha256'] == sconfig['index_sha256'] == sha['index']
             and sstatus['summary_config_sha256'] == sha['summary_config'], 'Summaries incomplete/changed')
    summary_cost, summary_methods, summary_hash = _new_cost(), Counter(), hashlib.sha256()
    summary_claims, counts, seen_communities, indexed = 0, Counter(), set(), set()
    metadata = None
    for section, key, row in _index_entries(index_path):
        if section == 'metadata':
            metadata = row
        else:
            counts[section] += 1
        if section == 'chunks':
            _require(key in chunks and key not in indexed and row['chunk_id'] == key
                     and row['source_document_id'] == chunks[key]['source_document_id']
                     and row['text'] == chunks[key]['text'], 'Index chunk identity mismatch')
            source = sources[row['source_document_id']]
            _require(row['source_path'] == source['path'] and row['source_sha256'] == source['sha256']
                     and row['source_uri'], 'Index source link mismatch')
            chunks[key].update({k: row[k] for k in ('source_path', 'source_sha256', 'source_uri')})
            indexed.add(key)
        if section != 'communities':
            continue
        cid = row['id']
        _require(re.fullmatch(r'[\w-]+', cid) and cid not in seen_communities, 'Unsafe/duplicate community ID')
        seen_communities.add(cid)
        path = summaries / 'records' / f'{cid}.json'
        saved = _read(path)
        _require(saved['community_id'] == cid and saved['status'] == 'ok' and saved.get('claims')
                 and 0 < len(saved['claims']) <= 4 and saved.get('title'), 'Missing/failed summary')
        _require(saved['supplied_evidence'] == row['evidence'][:sconfig['max_evidence_items']],
                 'Summary supplied evidence changed')
        for claim in saved['claims']:
            chunk = chunks.get(claim['chunk_id'])
            _require(chunk and claim['text'].strip() and claim['quote'] and claim['quote'] in chunk['text']
                     and any(e['chunk_id'] == claim['chunk_id'] and claim['quote'] in e['quote'] for e in row['evidence']),
                     'Summary claim provenance mismatch')
            summary_claims += 1
        _add_attempts(summary_cost, saved)
        summary_methods[saved['method']] += 1
        summary_hash.update(_canonical(saved) + b'\n')
    _require(metadata and metadata['corpus_scope'] == 'all_supplied_chunks'
             and metadata['chunks_sha256'] == sha['chunks']
             and metadata['source_manifest_sha256'] == sha['source_manifest']
             and metadata['extraction_config_sha256'] == sha['extraction_config']
             and metadata['checkpoint_sha256'] == record_hash.hexdigest()
             and metadata['total_corpus_chunks'] == metadata['indexed_chunks'] == len(chunks)
             and indexed == set(chunks), 'Index metadata/checkpoint set mismatch')
    _require(counts['nodes'] == data['index_cost']['nodes'] and counts['edges'] == data['index_cost']['relations']
             and counts['communities'] == data['index_cost']['communities'] == sstatus['communities'] == sstatus['completed']
             and summary_methods['extractive_fallback'] == sstatus['extractive_fallbacks']
             and sum(1 for _ in (summaries / 'records').glob('*.json')) == counts['communities'],
             'Index/summary counts differ')
    audit = data['source_audit']
    _require(audit['passed'] is True and not audit['errors'] and audit['index_sha256'] == sha['index']
             and audit['counts']['source_documents'] == len(sources) and audit['counts']['chunks'] == len(chunks)
             and audit['counts']['summary_claims'] == summary_claims, 'Source audit failed/changed')
    recovery_manifests = []
    for path in sorted(recovery_paths):
        manifest = _read(path)
        _require(manifest['primary_extraction_config_sha256'] == sha['extraction_config'], 'Recovery config changed')
        recovery_manifests.append({'artifact': _identity(path), 'data': manifest})
    questions, query_identities = _questions(output, frozen, data['selection'])
    runs = {split: _run_report(config, frozen, split, questions[split], query_identities[split], chunks, artifacts)
            for split in SPLITS}
    for split, run in runs.items():
        run['required_for_current_scope'] = split in REQUIRED_SPLITS
        if split not in REQUIRED_SPLITS and not run['present']:
            run['status'] = 'deferred_not_requested'
    artifacts['report_script'] = _identity(__file__)
    return {'schema_version': 1, 'graph_artifacts_validated': True,
            'scope': 'paired_12_development_40_preliminary_only',
            'all_graph_attempts_saved': all(runs[split]['status'] == 'all_attempts_saved' for split in REQUIRED_SPLITS),
            'artifacts': artifacts, 'corpus': corpus, 'corpus_selection': data['bundle'],
            'runs': runs, 'correctness': 'pending_external_grading', 'routing': 'deferred_until_paired_correctness',
            'development_grading_handoff': {'status': 'pending_external_grading',
                'prediction_exports': [runs['development']['artifacts'][system + '_predictions'] for system in SYSTEMS]
                                      if runs['development']['present'] else []},
            'external_blockers': ['Independent grading of 160 preliminary and 24 graph development answers; resolve disagreements.',
                                  'Team-agreed aggregate handling of failed and N/A attempts.',
                                  'Actual Kaggle T4 run and measured receipts; 5080 artifacts are not T4 evidence.'],
            'preprocessing': {'cost': {key: extraction_cost[key] + summary_cost[key] for key in extraction_cost},
                              'index_wall_seconds': _number(data['index_cost']['wall_seconds']),
                              'historical_wall_seconds': None},
            'T4_validation': 'unverified_no_actual_T4_measurements',
            'gpu_hours': None, 'electricity_cost': None,
            'limitations': ['No GPU-hour or electricity inference from client latency, memory estimates or hardware labels.',
                            'Preliminary/core question duplicates are not independent observations; split costs remain separate.',
                            'Corpus includes question-conditioned sources; subset results do not establish benchmark non-contamination.',
                            'Tokens with missing usage are unknown, not free. Request/attempt sums are not concurrent wall time.'],
            'source_audit': audit, 'index_cost': data['index_cost'], 'index_metadata': metadata,
            'index_counts': dict(counts), 'extraction_status': exstatus, 'summary_status': sstatus,
            'extraction': {'cost': extraction_cost, 'counts': dict(extraction_counts),
                           'current_records_sha256': record_hash.hexdigest(),
                           'model': data['extraction_config']['model'], 'recovery_manifests': recovery_manifests,
                           'sessions': _sessions(extraction, 'attempted_chunks', extraction_cost, interrupted=True)},
            'summaries': {'cost': summary_cost, 'methods': dict(summary_methods), 'claims': summary_claims,
                          'current_records_sha256_index_order': summary_hash.hexdigest(), 'model': sconfig['model'],
                          'sessions': _sessions(summaries, 'attempted_communities', summary_cost)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--extraction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args.config, args.extraction)
        config = _read(args.config)
        target = args.output.resolve()
        input_roots = [args.extraction, Path(config['bundle']), Path(config['output']), Path(config['graph_summaries'])]
        _require(str(target) not in {a['path'] for a in report['artifacts'].values()}
                 and not any(target.is_relative_to(root.resolve()) for root in input_roots),
                 'Report output would overwrite an input artifact')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f'Saved validated graph artifact report to {args.output}; correctness pending external grading; routing deferred.')


if __name__ == '__main__':
    main()
