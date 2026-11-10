"""A routing prototype fitted exclusively on externally graded development scores."""
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from .common import file_digest, read_jsonl, write_json
from .retrieve import terms


SYSTEMS = ('bm25', 'vector', 'graph-local', 'graph-global')
TYPES = ('single-fact', 'multi-hop', 'corpus-global', 'fuzzy')


def train_router(scores_path, questions_path, selection_path, output, systems=SYSTEMS):
    systems = tuple(systems)
    if not systems or len(set(systems)) != len(systems) or set(systems) - set(SYSTEMS):
        raise ValueError('Specify distinct known retrievers')
    selection = json.loads(Path(selection_path).read_text())
    development = set(selection['development_ids'])
    evaluated = {qid for fold in selection['folds'].values() for qid in fold}
    if development & evaluated:
        raise ValueError('Development/evaluation split overlaps')
    questions = {q['question_id']: q for q in read_jsonl(questions_path)}
    if set(questions) != development:
        raise ValueError('Training questions must exactly match development IDs')
    rows = read_jsonl(scores_path)
    observations = {}
    for row in rows:
        qid, system, score = row['question_id'], row['retriever'], row['score']
        if qid not in development or qid in evaluated:
            raise ValueError(f'Evaluation/unknown score prohibited in development routing: {qid}')
        if system not in systems:
            raise ValueError(f'Unexpected retriever in scores: {system}')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Scores must be finite correctness scores in [0, 1]')
        if (qid, system) in observations:
            raise ValueError('Duplicate score')
        if row.get('project_type', questions[qid]['project_type']) != questions[qid]['project_type']:
            raise ValueError('Score/question type mismatch')
        observations[qid, system] = score
    expected = {(qid, system) for qid in development for system in systems}
    if set(observations) != expected:
        raise ValueError(f'Paired development scores required: missing {len(expected - set(observations))} scores')
    if any(q['project_type'] not in TYPES for q in questions.values()):
        raise ValueError('Unknown project type')
    global_means = {s: sum(observations[qid, s] for qid in development) / len(development) for s in systems}
    choose = lambda means: min(systems, key=lambda s: (-means[s], systems.index(s)))
    fallback = choose(global_means)
    routes = {}
    centroids = defaultdict(lambda: defaultdict(float))
    for kind in TYPES:
        ids = sorted(qid for qid in development if questions[qid]['project_type'] == kind)
        means = {s: sum(observations[qid, s] for qid in ids) / len(ids) for s in systems} if ids else {}
        routes[kind] = {'retriever': choose(means) if ids else fallback, 'n': len(ids),
                        'means': means, 'fallback': not bool(ids)}
        for qid in ids:
            tokens = set(terms(questions[qid]['question']))
            norm = math.sqrt(len(tokens)) or 1
            for token in tokens:
                centroids[kind][token] += 1 / norm / len(ids)
    result = {'status': 'development_prototype_not_cross_fold_evaluation', 'systems': systems,
              'trained_question_ids': sorted(development), 'routes': routes, 'fallback_retriever': fallback,
              'global_means': global_means, 'classifier': 'development-only normalized token centroids',
              'centroids': dict(centroids), 'tie_order': systems,
              'unlearned_types': [t for t in TYPES if routes[t]['fallback']],
              'scores_sha256': file_digest(scores_path), 'questions_sha256': file_digest(questions_path),
              'selection_sha256': file_digest(selection_path)}
    write_json(output, result)
    return result


def route_question(router, question, known_type=None):
    started = time.monotonic()
    if known_type is None:
        tokens = set(terms(question))
        similarities = {}
        for kind, centroid in router['centroids'].items():
            norm = math.sqrt(sum(v*v for v in centroid.values())) * math.sqrt(len(tokens))
            similarities[kind] = sum(centroid.get(t, 0) for t in tokens) / norm if norm else 0
        predicted = min(similarities, key=lambda t: (-similarities[t], TYPES.index(t))) if any(similarities.values()) else None
    else:
        if known_type not in TYPES:
            raise ValueError('Unknown question type')
        predicted = known_type
    choice = router['routes'][predicted]['retriever'] if predicted else router['fallback_retriever']
    return {'retriever': choice, 'predicted_type': predicted, 'known_type': known_type,
            'used_fallback': predicted is None or router['routes'][predicted]['fallback'],
            'routing_seconds': time.monotonic() - started}
