"""Export existing graph development attempts for independent fact grading."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline_rag.common import file_digest, read_json, read_jsonl, write_json, write_jsonl
from preliminary_grading import export, validate_predictions


def export_development(config_path, output):
    config, output = read_json(config_path), Path(output)
    inputs = output.with_name(output.name + '.inputs')
    if output.exists() or inputs.exists():
        raise FileExistsError('Choose a new handoff directory; never overwrite reviews')
    bundle, run = Path(config['bundle']), Path(config['output'])
    if any(p.resolve() == bundle.resolve() or bundle.resolve() in p.resolve().parents for p in (output, inputs)):
        raise ValueError('Cannot write reviews inside the frozen bundle')
    selection = read_json(bundle / 'questions/splits.json')
    development = set(selection['development_ids'])
    evaluated = {qid for ids in selection['folds'].values() for qid in ids}
    if not development or development & evaluated:
        raise ValueError('Development must be nonempty and disjoint from evaluation')
    labels = read_jsonl(bundle / 'questions/development.labels.jsonl')
    if len(labels) != len(development) or {q['question_id'] for q in labels} != development:
        raise ValueError('Development labels differ from frozen selection')
    metadata = read_json(run / 'questions/metadata.json')
    labels = [{**q, 'project_type': metadata[q['question_id']]['project_type']} for q in labels]
    run_config = run / 'runs/development/run_config.json'
    identity = file_digest(run_config)
    predictions, hashes = [], {}
    for name in ('graph-local', 'graph-global'):
        path = run / f'runs/development/{name}.predictions.jsonl'
        rows = read_jsonl(path)
        if (len(rows) != len(development) or {r['question_id'] for r in rows} != development or
                any(r['retriever'] != name for r in rows)):
            raise ValueError('Graph predictions must completely pair the original development IDs')
        hashes[name] = file_digest(path)
        predictions.extend({**r, 'run_id': f'graph-development-{identity[:12]}',
                            'configuration_sha256': identity} for r in rows)
    document_ids = (bundle / 'document_ids.txt').read_text(encoding='utf-8')
    validate_predictions(predictions, labels, set(document_ids.splitlines()))
    # The existing blinded exporter selects at most ten answers per type. The frozen
    # graph development run has eight per represented type, so all successes fit.
    from collections import Counter
    counts = Counter(next(q['project_type'] for q in labels if q['question_id'] == p['question_id'])
                     for p in predictions if p['status'] == 'success')
    if any(n > 10 for n in counts.values()):
        raise ValueError('Development exceeds the existing full-review export capacity')
    (inputs / 'package/corpus').mkdir(parents=True)
    (inputs / 'package/corpus/document_ids.txt').write_text(document_ids, encoding='utf-8')
    write_jsonl(inputs / 'development.labels.jsonl', labels)
    write_jsonl(inputs / 'predictions.jsonl', predictions)
    write_json(inputs / 'config.json', {'package_root': str((inputs / 'package').resolve()),
        'output_root': str(inputs.resolve()), 'selection_manifest': str((bundle / 'questions/splits.json').resolve()),
        'coverage_exclusions': str((bundle / 'questions/coverage_exclusions.json').resolve()), 'seed': config['seed']})
    write_json(inputs / 'provenance.json', {'run_config': str(run_config), 'run_config_sha256': identity,
        'prediction_sha256': hashes, 'selection_sha256': file_digest(bundle / 'questions/splits.json'),
        'note': 'Human labels intentionally blank; failed attempts retained but not assigned invented scores.'})
    export(inputs / 'config.json', inputs / 'predictions.jsonl', output)
    return read_json(output / 'export_status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/graph_local_15000.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_development(args.config, args.output)
