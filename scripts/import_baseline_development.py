"""Import only verified baseline development answers for independent grading."""
import argparse
from pathlib import Path
import subprocess
import sys

from prepare_preliminary import ROOT, read_json, rows, sha, write_json, write_rows
from preliminary_grading import export


def main(config_path):
    config = read_json(config_path)
    package = (ROOT / config['baseline_package_root']).resolve()
    graph = (ROOT / config['package_root']).resolve()
    output = (ROOT / config['output_root']).resolve() / 'baseline_development'
    if output.exists():
        raise FileExistsError('Import exists; preserve reviews and choose a different output_root in config')
    subprocess.run([sys.executable, '-X', 'utf8', str(package / 'verify.py'), str(package)], check=True)
    comparison = read_json(package / 'configuration/comparison.json')
    if comparison['shared_chunks_sha256'] != sha(graph / 'corpus/chunks/chunks.jsonl') or comparison['frozen_source_manifest_sha256'] != sha(graph / 'corpus/corpus_manifest.jsonl'):
        raise ValueError('Baseline and graph corpora differ')
    expected = {r['question_id']: r for r in rows(ROOT / config['output_root'] / 'development.labels.jsonl')}
    delivered = rows(package / 'evaluation_labels/development.labels.jsonl')
    if len(delivered) != len(expected) or {r['question_id'] for r in delivered} != expected.keys():
        raise ValueError('Development IDs differ')
    for q in delivered:
        for key in ('question', 'project_type', 'gold_answer', 'answer_facts', 'expected_doc_ids'):
            if q[key] != expected[q['question_id']][key]:
                raise ValueError(f'Development reference mismatch: {q["question_id"]}/{key}')
    run_hash = sha(package / 'results/development/run_config.json')
    imported = []
    for system in ('bm25', 'vector'):
        for row in rows(package / f'results/development/{system}.predictions.jsonl'):
            imported.append({**row, 'run_id': f'baseline-delivery-development-{run_hash[:12]}',
                             'configuration_sha256': run_hash})
    output.mkdir(parents=True)
    write_rows(output / 'predictions.jsonl', imported)
    write_json(output / 'import_provenance.json', {
        'source_package': config['baseline_package_root'],
        'package_manifest_sha256': sha(package / 'PACKAGE.json'),
        'original_run_config_sha256': run_hash,
        'source_prediction_hashes': {s: sha(package / f'results/development/{s}.predictions.jsonl') for s in ('bm25', 'vector')},
        'transformation': 'Preserved all fields; added run_id and exact run_config file SHA256 for grading schema',
        'split': 'original development only; preliminary evaluation answers excluded'})
    export(config_path, output / 'predictions.jsonl', output / 'reviews')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/preliminary.json')
    main(parser.parse_args().config.resolve())
