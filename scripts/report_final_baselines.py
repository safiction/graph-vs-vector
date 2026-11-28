"""Preflight or audit the final 400 baseline attempts without running inference."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_rag.common import index_root, read_json, write_json
from baseline_rag.final import audit, verify_inputs
from baseline_rag.generation import SharedGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/baseline_final_200.json'))
    parser.add_argument('--preflight', action='store_true', help='Only verify inputs/indexes; no server, GPU or output writes')
    parser.add_argument('--output', type=Path, default=Path('reports/baselines/final_200.json'))
    args = parser.parse_args()
    config = read_json(args.config)
    try:
        if args.preflight:
            verify_inputs(config)
            print('Preflight passed: original settings, 200 question IDs and existing index checksums verified. '
                  'This is not a team readiness decision or an inference test.')
            return
        # Tokenizer-only adapter: no server contact, model loading or generation.
        from transformers import AutoTokenizer
        tokenizer = SharedGenerator.__new__(SharedGenerator)
        tokenizer.settings = config['generation_settings']
        tokenizer.tokenizer = AutoTokenizer.from_pretrained(tokenizer.settings['model'],
            revision=tokenizer.settings['revision'], cache_dir=tokenizer.settings['cache_dir'],
            trust_remote_code=False, local_files_only=True)
        csv_path = args.output.with_suffix('.per_question.csv')
        for target in (args.output, csv_path):
            for protected in (Path(config['output']), index_root(config), Path(config['bundle']),
                              Path(config['reference_experiment']).parent):
                if target.resolve() == protected.resolve() or protected.resolve() in target.resolve().parents:
                    raise ValueError('Audit output must be outside saved experiment/input directories')
        report, measurements = audit(config, tokenizer)
        write_json(args.output, report)
        if measurements:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(measurements[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(measurements)
        print(json.dumps({'report': str(args.output), 'validated_attempts': report['validated_attempts'],
                          'all_attempts_saved': report['all_attempts_saved'],
                          'all_answers_successful': report['all_answers_successful'],
                          'audit_problems': report['audit_problems']}, indent=2))
        if not report['all_attempts_saved']:
            raise SystemExit(1)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        parser.exit(1, f'Final baseline audit failed: {exc}\n')


if __name__ == '__main__':
    main()
