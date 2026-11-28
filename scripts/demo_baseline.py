"""Display a saved baseline answer and its actual evidence; no model or GPU needed."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_rag.common import file_digest, read_jsonl


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path,
                        default=Path('data/baselines/corpus_15000/runs/development/bm25.predictions.jsonl'))
    parser.add_argument('--question-id', help='Default: first successful saved attempt')
    parser.add_argument('--json', action='store_true', help='Print the original saved record and file checksum')
    args = parser.parse_args()
    try:
        rows = read_jsonl(args.predictions)
        matches = [r for r in rows if r['question_id'] == args.question_id] if args.question_id else [
            r for r in rows if r['status'] == 'success'][:1]
        if len(matches) != 1:
            parser.error('Choose one existing question ID, or supply a file containing a successful attempt')
        row = matches[0]
        if args.json:
            print(json.dumps({'predictions_sha256': file_digest(args.predictions), 'record': row},
                             indent=2, ensure_ascii=False))
            return
        print(f'Saved result: {args.predictions}')
        print(f'SHA256: {file_digest(args.predictions)}')
        print(f'{row["retriever"]} / {row["question_id"]} / {row["status"]}')
        print(f'\nQuestion:\n{row["question"]}')
        if row['status'] != 'success':
            print(f'\nError:\n{row.get("error", "No error recorded")}')
            return
        print(f'\nAnswer:\n{row["answer"]}')
        print(f'\nActual context ({row["context_tokens"]} tokens):\n{row["context"]}')
        print('\nSupplied sources:')
        for item in row['evidence']:
            for link in item['source_links']:
                print(f'  {link["chunk_id"]} -> {link["source_document_id"]} -> {link["source_path"]}')
        metrics = row.get('generation_metrics', {})
        print(f'\nFinish reason: {metrics.get("finish_reason")}; query seconds: {row.get("total_query_seconds")}')
        print('Correctness requires independent grading; this demo displays the saved model output.')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Cannot read saved baseline result: {exc}\n')


if __name__ == '__main__':
    main()
