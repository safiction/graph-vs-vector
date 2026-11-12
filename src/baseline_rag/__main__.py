"""CLI for the frozen 15k BM25/vector baselines and shared generation."""
import argparse
import json
from pathlib import Path
from .common import environment, output_lock, read_json, write_json
from .corpus import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/baseline_15000.json'))
    subs = parser.add_subparsers(dest='command', required=True)
    for command in ['prepare', 'build', 'run', 'all', 'status', 'query']:
        sub = subs.add_parser(command)
        if command in {'build', 'all'}:
            sub.add_argument('--indexes', nargs='+', choices=['bm25', 'vector'], default=['bm25', 'vector'])
        if command in {'run', 'all'}:
            sub.add_argument('--retrievers', nargs='+', choices=['bm25', 'vector', 'graph-local', 'graph-global'], default=['bm25', 'vector'])
            sub.add_argument('--splits', nargs='+', choices=['development', 'preliminary', 'fold_a', 'fold_b'], default=['development', 'preliminary'])
            sub.add_argument('--retry-errors', action='store_true')
        if command == 'query':
            sub.add_argument('--retriever', choices=['bm25', 'vector', 'graph-local', 'graph-global'], required=True)
            sub.add_argument('--question', required=True)
            sub.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    config = read_json(args.config)
    root = Path(config['output'])
    if args.command == 'status':
        files = [root / 'corpus_audit.json', root / 'indexes/bm25/metadata.json',
                 root / 'indexes/vector/progress.json', root / 'indexes/vector/metadata.json']
        files += sorted((root / 'runs').glob('*/summary.json'))
        print(json.dumps({str(p): read_json(p) for p in files if p.exists()}, indent=2))
        return
    with output_lock(root):
        corpus, frozen, audit = prepare(config)
        if args.command == 'prepare':
            result = audit
        elif args.command == 'query':
            from .retrieval import load_retriever
            retriever = load_retriever(args.retriever, config, corpus.identity)
            try:
                result = retriever.retrieve_with_trace(args.question, args.k)
            finally:
                if hasattr(retriever, 'close'):
                    retriever.close()
        else:
            result = {'corpus': audit, 'environment': environment()}
            if args.command in {'build', 'all'}:
                from .retrieval import build_bm25, build_vector
                # Vector rows use the common SQLite chunk store.
                result['bm25'] = build_bm25(corpus, root / 'indexes/bm25')
                if 'vector' in args.indexes:
                    result['vector'] = build_vector(corpus, root / 'indexes/vector', config['embedding'])
                write_json(root / 'indexing_report.json', result)
            if args.command in {'run', 'all'}:
                from .run import run
                previous = read_json(root / 'run_report.json') if (root / 'run_report.json').exists() else {}
                result['runs'] = previous.get('runs', {})
                for split in args.splits:
                    result['runs'][split] = run(config, corpus, frozen, split, args.retrievers, args.retry_errors)
                write_json(root / 'run_report.json', result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if 'runs' in result and any(not r['all_answers_successful'] for r in result['runs'].values()):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
