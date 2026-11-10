"""Run with PYTHONPATH=src python -m graph_rag --help."""
import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    def command(name, arguments):
        sub = commands.add_parser(name)
        for arg in arguments:
            sub.add_argument('--' + arg.replace('_', '-'), required=True, type=Path)
        return sub

    def local_model(sub):
        sub.add_argument('--model', default='qwen3.6:35b')
        sub.add_argument('--endpoint', default='http://127.0.0.1:11434')

    command('prepare', ['manifest', 'dataset', 'output', 'chunk_config'])
    sub = command('sample', ['chunks', 'manifest', 'questions_manifest', 'dataset', 'output'])
    sub.add_argument('--per-source', type=int, default=2)
    sub.add_argument('--seed', type=int, default=42)
    sub = command('extract', ['chunks', 'manifest', 'output'])
    sub.add_argument('--sample', type=Path)
    sub.add_argument('--max-seconds', type=float)
    sub.add_argument('--retry-errors', action='store_true')
    local_model(sub)
    command('build', ['chunks', 'manifest', 'extraction', 'dataset', 'output'])
    sub = command('summarize', ['index', 'output'])
    sub.add_argument('--max-seconds', type=float)
    sub.add_argument('--retry-errors', action='store_true')
    local_model(sub)
    sub = command('audit', ['index', 'manifest', 'dataset', 'output'])
    sub.add_argument('--summaries', type=Path)
    sub = command('cost', ['extraction', 'sample', 'manifest', 'output'])
    sub.add_argument('--full-index', type=Path)
    sub.add_argument('--summaries', type=Path)
    sub.add_argument('--budget-hours', type=float)
    sub = command('query', ['index'])
    sub.add_argument('--system', choices=['graph-local', 'graph-global'], required=True)
    sub.add_argument('--query', required=True)
    sub.add_argument('--summaries', type=Path)
    sub.add_argument('--k', type=int, default=10)
    sub.add_argument('--depth', type=int, choices=[1, 2], default=1)
    sub.add_argument('--output', type=Path)
    sub = command('demo', ['index', 'summaries', 'questions_manifest', 'dataset', 'output'])
    sub.add_argument('--k', type=int, default=10)
    sub = command('train-router', ['scores', 'questions', 'selection', 'output'])
    sub.add_argument('--systems', nargs='+', default=['bm25', 'vector', 'graph-local', 'graph-global'])
    sub = command('route', ['router'])
    sub.add_argument('--question', required=True)
    sub.add_argument('--known-type')
    sub = command('run-preliminary', ['index', 'summaries', 'questions', 'selection', 'comparison_config', 'output'])
    sub.add_argument('--adapter', required=True, help='module:factory; factory(generation_settings) returns shared adapter')
    args = vars(parser.parse_args())
    action = args.pop('command')
    result = None
    if action == 'prepare':
        from .prepare import prepare
        result = prepare(**args)
    elif action == 'sample':
        from .extract import select_sample
        result = select_sample(**args)
    elif action in ('extract', 'summarize'):
        from .llm import LocalModel
        model = LocalModel(args.pop('model'), args.pop('endpoint'))
        if action == 'extract':
            from .extract import extract
            result = extract(**args, model=model)
        else:
            from .index import summarize
            args['index_path'] = args.pop('index')
            result = summarize(**args, model=model)
    elif action == 'build':
        from .index import build_index
        result = build_index(**args)
    elif action == 'audit':
        from .audit import audit
        args['index_path'] = args.pop('index')
        result = audit(**args)
        if not result['passed']:
            raise SystemExit(json.dumps(result, indent=2))
    elif action == 'cost':
        from .cost import measure_cost
        result = measure_cost(**args)
    elif action == 'query':
        from .retrieve import GraphLocalRetriever, GraphGlobalRetriever
        from .common import write_json
        if args['system'] == 'graph-global':
            if not args['summaries']:
                parser.error('--summaries is required for graph-global')
            retriever = GraphGlobalRetriever(args['index'], args['summaries'])
        else:
            retriever = GraphLocalRetriever(args['index'], depth=args['depth'])
        result = retriever.retrieve_with_trace(args['query'], k=args['k'])
        if args['output']:
            write_json(args['output'], result)
    elif action in ('demo', 'run-preliminary'):
        from .retrieve import GraphLocalRetriever, GraphGlobalRetriever
        local = GraphLocalRetriever(args.pop('index'))
        retrievers = [local, GraphGlobalRetriever(local.index_path, args.pop('summaries'))]
        if action == 'demo':
            from .run import retrieval_demo
            args['selection_path'] = args.pop('questions_manifest')
            result = retrieval_demo(retrievers, **args)
        else:
            from .run import run_preliminary
            module, function = args.pop('adapter').split(':', 1)
            config = json.loads(args['comparison_config'].read_text())
            adapter = getattr(importlib.import_module(module), function)(config['generation_settings'])
            result = run_preliminary(retrievers, args['questions'], args['selection'], args['comparison_config'], adapter, args['output'])
    elif action == 'train-router':
        from .routing import train_router
        result = train_router(args['scores'], args['questions'], args['selection'], args['output'], systems=args['systems'])
    elif action == 'route':
        from .routing import route_question
        result = route_question(json.loads(args['router'].read_text()), args['question'], args['known_type'])
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
