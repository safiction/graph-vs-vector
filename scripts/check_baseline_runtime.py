"""Synthetic local-server startup/latency probe; never uses evaluation questions."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:11435')
    parser.add_argument('--model', default='qwen2.5:3b')
    parser.add_argument('--output', type=Path, default=Path('data/baselines/corpus_15000/diagnostics/ollama_startup.json'))
    args = parser.parse_args()
    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(args.endpoint + path, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)
    report = {'purpose': 'Synthetic runtime check, not a benchmark question',
              'server': request('/api/version'), 'models': request('/api/tags'), 'attempts': []}
    for _ in range(2):
        started = time.perf_counter()
        result = request('/api/chat', {'model': args.model, 'stream': False,
            'messages': [{'role': 'system', 'content': 'Answer using the evidence and cite the source.'},
                         {'role': 'user', 'content': '[smoke-test] Alpha is a test project. What is Alpha?'}],
            'options': {'temperature': 0, 'seed': 42, 'num_predict': 64, 'num_ctx': 4480, 'repeat_penalty': 1.0}})
        report['attempts'].append({'wall_seconds': time.perf_counter() - started, 'response': result})
    report['residency'] = request('/api/ps')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
