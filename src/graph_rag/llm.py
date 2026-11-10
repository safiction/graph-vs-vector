"""Local Ollama client with actual token/timing accounting."""
import json
import time
import urllib.request
from urllib.parse import urlparse


class LocalModel:
    def __init__(self, model='qwen3.6:35b', endpoint='http://127.0.0.1:11434', timeout=180):
        if urlparse(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Graph jobs require a local inference endpoint')
        self.endpoint, self.model, self.timeout = endpoint.rstrip('/'), model, timeout
        tags = self.request('/api/tags')['models']
        match = next((m for m in tags if m['name'] == model), None)
        if match is None:
            raise ValueError(f'Local model is not installed: {model}')
        self.identity = {'model': model, 'digest': match['digest'],
                         'server_version': self.request('/api/version')['version']}

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.endpoint + path, data=data,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def chat(self, system, user, schema=None, max_tokens=1100):
        payload = {'model': self.model, 'messages': [{'role': 'system', 'content': system},
                   {'role': 'user', 'content': user}], 'stream': False, 'think': False,
                   'options': {'temperature': 0, 'seed': 42, 'num_predict': max_tokens}}
        if schema is not None:
            payload['format'] = schema
        started = time.monotonic()
        response = self.request('/api/chat', payload)
        metrics = {k: response.get(k) for k in (
            'prompt_eval_count', 'prompt_eval_cached_count', 'eval_count', 'total_duration',
            'load_duration', 'prompt_eval_duration', 'eval_duration', 'done_reason')}
        metrics['wall_seconds'] = time.monotonic() - started
        metrics['gpu_residency'] = [{k: m.get(k) for k in ('name', 'digest', 'size_vram', 'context_length')}
                                    for m in self.request('/api/ps').get('models', [])]
        return response.get('message', {}).get('content', ''), metrics
