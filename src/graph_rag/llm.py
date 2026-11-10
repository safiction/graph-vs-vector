"""Local Ollama client with actual token/timing accounting."""
import json
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from .common import digest, file_digest


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


class VLLMModel:
    """Local OpenAI-compatible vLLM, with constrained JSON and pinned weights.

    The server config and weight manifest are explicit inputs so a changed server
    or checkpoint cannot silently reuse extraction checkpoints.
    """
    def __init__(self, model='graph-qwen35-9b-fp8', endpoint='http://127.0.0.1:8001',
                 weight_manifest=None, server_config=None, timeout=300):
        if urlparse(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Graph jobs require a local inference endpoint')
        if not weight_manifest or not server_config:
            raise ValueError('vLLM requires a weight manifest and server configuration')
        self.endpoint, self.model, self.timeout = endpoint.rstrip('/'), model, timeout
        available = self.request('/v1/models')['data']
        served = next((m for m in available if m['id'] == model), None)
        if served is None:
            raise ValueError(f'Model not served by local vLLM: {model}')
        weights = json.loads(Path(weight_manifest).read_text())
        config = json.loads(Path(server_config).read_text())
        self.grammars = {}
        for schema_hash, specification in config.get('json_grammars', {}).items():
            path = Path(server_config).resolve().parent / specification['file']
            if file_digest(path) != specification['sha256']:
                raise ValueError('JSON grammar checksum differs from pinned configuration')
            self.grammars[schema_hash] = path.read_text()
        if served.get('root') != weights['local_model_root'] or served.get('max_model_len') != config['max_model_len']:
            raise ValueError('Live vLLM model path/context differs from the pinned configuration')
        self.identity = {'model': model, 'backend': 'vllm',
                         'server_version': self.request('/version')['version'],
                         'weights_manifest_sha256': file_digest(weight_manifest),
                         'server_config_sha256': file_digest(server_config),
                         'sampling': {'temperature': 0, 'top_p': 1, 'seed': 42,
                                      'presence_penalty': 0, 'frequency_penalty': 0,
                                      'repetition_penalty': 1, 'enable_thinking': False}}

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.endpoint + path, data=data,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def chat(self, system, user, schema=None, max_tokens=1100):
        payload = {'model': self.model, 'messages': [{'role': 'system', 'content': system},
                   {'role': 'user', 'content': user}], 'stream': False, 'temperature': 0,
                   'top_p': 1, 'seed': 42, 'max_tokens': max_tokens,
                   'presence_penalty': 0, 'frequency_penalty': 0, 'repetition_penalty': 1,
                   'chat_template_kwargs': {'enable_thinking': False}}
        if schema is not None:
            if digest(schema) in self.grammars:
                payload['structured_outputs'] = {'grammar': self.grammars[digest(schema)]}
            else:
                payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                    'name': 'graph_evidence', 'strict': True, 'schema': schema}}
        started = time.monotonic()
        response = self.request('/v1/chat/completions', payload)
        choice = response['choices'][0]
        usage = response.get('usage', {})
        metrics = {'prompt_eval_count': usage.get('prompt_tokens', 0),
                   'eval_count': usage.get('completion_tokens', 0),
                   'done_reason': choice.get('finish_reason'),
                   'wall_seconds': time.monotonic() - started,
                   'usage': usage, 'backend': 'vllm'}
        return choice['message'].get('content', '') or '', metrics
