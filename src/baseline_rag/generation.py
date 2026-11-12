"""One real-tokenizer evidence-to-answer adapter for all four retrievers."""
import json
import re
import time
import urllib.request
from .common import digest


class SharedGenerator:
    """Compatible with graph_rag.run's module:factory adapter contract."""
    def __init__(self, settings):
        from transformers import AutoTokenizer
        self.settings = dict(settings)
        if not re.fullmatch('[0-9a-f]{40}', settings['revision']):
            raise ValueError('Use an immutable answer model/tokenizer revision')
        if settings.get('temperature') != 0:
            raise ValueError('The shared baseline uses deterministic greedy decoding')
        self.tokenizer = AutoTokenizer.from_pretrained(settings['model'], revision=settings['revision'],
            cache_dir=settings['cache_dir'], trust_remote_code=False)
        self.model = None
        self.identity = {'settings_sha256': digest(self.settings), 'model': settings['model'],
                         'revision': settings['revision'], 'backend': settings['backend'],
                         'tokenizer_sha256': digest(self.tokenizer.backend_tokenizer.to_str()),
                         'chat_template_sha256': digest(self.tokenizer.chat_template)}
        if settings['backend'] == 'ollama':
            version = self._request('/api/version')['version']
            served = settings.get('served_model', settings['model'])
            match = next((m for m in self._request('/api/tags')['models'] if m['name'] == served), None)
            if match is None:
                raise ValueError(f'Ollama model not installed: {served}')
            if settings.get('server_model_digest') != match['digest'] or settings.get('server_version') != version:
                raise ValueError('Ollama version/model digest differs from frozen settings')
            if settings.get('server_quantization') != match['details']['quantization_level']:
                raise ValueError('Ollama quantization differs from frozen settings')
            self.identity.update(server_version=version, server_model_digest=match['digest'],
                                 server_details=match['details'])

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def messages(self, question, context):
        return [{'role': 'system', 'content': self.settings['system_prompt']},
                {'role': 'user', 'content': self.settings['user_template'].format(question=question, context=context)}]

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        if self.settings['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.manual_seed(self.settings['seed'])
        extra = {}
        quantization = self.settings.get('quantization')
        if quantization:
            if quantization != 'nf4' or self.settings['device'] != 'cuda':
                raise ValueError('Quantization supports NF4 on CUDA; use null for unquantized inference')
            extra = {'quantization_config': BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, self.settings['dtype'])), 'device_map': {'': 0}}
        self.model = AutoModelForCausalLM.from_pretrained(self.settings['model'],
            revision=self.settings['revision'], cache_dir=self.settings['cache_dir'],
            trust_remote_code=False, dtype=getattr(torch, self.settings['dtype']),
            attn_implementation='sdpa', **extra)
        if not quantization:
            self.model.to(self.settings['device'])
        self.model.eval()
        self.identity['model_parameters'] = self.model.num_parameters()
        self.identity['model_memory_bytes'] = self.model.get_memory_footprint()
        self.identity['device'] = str(next(self.model.parameters()).device)
        self.identity['quantization'] = extra['quantization_config'].to_dict() if quantization else None

    def generate(self, question, context):
        messages = self.messages(question, context)
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
            tokenize=True, return_tensors='pt', return_dict=True)
        prompt_length = encoded['input_ids'].shape[-1]
        if prompt_length > self.settings['max_input_tokens']:
            raise ValueError('Full prompt exceeds the frozen input limit; no silent truncation')
        backend = self.settings['backend']
        if backend != 'transformers':
            return self._remote(messages, prompt_length)
        import torch
        if self.model is None:
            self._load()
        input_ids = encoded['input_ids'].to(self.settings['device'])
        mask = encoded.get('attention_mask', torch.ones_like(encoded['input_ids'])).to(self.settings['device'])
        if torch.cuda.is_available() and self.settings['device'] == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(input_ids=input_ids, attention_mask=mask, do_sample=False,
                max_new_tokens=self.settings['max_new_tokens'], use_cache=True,
                logits_to_keep=1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if self.settings['device'] == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        answer_ids = output[0, prompt_length:].tolist()
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        finish = 'length' if len(answer_ids) == self.settings['max_new_tokens'] else 'stop'
        return {'answer': answer, 'metrics': {'backend': backend, 'prompt_tokens': prompt_length,
            'completion_tokens': len(answer_ids), 'wall_seconds': elapsed, 'finish_reason': finish,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated() if self.settings['device'] == 'cuda' else 0}}

    def _remote(self, messages, prompt_tokens):
        backend = self.settings['backend']
        if backend == 'openai-compatible':
            path = '/v1/chat/completions'
            payload = {'model': self.settings.get('served_model', self.settings['model']), 'messages': messages,
                'temperature': 0, 'seed': self.settings['seed'], 'max_tokens': self.settings['max_new_tokens'], 'stream': False}
        elif backend == 'ollama':
            path = '/api/chat'
            payload = {'model': self.settings.get('served_model', self.settings['model']), 'messages': messages,
                'stream': False, 'keep_alive': self.settings.get('keep_alive', '5m'),
                'options': {'temperature': 0, 'seed': self.settings['seed'],
                'repeat_penalty': self.settings.get('repeat_penalty', 1.0),
                'top_p': self.settings.get('top_p', 1.0),
                'num_predict': self.settings['max_new_tokens'],
                'num_ctx': self.settings['max_input_tokens'] + self.settings['max_new_tokens']}}
            if 'enable_thinking' in self.settings:
                payload['think'] = self.settings['enable_thinking']
        else:
            raise ValueError(f'Unknown generation backend: {backend}')
        started = time.perf_counter()
        result = self._request(path, payload)
        if backend == 'ollama':
            answer = result['message']['content']
            metrics = {'prompt_tokens': result.get('prompt_eval_count'),
                       'completion_tokens': result.get('eval_count'), 'finish_reason': result.get('done_reason'),
                       'prompt_cached_tokens': result.get('prompt_eval_cached_count'),
                       'server_timings_ns': {key: result.get(key) for key in ['total_duration', 'load_duration',
                           'prompt_eval_duration', 'eval_duration']}}
            model_name = self.settings.get('served_model', self.settings['model'])
            loaded = next((m for m in self._request('/api/ps').get('models', []) if m['name'] == model_name), None)
            if loaded and loaded.get('digest') != self.settings.get('server_model_digest'):
                raise ValueError('Loaded Ollama model digest changed during the run')
            if self.settings.get('require_gpu') and (not loaded or loaded.get('size_vram', 0) <= 0):
                raise RuntimeError('Answer model has no GPU residency; GPU execution was required')
            metrics['gpu_residency'] = loaded
        else:
            answer = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            metrics = {'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
                       'finish_reason': result['choices'][0].get('finish_reason')}
        return {'answer': answer, 'metrics': {**metrics, 'backend': backend,
            'wall_seconds': time.perf_counter() - started, 'local_prompt_tokens': prompt_tokens}}

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.settings['endpoint'].rstrip('/') + path, data=data,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.settings.get('timeout_seconds', 300)) as response:
            return json.load(response)

    def close(self):
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_adapter(generation_settings):
    return SharedGenerator(generation_settings)
