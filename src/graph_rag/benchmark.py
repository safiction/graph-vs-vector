"""Measure actual batch wall time and GPU telemetry, rather than summing latency."""
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from .common import write_json
from .extract import extract


class GPUSampler:
    def __init__(self):
        self.samples, self.errors = [], []
        self.stop = threading.Event()

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
                utilization, memory, power = map(float, result.stdout.splitlines()[0].split(','))
                self.samples.append({'time': time.time(), 'gpu_utilization_percent': utilization,
                                     'memory_mib': memory, 'power_watts': power})
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.errors.append(str(exc))
            self.stop.wait(0.5)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)

    def report(self):
        if not self.samples:
            return {'samples': [], 'errors': self.errors}
        return {'samples': self.samples, 'errors': self.errors,
                'mean_gpu_utilization_percent': statistics.mean(s['gpu_utilization_percent'] for s in self.samples),
                'peak_gpu_utilization_percent': max(s['gpu_utilization_percent'] for s in self.samples),
                'peak_memory_mib': max(s['memory_mib'] for s in self.samples),
                'mean_power_watts': statistics.mean(s['power_watts'] for s in self.samples)}


def profile_extract(chunks, manifest, output, model, sample, workers):
    output = Path(output)
    if (output / 'extraction_config.json').exists():
        raise ValueError('Benchmark needs a fresh directory; cached calls would invalidate throughput')
    with GPUSampler() as gpu:
        started = time.monotonic()
        status = extract(chunks, manifest, output, model, sample=sample, workers=workers)
        wall = time.monotonic() - started
    records = [json.loads(p.read_text()) for p in sorted((output / 'records').glob('*.json'))]
    attempts = [a for r in records for a in r['attempts']]
    generated = sum(a.get('metrics', {}).get('eval_count', 0) or 0 for a in attempts)
    result = {'workers': workers, 'elapsed_wall_seconds': wall, 'generated_tokens': generated,
              'completion_tokens_per_second': generated / wall,
              'chunks_per_second': status['completed_chunks'] / wall,
              'status': status, 'model': model.identity,
              'accepted_entity_mentions': sum(len(r.get('entities', [])) for r in records),
              'accepted_relations': sum(len(r.get('relations', [])) for r in records),
              'rejected_items': sum(len(r.get('rejected', [])) for r in records),
              'gpu': gpu.report()}
    write_json(output / 'profile.json', result)
    return result
