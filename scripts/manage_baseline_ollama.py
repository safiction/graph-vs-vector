"""Start/pull/inspect/stop only the dedicated, project-owned Ollama server."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline_rag.common import read_json, write_json


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'pull', 'status', 'stop'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baseline_15000.json')
    args = parser.parse_args()
    config = read_json(args.config)
    settings = config['generation_settings']
    if settings['backend'] != 'ollama':
        raise ValueError('This helper requires an Ollama generation configuration')
    endpoint = settings['endpoint'].rstrip('/')
    url = urlparse(endpoint)
    if url.hostname not in {'127.0.0.1', 'localhost', '::1'} or url.scheme != 'http':
        raise ValueError('The managed server must use a local HTTP endpoint')
    output = (ROOT / config['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = (ROOT / 'data/models/ollama').resolve()
    models.mkdir(parents=True, exist_ok=True)
    pid_path = output / 'ollama.pid'

    def request(path, payload=None, timeout=10):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(endpoint + path, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def owned_process():
        if not pid_path.exists():
            return None
        raw = pid_path.read_bytes()
        text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        try:
            process = psutil.Process(int(text.strip()))
            env = process.environ()
            same_models = os.path.normcase(os.path.abspath(env.get('OLLAMA_MODELS', ''))) == os.path.normcase(str(models))
            if ('serve' in process.cmdline() and Path(process.exe()).name.lower() in {'ollama', 'ollama.exe'} and
                    env.get('OLLAMA_HOST') == url.netloc and same_models):
                return process
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            return None
        return None

    def check_version():
        version = request('/api/version')['version']
        if version != settings['server_version']:
            raise ValueError(f'Ollama {version} differs from pinned {settings["server_version"]}')
        return version

    def start():
        process = owned_process()
        if process is not None:
            return {'pid': process.pid, 'version': check_version(), 'already_running': True}
        try:
            request('/api/version', timeout=1)
        except (urllib.error.URLError, TimeoutError):
            pass
        else:
            raise RuntimeError('Endpoint is already occupied by an unmanaged server; choose another port')
        executable = shutil.which('ollama')
        if not executable:
            raise RuntimeError('Install the pinned Ollama release and put ollama on PATH, or use the Transformers backend')
        env = {**os.environ, 'OLLAMA_HOST': url.netloc, 'OLLAMA_MODELS': str(models)}
        options = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        with (output / 'ollama.stdout.log').open('ab') as stdout, (output / 'ollama.stderr.log').open('ab') as stderr:
            child = subprocess.Popen([executable, 'serve'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, **options)
        pid_path.write_text(str(child.pid) + '\n', encoding='utf-8')
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError('Ollama exited during startup; inspect ollama.stderr.log')
            try:
                version = check_version()
                write_json(output / 'ollama_server.json', {'pid': child.pid, 'endpoint': endpoint,
                           'models_directory': str(models), 'version': version})
                return {'pid': child.pid, 'version': version, 'already_running': False}
            except (urllib.error.URLError, TimeoutError):
                time.sleep(.1)
        raise TimeoutError('Ollama did not become ready; inspect its logs')

    if args.action == 'start':
        result = start()
    elif args.action == 'pull':
        start()
        result = request('/api/pull', {'model': settings['served_model'], 'stream': False}, timeout=1800)
        match = next((m for m in request('/api/tags')['models'] if m['name'] == settings['served_model']), None)
        if not match or match['digest'] != settings['server_model_digest']:
            raise ValueError('Downloaded model does not match the pinned digest; copy the recorded model cache instead')
        result['verified_digest'] = match['digest']
    elif args.action == 'status':
        result = {'version': check_version(), 'models': request('/api/tags'), 'residency': request('/api/ps')}
    else:
        process = owned_process()
        if process is None:
            raise RuntimeError('No verified project-owned server; refusing to stop another process')
        try:
            request('/api/generate', {'model': settings['served_model'], 'keep_alive': 0}, timeout=30)
        except (urllib.error.URLError, TimeoutError):
            pass
        processes = process.children(recursive=True) + [process]
        for child in processes:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=5)
        for child in alive:
            child.kill()
        result = {'stopped_project_server': process.pid}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
