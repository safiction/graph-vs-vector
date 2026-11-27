"""Standalone Kaggle T4 kernel: verify a private paired-inputs dataset, then run a
paired graph preliminary evaluation on an actual NVIDIA T4 using the frozen
baseline runner. Imports no repository helpers until the project code is unpacked.

Hardware is asserted, never assumed: nvidia-smi must report a real Tesla/NVIDIA T4
on GPU0 and CUDA_VISIBLE_DEVICES=0 before any asset is executed. Every asset is
digest-checked against the manifest and the official Ollama release hash before
extraction. Receipts are written to <workdir>/output and an explicit error receipt
is saved on failure; a failed run is never reported as success.
"""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

ASSET_WHITELIST = (
    'project_code.tar.gz',
    'graph_results_15000.zip',
    'baseline_results_15000.zip',
    'ollama-0.18.0.tar.zst',
    'ollama_models.tar.gz',
    'answer_tokenizer.tar.gz',
)
OLLAMA_RELEASE_SHA256 = 'e364aedc9a991cbf36f704baa807a04f432d46e32265b783e7dec9ce0336af88'
SOURCE_LABEL = 'EnterpriseRAG-Bench frozen 15k; mixed upstream licenses retained'
DEFAULT_WORKDIR = '/kaggle/working/paired_graph'
DEFAULT_STAGE = 'smoke'
STAGES = ('smoke', 'preliminary')
OLLAMA_HOST = '127.0.0.1:11435'
OLLAMA_ENDPOINT = 'http://127.0.0.1:11435'
EXPERIMENT_OUTPUT = 'data/experiments/graph_t4_preliminary'

# Archives whose contents legitimately carry in-root relative symlinks
# (Ollama library and model-cache symlinks). Project source must not.
ASSET_INROOT_SYMLINKS = frozenset({'ollama_models.tar.gz', 'answer_tokenizer.tar.gz'})

_T4_TOKEN = re.compile(r'(?<![A-Za-z0-9])T4(?![A-Za-z0-9])')

SMOKE_QUESTION = 'What does the supplied evidence state?'
SMOKE_CONTEXT = ('[chunk-smoke]\nThis is a synthetic smoke context used only to '
                 'verify the pinned answer pipeline end to end.')

_SMOKE_SCRIPT = r'''
import json
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
output_path = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(project / 'src'))
from baseline_rag.generation import SharedGenerator

config = json.loads(config_path.read_text(encoding='utf-8'))
adapter = SharedGenerator(config['generation_settings'])
try:
    result = adapter.generate(
        "What does the supplied evidence state?",
        "[chunk-smoke]\nThis is a synthetic smoke context used only to verify the pinned answer pipeline end to end.")
finally:
    adapter.close()
if not isinstance(result.get('answer'), str) or not result['answer'].strip():
    raise ValueError('Synthetic smoke returned an empty answer')
metrics = result.get('metrics', {})
diagnostic = {
    'status': 'ok',
    'answer': result.get('answer'),
    'model_identity': adapter.identity,
    'gpu_residency': metrics.get('gpu_residency'),
    'cost': {k: metrics.get(k) for k in ('prompt_tokens', 'completion_tokens', 'wall_seconds', 'finish_reason')},
    'note': 'Synthetic adapter smoke; not one of the 104 graph attempts.',
}
output_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(diagnostic, indent=2, ensure_ascii=False))
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_receipt(output, name, value):
    path = Path(output) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    return path


def parse_manifest(raw):
    """Validate paired-inputs.json and return its normalized fields."""
    if not isinstance(raw, dict):
        raise ValueError('paired-inputs.json must be a JSON object')
    if raw.get('schema_version') != 1:
        raise ValueError('unsupported paired-inputs schema_version')
    commit = raw.get('commit')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('commit must be a 40-hex git sha')
    assets = raw.get('assets')
    if not isinstance(assets, dict) or not assets:
        raise ValueError('assets must be a nonempty object')
    normalized = {}
    for name, meta in assets.items():
        if not isinstance(name, str) or not name:
            raise ValueError('asset name must be a nonempty string')
        sha = meta.get('sha256') if isinstance(meta, dict) else None
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise ValueError(f'asset {name!r} requires a 64-hex sha256')
        normalized[name] = {'sha256': sha}
    if raw.get('source') != SOURCE_LABEL:
        raise ValueError('paired-inputs source must match the frozen EnterpriseRAG-Bench label')
    return {'schema_version': 1, 'commit': commit, 'assets': normalized, 'source': SOURCE_LABEL}


def verify_assets(assets_dir, manifest, ollama_release_sha256=OLLAMA_RELEASE_SHA256):
    """Verify the exact asset whitelist and every digest; return per-asset records."""
    declared = manifest['assets']
    if set(declared) != set(ASSET_WHITELIST):
        raise ValueError('asset set must exactly match the whitelist')
    records = []
    for name in ASSET_WHITELIST:
        path = Path(assets_dir) / name
        if not path.is_file():
            raise ValueError(f'missing asset: {name}')
        sha = file_digest(path)
        if sha != declared[name]['sha256']:
            raise ValueError(f'asset digest mismatch: {name}')
        records.append({'name': name, 'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size})
    ollama = next(r for r in records if r['name'] == 'ollama-0.18.0.tar.zst')
    if ollama['sha256'] != ollama_release_sha256:
        raise ValueError('ollama-0.18.0.tar.zst digest differs from the official release hash')
    return {'assets': records, 'ollama_release_verified': True}


def parse_nvidia_smi(text):
    gpus, driver = [], None
    for row in csv.reader(text.splitlines()):
        if len(row) < 3 or not row[0].strip():
            continue
        gpus.append({'index': len(gpus), 'name': row[0].strip(),
                     'memory': row[1].strip(), 'driver': row[2].strip()})
        if driver is None:
            driver = row[2].strip()
    return {'driver': driver, 'gpus': gpus}


def is_t4(name):
    return bool(_T4_TOKEN.search(name or ''))


def require_t4(listing):
    gpus = listing.get('gpus') or []
    if not gpus:
        raise RuntimeError('nvidia-smi reported no GPUs')
    gpu0 = gpus[0]
    if not is_t4(gpu0['name']):
        raise RuntimeError(f'GPU0 is {gpu0["name"]!r}, not an NVIDIA T4 (Tesla T4); '
                           'refusing to relabel non-T4 hardware')
    return gpu0


def safe_member(root, name):
    root = Path(root).resolve()
    name = str(name).replace('\\', '/')
    if name.startswith('/') or re.match(r'^[A-Za-z]:', name):
        raise ValueError(f'unsafe absolute member path: {name!r}')
    target = (root / name).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f'unsafe member path escapes root: {name!r}')
    return target


def safe_symlink_target(root, name, target):
    root = Path(root).resolve()
    target = str(target)
    if target.startswith('/') or re.match(r'^[A-Za-z]:', target):
        raise ValueError(f'unsafe absolute symlink target: {name!r} -> {target!r}')
    resolved = (Path(root) / name).parent.joinpath(target).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f'unsafe symlink escapes root: {name!r} -> {target!r}')
    return resolved


def validate_tar_members(dest, names, symlinks=None):
    for name in names:
        safe_member(dest, name)
    for name, target in (symlinks or []):
        safe_member(dest, name)
        safe_symlink_target(dest, name, target)


def require_executable(name, hint):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f'{name} not found: {hint}')
    return path


def extract_zip(zip_path, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            safe_member(dest, info.filename)
            if (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
                raise ValueError(f'symlink rejected in {Path(zip_path).name}: {info.filename}')
        zf.extractall(dest)


def extract_tar_gz(tar_path, dest, allow_inroot_symlink=False):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tf:
        members = tf.getmembers()
        for member in members:
            safe_member(dest, member.name)
            if member.issym():
                if not allow_inroot_symlink:
                    raise ValueError(f'symlink rejected in {Path(tar_path).name}: {member.name}')
                safe_symlink_target(dest, member.name, member.linkname)
            elif member.islnk():
                raise ValueError(f'hard link rejected in {Path(tar_path).name}: {member.name}')
            elif not (member.isfile() or member.isdir()):
                raise ValueError(f'unsupported member type in {Path(tar_path).name}: {member.name}')
        for member in members:
            safe_member(dest, member.name)
            if member.issym():
                safe_symlink_target(dest, member.name, member.linkname)
            tf.extract(member, str(dest), filter='data')


def extract_zst(tar_zst, dest):
    require_executable('tar', 'install GNU tar to unpack archives')
    require_executable('zstd', 'install zstd (e.g. apt-get install zstd) to unpack the pinned Ollama release')
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = subprocess.run(['tar', '--zstd', '-tf', str(tar_zst)], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    verbose = subprocess.run(['tar', '--zstd', '-tvf', str(tar_zst)], capture_output=True,
                             text=True, check=True).stdout.splitlines()
    symlinks = []
    for line in verbose:
        if ' -> ' in line:
            left, target = line.split(' -> ', 1)
            symlinks.append((left.rsplit(None, 1)[-1], target.strip()))
    validate_tar_members(dest, names, symlinks)
    subprocess.run(['tar', '--zstd', '-xf', str(tar_zst), '-C', str(dest)], check=True)


_ASSET_DEST = {
    'project_code.tar.gz': 'project',
    'graph_results_15000.zip': 'project/data/packages',
    'baseline_results_15000.zip': 'project/data/packages',
    'ollama-0.18.0.tar.zst': 'project/data/tools/ollama-0.18.0',
    'ollama_models.tar.gz': 'project/data/models/ollama',
    'answer_tokenizer.tar.gz': 'project/data/models',
}


def asset_destination(workdir, filename):
    if filename not in _ASSET_DEST:
        raise ValueError(f'unknown asset: {filename}')
    return Path(workdir) / _ASSET_DEST[filename]


def discover_dataset_dir(input_root=None, search_root='/kaggle/input'):
    if input_root:
        directory = Path(input_root).resolve()
        if not (directory / 'paired-inputs.json').is_file():
            raise ValueError(f'paired-inputs.json not found in --input-root: {input_root}')
        return directory
    base = Path(search_root)
    matches = sorted(base.rglob('paired-inputs.json')) if base.exists() else []
    if len(matches) != 1:
        raise ValueError(f'expected exactly one paired-inputs.json under {search_root}, found {len(matches)}')
    return matches[0].parent.resolve()


def ollama_binary(workdir):
    return Path(workdir) / 'project/data/tools/ollama-0.18.0/bin/ollama'


def ollama_models_dir(workdir):
    return Path(workdir) / 'project/data/models/ollama'


def build_runtime_config(delivered_config, output=EXPERIMENT_OUTPUT):
    """Change only the local output path; generator/cache settings stay frozen."""
    if not isinstance(delivered_config, dict) or 'output' not in delivered_config:
        raise ValueError('delivered config must be an object with an output field')
    generation = delivered_config.get('generation_settings')
    if not isinstance(generation, dict) or generation.get('cache_dir') != 'data/models':
        raise ValueError('generation_settings.cache_dir must remain data/models relative to the project cwd')
    config = dict(delivered_config)
    config['output'] = output
    return config


def verify_frozen_parity(generated_experiment_path, generated_questions_dir,
                         reference_experiment_path, reference_questions_dir,
                         splits=('development', 'preliminary')):
    if load_json(generated_experiment_path) != load_json(reference_experiment_path):
        raise ValueError('generated experiment.json differs from the delivered baseline experiment.json')
    for split in splits:
        generated = Path(generated_questions_dir) / f'{split}.queries.jsonl'
        reference = Path(reference_questions_dir) / f'{split}.queries.jsonl'
        if generated.read_bytes() != reference.read_bytes():
            raise ValueError(f'{split}.queries.jsonl bytes differ from the delivered baseline')
    return True


def baseline_prepare_command(python_exe, config_path):
    return [str(python_exe), '-m', 'baseline_rag', '--config', str(config_path), 'prepare']


def baseline_run_command(python_exe, config_path,
                         retrievers=('graph-local', 'graph-global'),
                         splits=('development', 'preliminary')):
    return [str(python_exe), '-m', 'baseline_rag', '--config', str(config_path), 'run',
            '--retrievers', *retrievers, '--splits', *splits]


def pip_install_command(python_exe, requirements_path):
    return [str(python_exe), '-m', 'pip', 'install', '-r', str(requirements_path)]


def ollama_serve_command(ollama_exe, models_dir, host=OLLAMA_HOST):
    return ([str(ollama_exe), 'serve'],
            {'OLLAMA_MODELS': str(models_dir), 'OLLAMA_NO_CLOUD': '1', 'OLLAMA_HOST': host})


def project_env(project):
    return {**os.environ, 'PYTHONPATH': str(Path(project) / 'src'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0')}


def run_nvidia_smi():
    proc = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                           '--format=csv,noheader'], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError('nvidia-smi failed: ' + proc.stderr.strip())
    return proc.stdout


def extract_asset(workdir, filename, assets_dir):
    dest = asset_destination(workdir, filename)
    source = Path(assets_dir) / filename
    if filename.endswith('.zip'):
        extract_zip(source, dest)
    elif filename.endswith('.tar.zst'):
        extract_zst(source, dest)
    else:
        extract_tar_gz(source, dest, allow_inroot_symlink=filename in ASSET_INROOT_SYMLINKS)


def create_venv(venv_dir):
    subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
    return venv_dir / 'bin' / 'python'


def pip_install(python_exe, requirements_path):
    subprocess.run(pip_install_command(python_exe, requirements_path), check=True)


def pip_freeze(python_exe):
    proc = subprocess.run([str(python_exe), '-m', 'pip', 'freeze'],
                          capture_output=True, text=True, check=True)
    return proc.stdout


def run_command(command, cwd, log_path):
    with open(log_path, 'ab') as log:
        proc = subprocess.run(command, cwd=str(cwd), env=project_env(cwd),
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def start_ollama(ollama_exe, models_dir, log_path):
    argv, env = ollama_serve_command(ollama_exe, models_dir)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'), **env}
    log = open(log_path, 'ab')
    return subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_for_ollama(endpoint=OLLAMA_ENDPOINT, timeout_seconds=300, poll_seconds=0.5):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint + '/api/version', timeout=2) as response:
                if response.status == 200:
                    return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(poll_seconds)
    raise TimeoutError('Ollama server did not become ready within the bounded poll')


def stop_ollama(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def run_smoke(python_exe, project, config_path, output_path):
    script = Path(project).parent / '_smoke.py'
    script.write_text(_SMOKE_SCRIPT, encoding='utf-8')
    return run_command([str(python_exe), str(script), str(project), str(config_path), str(output_path)],
                       project, Path(output_path).parent / 'smoke.log')


def locate_baseline_reference(baseline_dir):
    baseline = Path(baseline_dir)
    experiments = sorted(baseline.rglob('experiment.json'))
    questions = sorted(baseline.rglob('development.queries.jsonl'))
    if len(experiments) != 1 or len(questions) != 1:
        raise RuntimeError('baseline_results_15000 package must contain exactly one frozen '
                           'experiment.json and one development.queries.jsonl')
    return experiments[0], questions[0].parent


def make_output_archive(workdir, output, experiment_dir, final_name, final_receipt):
    if final_receipt.get('success') is False:
        write_receipt(output, 'status.json', final_receipt)
    write_receipt(output, final_name, final_receipt)
    archive = Path(workdir) / 'paired_graph_output.tar.gz'
    temporary = archive.with_suffix(archive.suffix + '.tmp')
    try:
        with tarfile.open(temporary, 'w:gz') as tf:
            tf.add(output, arcname='output')
            if experiment_dir and Path(experiment_dir).exists():
                tf.add(experiment_dir, arcname='experiment')
        temporary.replace(archive)
    except Exception as exc:
        if final_receipt.get('success') is not False:
            raise
        failed = {**final_receipt, 'archive_error': f'{type(exc).__name__}: {exc}'}
        write_receipt(output, 'status.json', failed)
        write_receipt(output, final_name, failed)
        print(f'Failure archive could not be written: {exc}', file=sys.stderr)
        return None
    return archive


def main(argv=None):
    args = parse_args(argv)
    workdir = Path(args.workdir).resolve()
    if workdir.exists():
        print(f'Refusing to overwrite existing workdir: {workdir}', file=sys.stderr)
        return 2
    workdir.mkdir(parents=True)
    output = workdir / 'output'
    output.mkdir(parents=True)
    receipts = {'workdir': str(workdir), 'stage': args.stage}
    server = None
    try:
        dataset = discover_dataset_dir(args.input_root)
        manifest = parse_manifest(load_json(dataset / 'paired-inputs.json'))
        receipts['dataset_dir'] = str(dataset)
        receipts['commit'] = manifest['commit']
        write_receipt(output, 'inputs.json', receipts)

        listing = parse_nvidia_smi(run_nvidia_smi())
        gpu0 = require_t4(listing)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        write_receipt(output, 'hardware.json', {'driver': listing['driver'], 'gpus': listing['gpus'],
                                                'selected_gpu': gpu0, 'cuda_visible_devices': '0'})

        write_receipt(output, 'assets.json', verify_assets(dataset, manifest))

        # Unpack project first so the delivered config can drive the remaining paths.
        extract_asset(workdir, 'project_code.tar.gz', dataset)
        project = workdir / 'project'
        runtime_config = build_runtime_config(load_json(project / 'configs/graph_delivery_15000.json'))
        config_path = output / 'config.json'
        config_path.write_text(json.dumps(runtime_config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        write_receipt(output, 'config_meta.json', {'config_sha256': digest(runtime_config),
                                                   'delivered_config': 'configs/graph_delivery_15000.json'})
        for name in ASSET_WHITELIST:
            if name != 'project_code.tar.gz':
                extract_asset(workdir, name, dataset)
        write_receipt(output, 'runtime.json', {
            'ollama_binary_sha256': file_digest(ollama_binary(workdir)),
            'ollama_models_dir': str(ollama_models_dir(workdir)),
            'tokenizer_cache_dir': str(project / 'data/models'),
        })

        setup_started = time.perf_counter()
        venv_python = create_venv(workdir / 'venv')
        pip_install(venv_python, project / 'requirements-graph-answers.txt')
        freeze = pip_freeze(venv_python)
        setup_seconds = time.perf_counter() - setup_started
        write_receipt(output, 'setup.json', {'venv': str(workdir / 'venv'), 'pip_freeze': freeze,
                                             'setup_seconds': setup_seconds,
                                             'note': 'Setup duration is reported separately from inference cost.'})

        server = start_ollama(ollama_binary(workdir), ollama_models_dir(workdir), output / 'ollama.log')
        try:
            version = wait_for_ollama()
            write_receipt(output, 'ollama_server.json', {'endpoint': OLLAMA_ENDPOINT, 'version': version})

            smoke_exit = run_smoke(venv_python, project, config_path, output / 'smoke.json')
            if smoke_exit != 0 or not (output / 'smoke.json').is_file() or load_json(output / 'smoke.json').get('status') != 'ok':
                raise RuntimeError('synthetic adapter smoke failed; see smoke.log')
            receipts['smoke'] = 'synthetic adapter smoke saved to output/smoke.json'
            if args.stage == 'preliminary':
                prepare_cmd = baseline_prepare_command(venv_python, config_path)
                if run_command(prepare_cmd, project, output / 'prepare.log') != 0:
                    raise RuntimeError('baseline prepare failed; see output/prepare.log')
                baseline_pkg = workdir / 'project/data/packages/baseline_results_15000'
                ref_exp, ref_q = locate_baseline_reference(baseline_pkg)
                verify_frozen_parity(project / EXPERIMENT_OUTPUT / 'experiment.json',
                                     project / EXPERIMENT_OUTPUT / 'questions', ref_exp, ref_q)
                run_cmd = baseline_run_command(venv_python, config_path)
                exit_status = run_command(run_cmd, project, output / 'run.log')
                write_receipt(output, 'run.json', {'exit_status': exit_status,
                                                   'experiment_output': str(project / EXPERIMENT_OUTPUT),
                                                   'success': exit_status == 0})
                if exit_status != 0:
                    raise RuntimeError('baseline run exited nonzero; see output/run.log')
        finally:
            stop_ollama(server)
            server = None

        make_output_archive(workdir, output, project / EXPERIMENT_OUTPUT, 'status.json',
                            {**receipts, 'success': True, 'stage': args.stage,
                             'hardware_verified': 'NVIDIA T4',
                             'note': 'Query-time use of a prebuilt graph; not a T4 construction claim.'})
        print(json.dumps({'workdir': str(workdir), 'output': str(output), 'success': True}, indent=2))
        return 0
    except Exception as exc:
        if server is not None:
            stop_ollama(server)
        make_output_archive(workdir, output, workdir / 'project' / EXPERIMENT_OUTPUT, 'error.json',
                            {'success': False, 'stage': args.stage,
                             'error': f'{type(exc).__name__}: {exc}', 'receipts': receipts})
        print(f'FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=None,
                        help='dataset directory containing paired-inputs.json (default: discover under /kaggle/input)')
    parser.add_argument('--workdir', type=Path, default=Path(DEFAULT_WORKDIR))
    parser.add_argument('--stage', choices=STAGES, default=DEFAULT_STAGE)
    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main())
