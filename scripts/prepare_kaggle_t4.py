"""Build local-only private Kaggle inputs; never upload or run a kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_SHA256 = 'e364aedc9a991cbf36f704baa807a04f432d46e32265b783e7dec9ce0336af88'
GRAPH_SHA256 = '2e9576c1fcb6160567d1a5a48786b36ceff4b5bb4723cbfcf17fc12f20bcc546'
BASELINE_SHA256 = 'e19bfbc170a207a7271c241c84bd3a551e29009c9ba1a137306a1c0d4cf6b2e2'
TOKENIZER_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_model_cache(root, expected_digest):
    """Return only the pinned manifest and its verified blobs, not unrelated models."""
    root = Path(root)
    manifest = root / 'manifests/registry.ollama.ai/library/qwen2.5/3b'
    if sha256(manifest) != expected_digest:
        raise ValueError('Ollama model manifest differs from the frozen digest')
    value = json.loads(manifest.read_text(encoding='utf-8'))
    paths = {manifest}
    for descriptor in [value['config'], *value['layers']]:
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', descriptor['digest']):
            raise ValueError('Invalid model blob digest')
        path = root / 'blobs' / descriptor['digest'].replace(':', '-')
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Model blob escapes cache')
        if path.stat().st_size != descriptor['size'] or sha256(path) != descriptor['digest'][7:]:
            raise ValueError('Model blob is incomplete or changed')
        paths.add(path)
    return sorted(paths)


def package_code(root, target):
    """Archive only public source/configuration inputs, never a checkout or data tree."""
    root, target = Path(root), Path(target)
    paths = [p for directory in ('src', 'scripts') for p in (root / directory).rglob('*.py')]
    paths += list((root / 'configs').glob('*.json')) + list(root.glob('requirements*.txt'))
    paths = sorted(p for p in paths if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts)
    names = [p.relative_to(root).as_posix() for p in paths]
    if target.exists():
        raise FileExistsError('Refusing to replace a frozen code archive')
    with tarfile.open(target, 'w:gz') as tar:
        for path, name in zip(paths, names):
            tar.add(path, arcname=name, recursive=False)
    return {'files': names, 'sha256': sha256(target)}


def prepare_payload(output, baseline_archive, project_root=ROOT):
    """Prepare verified offline assets; ownership/private metadata is added after login."""
    root, output = Path(project_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new payload directory; never overwrite an upload snapshot')
    config = json.loads((root / 'configs/baseline_15000.json').read_text(encoding='utf-8'))
    supplied = {
        'graph_results_15000.zip': (root / 'data/exports/graph_results_15000.zip', GRAPH_SHA256),
        'baseline_results_15000.zip': (Path(baseline_archive), BASELINE_SHA256),
        'ollama-0.18.0.tar.zst': (root / 'data/tools/ollama-0.18.0.complete.tar.zst', OLLAMA_SHA256)}
    for name, (path, expected) in supplied.items():
        if sha256(path) != expected:
            raise ValueError(f'Input archive checksum mismatch: {name}')
    models = root / 'data/models/ollama'
    model_files = validate_model_cache(models, config['generation_settings']['server_model_digest'])
    tokenizer = root / 'data/models/models--Qwen--Qwen2.5-3B-Instruct'
    snapshot = tokenizer / 'snapshots' / TOKENIZER_REVISION
    if not all((snapshot / name).is_file() for name in ('tokenizer.json', 'tokenizer_config.json')):
        raise ValueError('Pinned answer tokenizer snapshot is incomplete')
    for path in tokenizer.rglob('*'):
        if not path.resolve().is_relative_to(tokenizer.resolve()):
            raise ValueError('Tokenizer cache contains a link outside its repository')
    output.mkdir(parents=True)
    code = package_code(root, output / 'project_code.tar.gz')
    for name, (path, _) in supplied.items():
        shutil.copy2(path, output / name)
    with tarfile.open(output / 'ollama_models.tar.gz', 'w:gz', compresslevel=1) as tar:
        for path in model_files:
            tar.add(path, arcname=path.relative_to(models).as_posix(), recursive=False)
    with tarfile.open(output / 'answer_tokenizer.tar.gz', 'w:gz') as tar:
        tar.add(tokenizer, arcname=tokenizer.name)
    assets = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir())}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
    manifest = {'schema_version': 1, 'commit': commit, 'dirty': dirty, 'assets': assets,
                'source': 'EnterpriseRAG-Bench frozen 15k; mixed upstream licenses retained',
                'privacy': 'private inputs and private kernel only', 'code_files': code['files']}
    (output / 'paired-inputs.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(assets)} verified assets in {output}; nothing uploaded or executed')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-archive', type=Path, required=True)
    args = parser.parse_args()
    prepare_payload(args.output, args.baseline_archive)
