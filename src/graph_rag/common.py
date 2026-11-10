import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def normalize(text):
    # Only case/Unicode/punctuation variants are merged, never ambiguous acronyms.
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def node_id(name):
    return 'ent_' + digest(normalize(name))[:20]


def verified_source(raw, expected):
    """Accept newline conversion only when the EXACT manifest hash is recovered."""
    if digest(raw) == expected:
        return raw, 'exact'
    lf = raw.replace(b'\r\n', b'\n')
    for candidate, label in [(lf, 'LF'), (lf.replace(b'\n', b'\r\n'), 'CRLF')]:
        if digest(candidate) == expected:
            return candidate, label
    raise ValueError('Source content differs from manifest (including newline variants)')


def safe_source(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Source escapes dataset root: {relative}')
    return path
