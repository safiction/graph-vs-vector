import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_kaggle_t4 import validate_model_cache, package_code


class KagglePayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / 'models'
        blob = b'Synthetic model blob'
        self.blob_digest = hashlib.sha256(blob).hexdigest()
        self.blob = self.cache / 'blobs' / ('sha256-' + self.blob_digest)
        self.blob.parent.mkdir(parents=True)
        self.blob.write_bytes(blob)
        self.manifest = self.cache / 'manifests/registry.ollama.ai/library/qwen2.5/3b'
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text(json.dumps({'config': {'digest': 'sha256:' + self.blob_digest, 'size': len(blob)},
                                             'layers': []}), encoding='utf-8')
        self.identity = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def test_validates_exact_model_manifest_and_every_blob_before_packaging(self):
        paths = validate_model_cache(self.cache, self.identity)
        self.assertEqual(set(paths), {self.blob, self.manifest})

    def test_project_archive_excludes_credentials_caches_and_data(self):
        project = self.root / 'project'
        for relative in ['src/module.py', 'scripts/run.py', 'configs/baseline_15000.json', 'requirements.txt',
                         '.kaggle/kaggle.json', '.env', 'data/private.txt', 'src/__pycache__/x.pyc']:
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}' if path.suffix == '.json' else 'fixture')
        archive = self.root / 'project_code.tar.gz'
        result = package_code(project, archive)
        expected = {'src/module.py', 'scripts/run.py', 'configs/baseline_15000.json', 'requirements.txt'}
        self.assertEqual(set(result['files']), expected)
        with tarfile.open(archive) as tar:
            self.assertEqual(set(tar.getnames()), expected)

    def test_rejects_changed_manifest_or_tampered_blob(self):
        with self.assertRaises(ValueError):
            validate_model_cache(self.cache, '0' * 64)
        self.blob.write_bytes(b'TAMPERED')
        with self.assertRaises(ValueError):
            validate_model_cache(self.cache, self.identity)


if __name__ == '__main__':
    unittest.main()
