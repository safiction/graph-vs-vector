"""Unit tests for the standalone Kaggle T4 preliminary kernel driver.

No Kaggle, GPU, network, or subprocess is required: hardware parsing, asset
verification, safe extraction, and command/config construction are tested
against pure seams and synthetic tempdir archives.
"""
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import kaggle_graph_preliminary as kgp


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HardwareTests(unittest.TestCase):
    def test_accepts_tesla_and_nvidia_t4_names(self):
        for name in ('Tesla T4', 'NVIDIA T4', 'T4'):
            listing = kgp.parse_nvidia_smi(f'{name}, 15360 MiB, 535.104.05\n')
            self.assertEqual(kgp.require_t4(listing)['name'], name)

    def test_rejects_non_t4_hardware(self):
        for name in ('NVIDIA GeForce RTX 5080', 'RTX 5080', 'NVIDIA L4', 'L4', 'CPU'):
            with self.subTest(name=name):
                listing = kgp.parse_nvidia_smi(f'{name}, 24564 MiB, 535.104.05\n')
                with self.assertRaises(RuntimeError):
                    kgp.require_t4(listing)

    def test_rejects_empty_gpu_listing(self):
        with self.assertRaises(RuntimeError):
            kgp.require_t4(kgp.parse_nvidia_smi(''))

    def test_t4_token_is_boundary_matched(self):
        self.assertTrue(kgp.is_t4('Tesla T4'))
        self.assertTrue(kgp.is_t4('NVIDIA T4'))
        self.assertFalse(kgp.is_t4('RTX 5080'))
        self.assertFalse(kgp.is_t4('L4'))
        self.assertFalse(kgp.is_t4('Tesla T4G'))


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self, assets, source=kgp.SOURCE_LABEL, schema_version=2):
        return {'schema_version': schema_version, 'commit': 'a' * 40,
                'assets': assets, 'source': source}

    def write_assets(self):
        contents = {name: (name + '-content').encode() for name in kgp.ASSET_WHITELIST}
        assets = {}
        for name, data in contents.items():
            (self.root / (name + '.bin')).write_bytes(data)
            assets[name] = {'sha256': sha256(data)}
        return self.manifest(assets), contents

    def test_schema2_opaque_storage_preserves_archive_bytes_on_kaggle(self):
        contents = {name: (name + '-opaque').encode() for name in kgp.ASSET_WHITELIST}
        manifest = self.manifest({name: {'sha256': sha256(data)} for name, data in contents.items()}, schema_version=2)
        for name, data in contents.items():
            (self.root / (name + '.bin')).write_bytes(data)
        try:
            parsed = kgp.parse_manifest(manifest)
            verified = kgp.verify_assets(self.root, parsed, sha256(contents['ollama-0.18.0.tar.zst']))
        except ValueError as exc:
            self.fail(f'Opaque version2 input archives rejected: {exc}')
        self.assertEqual(len(verified['assets']), 6)
        self.assertTrue(all(r['path'].endswith('.bin') for r in verified['assets']))

    def test_official_ollama_release_hash_is_pinned(self):
        self.assertEqual(
            kgp.OLLAMA_RELEASE_SHA256,
            'e364aedc9a991cbf36f704baa807a04f432d46e32265b783e7dec9ce0336af88')

    def test_parse_manifest_accepts_valid_input(self):
        manifest, contents = self.write_assets()
        parsed = kgp.parse_manifest(manifest)
        self.assertEqual(parsed['schema_version'], 2)
        self.assertEqual(parsed['commit'], 'a' * 40)
        self.assertEqual(set(parsed['assets']), set(kgp.ASSET_WHITELIST))

    def test_parse_manifest_rejects_bad_schema(self):
        manifest, _ = self.write_assets()
        cases = [
            dict(manifest, schema_version=1),
            dict(manifest, commit='short'),
            dict(manifest, commit='g' * 40),
            dict(manifest, source='other'),
        ]
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    kgp.parse_manifest(bad)
        bad_assets = dict(manifest, assets={'x.tar.gz': {'sha256': 'not-hex'}})
        with self.assertRaises(ValueError):
            kgp.parse_manifest(bad_assets)

    def test_verify_assets_accepts_exact_whitelist(self):
        manifest, contents = self.write_assets()
        ollama_hash = contents['ollama-0.18.0.tar.zst']
        result = kgp.verify_assets(self.root, kgp.parse_manifest(manifest),
                                   ollama_release_sha256=sha256(ollama_hash))
        self.assertTrue(result['ollama_release_verified'])
        self.assertEqual({r['name'] for r in result['assets']}, set(kgp.ASSET_WHITELIST))

    def test_verify_assets_rejects_tampered_asset(self):
        manifest, _ = self.write_assets()
        (self.root / 'project_code.tar.gz.bin').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            kgp.verify_assets(self.root, kgp.parse_manifest(manifest))

    def test_verify_assets_rejects_extra_or_missing_asset(self):
        manifest, contents = self.write_assets()
        extra = dict(manifest, assets={**manifest['assets'], 'extra.bin': {'sha256': 'a' * 64}})
        with self.assertRaisesRegex(ValueError, 'whitelist'):
            kgp.verify_assets(self.root, kgp.parse_manifest(extra))
        missing = {k: v for k, v in manifest['assets'].items() if k != 'graph_results_15000.zip'}
        (self.root / 'graph_results_15000.zip.bin').unlink()
        with self.assertRaises(ValueError):
            kgp.verify_assets(self.root, kgp.parse_manifest(self.manifest(missing)))

    def test_verify_assets_rejects_wrong_official_ollama_hash(self):
        manifest, contents = self.write_assets()
        with self.assertRaisesRegex(ValueError, 'official release hash'):
            kgp.verify_assets(self.root, kgp.parse_manifest(manifest),
                              ollama_release_sha256='f' * 64)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dest = self.root / 'out'

    def test_safe_member_rejects_absolute_and_traversal(self):
        for name in ('/etc/passwd', '../evil', 'a/../../evil', 'C:\\evil', '..\\evil'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    kgp.safe_member(self.dest, name)
        self.assertEqual(kgp.safe_member(self.dest, 'a/b.txt'), (self.dest / 'a/b.txt').resolve())

    def test_validate_tar_members_rejects_unsafe_paths_and_links(self):
        kgp.validate_tar_members(self.dest, ['bin/ollama', 'lib/libggml.so'], [])
        with self.assertRaises(ValueError):
            kgp.validate_tar_members(self.dest, ['../../evil'], [])
        kgp.validate_tar_members(self.dest, ['lib/libollama.so'],
                                 [('lib/libollama.so', 'libollama.so.0')])
        with self.assertRaises(ValueError):
            kgp.validate_tar_members(self.dest, ['lib/libollama.so'],
                                     [('lib/libollama.so', '/etc/passwd')])
        with self.assertRaises(ValueError):
            kgp.validate_tar_members(self.dest, ['lib/libollama.so'],
                                     [('lib/libollama.so', '../../etc/passwd')])

    def test_extract_zip_extracts_and_rejects_unsafe(self):
        good = self.root / 'good.zip'
        with zipfile.ZipFile(good, 'w') as zf:
            zf.writestr('a/b.txt', 'hello')
        kgp.extract_zip(good, self.dest)
        self.assertEqual((self.dest / 'a/b.txt').read_text(), 'hello')

        bad = self.root / 'bad.zip'
        with zipfile.ZipFile(bad, 'w') as zf:
            zf.writestr('../evil.txt', 'evil')
        with self.assertRaises(ValueError):
            kgp.extract_zip(bad, self.root / 'out2')
        self.assertFalse((self.root / 'evil.txt').exists())

        link = self.root / 'link.zip'
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(link, 'w') as zf:
            zf.writestr(info, 'target')
        with self.assertRaises(ValueError):
            kgp.extract_zip(link, self.root / 'out3')

    def test_extract_tar_gz_extracts_and_rejects_unsafe(self):
        good = self.root / 'good.tar.gz'
        with tarfile.open(good, 'w:gz') as tf:
            data = b'hello'
            info = tarfile.TarInfo('a/b.txt')
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        kgp.extract_tar_gz(good, self.dest)
        self.assertEqual((self.dest / 'a/b.txt').read_text(), 'hello')

        bad = self.root / 'bad.tar.gz'
        with tarfile.open(bad, 'w:gz') as tf:
            data = b'evil'
            info = tarfile.TarInfo('../evil.txt')
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        with self.assertRaises(ValueError):
            kgp.extract_tar_gz(bad, self.root / 'out2')
        self.assertFalse((self.root / 'evil.txt').exists())

        sym = self.root / 'sym.tar.gz'
        with tarfile.open(sym, 'w:gz') as tf:
            info = tarfile.TarInfo('lib/link')
            info.type = tarfile.SYMTYPE
            info.linkname = '../../etc/passwd'
            tf.addfile(info)
        with self.assertRaises(ValueError):
            kgp.extract_tar_gz(sym, self.root / 'out3')

    def test_extract_tar_gz_allows_inroot_symlink_when_permitted(self):
        sym = self.root / 'sym.tar.gz'
        with tarfile.open(sym, 'w:gz') as tf:
            info = tarfile.TarInfo('lib/libfoo.so')
            info.type = tarfile.SYMTYPE
            info.linkname = 'libfoo.so.1'
            tf.addfile(info)
            data = b'x'
            target = tarfile.TarInfo('lib/libfoo.so.1')
            target.size = len(data)
            tf.addfile(target, io.BytesIO(data))
        kgp.extract_tar_gz(sym, self.dest, allow_inroot_symlink=True)
        self.assertTrue((self.dest / 'lib/libfoo.so').is_symlink())
        with self.assertRaises(ValueError):
            kgp.extract_tar_gz(sym, self.root / 'out4')

    def test_inroot_symlink_chain_cannot_escape_during_extraction(self):
        archive = self.root / 'chain.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name, target in [('a', '.'), ('a/b', '..')]:
                info = tarfile.TarInfo(name)
                info.type = tarfile.SYMTYPE
                info.linkname = target
                tar.addfile(info)
            info = tarfile.TarInfo('a/b/escaped.txt')
            info.size = 3
            tar.addfile(info, io.BytesIO(b'bad'))
        with self.assertRaises((ValueError, tarfile.FilterError)):
            kgp.extract_tar_gz(archive, self.dest, allow_inroot_symlink=True)
        self.assertFalse((self.root / 'escaped.txt').exists())

    def test_require_executable_gives_actionable_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'zstd'):
            kgp.require_executable('definitely-not-a-real-zstd-binary', 'install zstd')


class OutputArchiveTests(unittest.TestCase):
    def test_failure_receipt_overrides_success_even_when_archive_cannot_be_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            output = work / 'output'
            output.mkdir()
            (output / 'status.json').write_text('{"success": true}')
            with patch.object(kgp.tarfile, 'open', side_effect=OSError('disk full')):
                kgp.make_output_archive(work, output, None, 'error.json',
                                        {'success': False, 'error': 'original failure'})
            self.assertFalse(json.loads((output / 'status.json').read_text())['success'])
            self.assertEqual(json.loads((output / 'error.json').read_text())['error'], 'original failure')

    def test_partial_archive_never_replaces_published_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            output = work / 'output'
            output.mkdir()
            archive = work / 'paired_graph_output.tar.gz'
            archive.write_bytes(b'previous complete archive')
            def failed_write(path, *args, **kwargs):
                Path(path).write_bytes(b'partial')
                raise OSError('archive write failed')
            with patch.object(kgp.tarfile, 'open', side_effect=failed_write):
                with self.assertRaises(OSError):
                    kgp.make_output_archive(work, output, None, 'status.json', {'success': True})
            self.assertEqual(archive.read_bytes(), b'previous complete archive')

    def test_synthetic_model_boundary_rejects_empty_success_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text(json.dumps({'generation_settings': {}}))
            adapter = SimpleNamespace(identity={}, close=lambda: None,
                                      generate=lambda *args: {'answer': ' ', 'metrics': {}})
            module = SimpleNamespace(SharedGenerator=lambda settings: adapter)
            with patch.dict(sys.modules, {'baseline_rag.generation': module}):
                with patch.object(sys, 'argv', ['smoke', str(root), str(config), str(root / 'result.json')]):
                    with self.assertRaisesRegex(ValueError, 'empty'):
                        exec(kgp._SMOKE_SCRIPT, {})

    def test_final_success_and_error_receipts_are_inside_downloadable_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for final_name, success in [('status.json', True), ('error.json', False)]:
                work = root / final_name
                output = work / 'output'
                output.mkdir(parents=True)
                receipt = {'success': success, 'stage': 'smoke'}
                archive = kgp.make_output_archive(work, output, None, final_name, receipt)
                with tarfile.open(archive) as tar:
                    self.assertIn('output/' + final_name, tar.getnames())
                    saved = json.load(tar.extractfile('output/' + final_name))
                    self.assertEqual(saved, receipt)


class ConfigAndCommandTests(unittest.TestCase):
    DELIVERED = {
        'bundle': 'data/packages/graph_results_15000/corpus',
        'output': 'data/experiments/graph_delivery_15000',
        'seed': 42, 'retrieval_k': 10, 'context_token_budget': 2560,
        'embedding': {'model': 'BAAI/bge-base-en-v1.5', 'cache_dir': 'data/models'},
        'generation_settings': {
            'backend': 'ollama', 'endpoint': 'http://127.0.0.1:11435',
            'served_model': 'qwen2.5:3b', 'model': 'Qwen/Qwen2.5-3B-Instruct',
            'cache_dir': 'data/models', 'temperature': 0, 'seed': 42,
        },
        'graph_index': 'data/packages/graph_results_15000/results/index.json',
        'graph_summaries': 'data/packages/graph_results_15000/results/summaries',
        'graph_depth': 1,
    }

    def test_build_runtime_config_changes_only_output(self):
        result = kgp.build_runtime_config(self.DELIVERED, 'data/experiments/graph_t4_preliminary')
        self.assertEqual(result['output'], 'data/experiments/graph_t4_preliminary')
        for key in self.DELIVERED:
            if key != 'output':
                self.assertEqual(result[key], self.DELIVERED[key])
        self.assertEqual(result['generation_settings']['cache_dir'], 'data/models')

    def test_build_runtime_config_rejects_cache_dir_change(self):
        delivered = json.loads(json.dumps(self.DELIVERED))
        delivered['generation_settings']['cache_dir'] = '/elsewhere'
        with self.assertRaisesRegex(ValueError, 'cache_dir'):
            kgp.build_runtime_config(delivered, 'data/experiments/graph_t4_preliminary')

    def test_baseline_run_command_exact_retrievers_and_splits_only(self):
        cmd = kgp.baseline_run_command('/v/bin/python', '/cfg.json')
        self.assertEqual(cmd[0], '/v/bin/python')
        self.assertIn('run', cmd)
        self.assertEqual(cmd[cmd.index('--retrievers') + 1:cmd.index('--retrievers') + 3],
                         ['graph-local', 'graph-global'])
        self.assertEqual(cmd[cmd.index('--splits') + 1:cmd.index('--splits') + 3],
                         ['development', 'preliminary'])
        self.assertNotIn('fold_a', cmd)
        self.assertNotIn('fold_b', cmd)
        self.assertNotIn('route', cmd)

    def test_prepare_pip_and_ollama_commands(self):
        prepare = kgp.baseline_prepare_command('/v/bin/python', '/cfg.json')
        self.assertEqual(prepare, ['/v/bin/python', '-m', 'baseline_rag', '--config', '/cfg.json', 'prepare'])
        pip = kgp.pip_install_command('/v/bin/python', '/req.txt')
        self.assertEqual(pip, ['/v/bin/python', '-m', 'pip', 'install', '-r', '/req.txt'])
        argv, env = kgp.ollama_serve_command('/o/bin/ollama', '/models')
        self.assertEqual(argv, ['/o/bin/ollama', 'serve'])
        self.assertEqual(env['OLLAMA_MODELS'], '/models')
        self.assertEqual(env['OLLAMA_NO_CLOUD'], '1')
        self.assertEqual(env['OLLAMA_HOST'], kgp.OLLAMA_HOST)

    def test_asset_destinations_match_unpack_layout(self):
        workdir = Path('/kaggle/working/paired_graph')
        self.assertEqual(kgp.asset_destination(workdir, 'project_code.tar.gz'), workdir / 'project')
        self.assertEqual(kgp.asset_destination(workdir, 'graph_results_15000.zip'),
                         workdir / 'project/data/packages')
        self.assertEqual(kgp.asset_destination(workdir, 'ollama-0.18.0.tar.zst'),
                         workdir / 'project/data/tools/ollama-0.18.0')
        self.assertEqual(kgp.asset_destination(workdir, 'ollama_models.tar.gz'),
                         workdir / 'project/data/models/ollama')
        self.assertEqual(kgp.asset_destination(workdir, 'answer_tokenizer.tar.gz'),
                         workdir / 'project/data/models')

    def test_discover_dataset_dir_requires_exactly_one(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        search = Path(tmp.name)
        with self.assertRaises(ValueError):
            kgp.discover_dataset_dir(search_root=search)
        (search / 'a').mkdir(parents=True)
        (search / 'a' / 'paired-inputs.json').write_text('{}')
        self.assertEqual(kgp.discover_dataset_dir(search_root=search), (search / 'a').resolve())
        (search / 'b').mkdir(parents=True)
        (search / 'b' / 'paired-inputs.json').write_text('{}')
        with self.assertRaises(ValueError):
            kgp.discover_dataset_dir(search_root=search)


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_verify_frozen_parity_rejects_any_difference(self):
        gen_exp = self.root / 'gen/experiment.json'
        ref_exp = self.root / 'ref/experiment.json'
        gen_q = self.root / 'gen/questions'
        ref_q = self.root / 'ref/questions'
        for d in (gen_exp, ref_exp):
            d.parent.mkdir(parents=True, exist_ok=True)
        for d in (gen_q, ref_q):
            d.mkdir(parents=True, exist_ok=True)
        gen_exp.write_text(json.dumps({'seed': 42}))
        ref_exp.write_text(json.dumps({'seed': 42}))
        (gen_q / 'development.queries.jsonl').write_text('a\n')
        (ref_q / 'development.queries.jsonl').write_text('a\n')
        (gen_q / 'preliminary.queries.jsonl').write_text('b\n')
        (ref_q / 'preliminary.queries.jsonl').write_text('b\n')
        self.assertTrue(kgp.verify_frozen_parity(gen_exp, gen_q, ref_exp, ref_q))

        ref_exp.write_text(json.dumps({'seed': 43}))
        with self.assertRaisesRegex(ValueError, 'experiment'):
            kgp.verify_frozen_parity(gen_exp, gen_q, ref_exp, ref_q)
        ref_exp.write_text(json.dumps({'seed': 42}))
        (ref_q / 'preliminary.queries.jsonl').write_text('c\n')
        with self.assertRaisesRegex(ValueError, 'preliminary'):
            kgp.verify_frozen_parity(gen_exp, gen_q, ref_exp, ref_q)


if __name__ == '__main__':
    unittest.main()
