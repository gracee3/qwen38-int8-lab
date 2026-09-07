"""Catalog/consumer contract and safe command generation, without a GPU or Docker."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving import catalog
from serving.launch import docker_command


class ServingTests(unittest.TestCase):
    def resolved(self, profile='int4-v1-96k-fp8-tp1', **kwargs):
        with patch.object(catalog.subprocess, 'check_output', side_effect=['a' * 40 + '\n', '']):
            return catalog.resolve(profile, **kwargs)

    def test_all_profiles_discovered_and_bound_to_recipes(self):
        profiles = catalog.discover()
        self.assertEqual(len(profiles), 8)
        self.assertEqual({p['model_id'] for p in profiles.values()}, {'int8-original', 'int8-v2', 'int4-v1'})
        self.assertEqual(profiles['int4-v1-262k-fp8-tp2']['evidence']['status'], 'imported_unvalidated')

    def test_overrides_and_bindings_do_not_mutate_source(self):
        before = catalog.discover()
        result = self.resolved(overrides={'max_model_len': 32768}, model_path='/other/model', gpu_devices=['GPU-abc-123'])
        self.assertEqual(result['profile']['runtime']['max_model_len'], 32768)
        self.assertEqual(result['overrides'], {'max_model_len': 32768})
        self.assertEqual(result['bindings'], {'model_path': '/other/model', 'gpu_devices': ['GPU-abc-123']})
        self.assertEqual(catalog.discover(), before)
        self.assertEqual(result['source']['commit'], 'a' * 40)
        self.assertFalse(result['source']['dirty'])
        self.assertEqual(len(result['source']['profile_sha256']), 64)

    def test_unknown_and_incompatible_selections_fail(self):
        for kwargs in ({'model_id': 'int8-v2'}, {'gpu_devices': ['0', '1']},
                       {'overrides': {'dtype': 'float16'}}, {'overrides': {'max_model_len': True}},
                       {'overrides': {'max_model_len': 262145}}, {'gpu_devices': ['all']},
                       {'model_path': '/tmp/one,two'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.resolved(**kwargs)
        with self.assertRaises(ValueError):
            self.resolved('missing-profile')
        with self.assertRaises(ValueError):
            self.resolved('int8-v2-16k-bf16-tp2', gpu_devices=['0', '0'])

    def test_unknown_schema_fields_and_inconsistent_hardware_fail(self):
        p = catalog.discover()['int4-v1-96k-fp8-tp1']
        for mutate in (lambda q: q.update(schema_version=2), lambda q: q.update(extra=True),
                       lambda q: q['runtime'].update(misspelled=True),
                       lambda q: q['hardware'].update(gpu_count=2),
                       lambda q: q['runtime'].update(speculative_decoding=True)):
            q = copy.deepcopy(p); mutate(q)
            with self.assertRaises(ValueError):
                catalog.validate(q)

    def test_engine_and_http_share_effective_values(self):
        result = self.resolved('int8-v2-16k-bf16-tp2')
        engine = catalog.vllm_args(result)
        server = catalog.server_args(result)
        self.assertEqual(engine['max_model_len'], 16384)
        self.assertEqual(server[server.index('--max-model-len') + 1], str(engine['max_model_len']))
        self.assertEqual(server[server.index('--kv-cache-memory-bytes') + 1], str(engine['kv_cache_memory_bytes']))
        self.assertIn('--enforce-eager', server)
        self.assertIn('--no-enable-prefix-caching', server)
        self.assertNotIn('speculative_decoding', engine)
        self.assertEqual(json.loads(server[server.index('--default-chat-template-kwargs') + 1]), {'enable_thinking': False})

    def test_command_never_pulls_and_does_not_expose_key(self):
        result = self.resolved('int8-original-262k-fp8-tp2')
        with patch.dict(os.environ, {'VLLM_API_KEY': 'not-a-real-secret'}):
            command = docker_command(result, work_root='/tmp/work space', cuda_toolkit='/usr/local/cuda-13.3', port=8123)
        self.assertNotIn('not-a-real-secret', ' '.join(command))
        self.assertIn('VLLM_API_KEY', command)
        self.assertEqual(command[command.index('--pull') + 1], 'never')
        self.assertEqual(command[command.index('--gpus') + 1], '"device=0,1"')
        self.assertIn('127.0.0.1:8123:8000', command)
        self.assertIn('type=bind,src=/usr/local/cuda-13.3,dst=/usr/local/cuda,readonly', command)
        self.assertIn(result['image']['local_image_id'], command)
        self.assertIn('serve', command)

    def test_bf16_profile_does_not_mount_cuda(self):
        command = docker_command(self.resolved('int8-v2-16k-bf16-tp2'), work_root='/tmp/work', cuda_toolkit='/no/toolkit', port=8000)
        self.assertNotIn('CUDA_HOME=/usr/local/cuda', command)
        self.assertFalse(any('src=/no/toolkit' in c for c in command))

    def test_discovery_has_no_docker_or_git_side_effects(self):
        with patch.object(catalog.subprocess, 'check_output', side_effect=AssertionError('external process')):
            self.assertTrue(catalog.discover())

    def test_cli_list_runs_from_another_directory(self):
        result = subprocess.run([sys.executable, '-B', str(catalog.ROOT / 'serving/launch.py'), 'list', '--json'], cwd='/tmp', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('int4-v1-96k-fp8-tp1', json.loads(result.stdout))

    def test_unknown_fields_rejected_during_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'serving/profiles').mkdir(parents=True)
            (root / 'environments').mkdir()
            (root / 'environments/images.lock.json').write_text((catalog.ROOT / 'environments/images.lock.json').read_text())
            p = catalog.discover()['int4-v1-96k-fp8-tp1']
            p['unexpected'] = 1
            (root / 'serving/profiles/int4-v1-96k-fp8-tp1.yaml').write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError, 'profile requires exactly'):
                catalog.discover(root)
