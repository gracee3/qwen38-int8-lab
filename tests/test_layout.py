"""Frozen baseline comparisons guard the no-retuning/no-dependency-update promise."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving.catalog import ROOT, discover


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads((ROOT / 'tests/fixtures/layout-baseline.json').read_text())

    def test_dependency_files_are_byte_identical_to_baseline(self):
        for path, expected in self.baseline['dependency_hashes'].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

    def test_recipe_settings_are_unchanged_except_relocated_preparer(self):
        for path, expected in self.baseline['quant_recipe_hashes'].items():
            with self.subTest(path=path):
                actual = yaml.safe_load((ROOT / path).read_text())
                digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                self.assertEqual(digest, expected)

    def test_native_profile_values_match_historical_yaml(self):
        profiles = discover()
        for name, prior in self.baseline['serving_profiles'].items():
            p = profiles[name]
            with self.subTest(name=name):
                for key in ('tensor_parallel_size', 'max_model_len', 'kv_cache_dtype',
                            'kv_cache_memory_bytes', 'enforce_eager', 'language_model_only',
                            'max_num_batched_tokens', 'max_num_seqs'):
                    self.assertEqual(p['runtime'][key], prior[key])
                for key in ('seed', 'enable_thinking', 'generation_config'):
                    self.assertEqual(p['generation'][key], prior[key])
                self.assertEqual(p['runtime']['enable_prefix_caching'], prior['initial_features']['prefix_caching'])
                self.assertEqual(p['runtime']['enable_chunked_prefill'], prior['initial_features']['chunked_prefill'])
                self.assertEqual(p['checkpoint_name'], Path(prior['model']).name)
                self.assertEqual(p['environment'], prior['environment'])

    def test_manifests_bind_the_published_recipe_bytes(self):
        for path in (ROOT / 'recipes').glob('*/manifest.json'):
            manifest = json.loads(path.read_text())
            self.assertEqual(hashlib.sha256((path.parent / manifest['quant_config']).read_bytes()).hexdigest(), manifest['quant_config_sha256'])
            for evidence in manifest['evidence']:
                self.assertTrue((ROOT / evidence).is_file())

    def test_all_moved_paths_exist_and_retired_paths_are_absent(self):
        mapping = json.loads((ROOT / 'docs/layout-map.json').read_text())
        for old, new in mapping['moves'].items():
            with self.subTest(path=old):
                self.assertFalse((ROOT / old).is_file())
                self.assertTrue((ROOT / new).is_file())
        for retired in mapping['retired']:
            self.assertFalse((ROOT / retired).is_file())
