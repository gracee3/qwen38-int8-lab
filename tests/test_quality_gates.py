#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quant"))
SPEC = importlib.util.spec_from_file_location("quantize", ROOT / "quant/quantize.py")
assert SPEC and SPEC.loader
quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantize)


class ResourceGuardTests(unittest.TestCase):
    def test_defaults_and_override(self):
        self.assertEqual(quantize.resource_abort_limits({}), {
            "min_mem_available_bytes": 8 * 1024**3,
            "max_swap_growth_bytes": 32 * 1024**3,
            "sustain_seconds": 10,
        })
        self.assertEqual(quantize.resource_abort_limits({"safety": {"max_swap_growth_gib": 48}})["max_swap_growth_bytes"], 48 * 1024**3)

    def test_rejects_invalid_limits(self):
        for value in (0, -1, True, "32", 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantize.resource_abort_limits({"safety": {"max_swap_growth_gib": value}})

    def test_sustained_pressure_reports_reason_before_signal(self):
        # Exercise the monitor without allocating memory or delivering a signal.
        with patch.object(quantize.PeakMonitor, "_process_memory_status", return_value=(0, 0)), \
             patch.object(quantize.PeakMonitor, "_memory_status", return_value=(64 * 1024**3, 0)):
            monitor = quantize.PeakMonitor(abort_limits=quantize.resource_abort_limits({}))
        monitor._unsafe_since = 1
        with patch.object(monitor, "_process_memory_status", return_value=(0, 0)), \
             patch.object(monitor, "_memory_status", return_value=(64 * 1024**3, 33 * 1024**3)), \
             patch.object(monitor, "_read_gpus", return_value={}), \
             patch.object(quantize.shutil, "disk_usage") as disk, \
             patch.object(quantize.time, "monotonic", return_value=20), \
             patch.object(quantize.os, "kill") as kill, \
             patch.object(quantize, "print") as output:
            disk.return_value.free = 100 * 1024**3
            monitor._run()
            kill.assert_called_once_with(quantize.os.getpid(), quantize.signal.SIGINT)
            self.assertEqual(monitor.safety_trigger_reason, "host_swap_growth_above_limit")
            self.assertIn("RESOURCE_SAFETY_ABORT", output.call_args.args[0])


class ProcessorCopyTests(unittest.TestCase):
    def test_copies_required_json_byte_for_byte_and_reports_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir(); output.mkdir()
            payloads = {
                "preprocessor_config.json": b'{"processor_class":"A", "n":1}\n',
                "video_preprocessor_config.json": b'{\n  "processor_class": "B"\n}\n',
            }
            for name, payload in payloads.items():
                (source / name).write_bytes(payload)
            result = quantize.copy_processor_configs(source, output)
            for name, payload in payloads.items():
                self.assertEqual((output / name).read_bytes(), payload)
                self.assertEqual(result[name]["size_bytes"], len(payload))
                self.assertEqual(result[name]["sha256"], hashlib.sha256(payload).hexdigest())

    def test_rejects_external_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir(); output.mkdir()
            external = root / "external.json"
            external.write_text("{}")
            (source / "preprocessor_config.json").symlink_to(external)
            (source / "video_preprocessor_config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "escapes source checkpoint"):
                quantize.copy_processor_configs(source, output)

    def test_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir(); output.mkdir()
            (source / "preprocessor_config.json").write_text("[]")
            (source / "video_preprocessor_config.json").write_text("{}")
            with self.assertRaisesRegex(TypeError, "JSON object"):
                quantize.copy_processor_configs(source, output)


class DatasetMetadataTests(unittest.TestCase):
    def test_statistics_contain_identity_without_content(self) -> None:
        class Dataset:
            _source_fingerprint = "source-fp"
            _fingerprint = "token-fp"
            rows = [{"input_ids": [1] * 5}, {"input_ids": [2] * 8}]
            def __len__(self): return len(self.rows)
            def __getitem__(self, index): return self.rows[index]

        result = quantize.dataset_statistics(
            Dataset(),
            {"dataset": "owner/data", "revision": "abc", "split": "train", "seed": 42},
            {"num_samples": 2, "max_seq_length": 8},
        )
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["revision"], "abc")
        self.assertEqual(result["token_lengths"]["maximum"], 8)
        self.assertEqual(result["token_lengths"]["at_maximum"], 1)
        self.assertNotIn("rows", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
