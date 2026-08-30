#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark", ROOT / "inference/scripts/benchmark.py")
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkSummaryTests(unittest.TestCase):
    def test_distribution_reports_dispersion_and_extrema(self) -> None:
        result = benchmark.distribution([1.0, 2.0, 4.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["minimum"], 1.0)
        self.assertEqual(result["maximum"], 4.0)
        self.assertGreater(result["population_standard_deviation"], 0)

    def test_payload_forces_exact_length_generation(self) -> None:
        result = benchmark.payload("model", "prompt", 256)
        self.assertTrue(result["ignore_eos"])
        self.assertTrue(result["stream_options"]["include_usage"])
        self.assertEqual(result["max_tokens"], 256)
        self.assertEqual(result["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
