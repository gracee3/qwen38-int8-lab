import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "eval" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class DatasetPinTests(unittest.TestCase):
    def test_injects_and_rejects_revisions(self):
        calls = []

        def fake_load(*args, **kwargs):
            calls.append((args, kwargs))
            return "loaded"

        fake = types.SimpleNamespace(
            load_dataset=fake_load,
            DownloadMode=types.SimpleNamespace(REUSE_DATASET_IF_EXISTS="reuse"),
        )
        prior = sys.modules.get("datasets")
        sys.modules["datasets"] = fake
        try:
            module = load_module("pinned_datasets")
            module.install({"approved/data": "abc123"}, offline=True)
            self.assertEqual(fake.load_dataset("approved/data"), "loaded")
            self.assertEqual(calls[-1][1]["revision"], "abc123")
            self.assertEqual(calls[-1][1]["download_mode"], "reuse")
            with self.assertRaises(RuntimeError):
                fake.load_dataset("approved/data", revision="wrong")
            with self.assertRaises(RuntimeError):
                fake.load_dataset("unapproved/data")
        finally:
            if prior is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = prior


class SuitePolicyTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / "eval/config/leaderboard-v2.yaml").open(encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

    def test_exact_harness_and_dataset_pins(self):
        self.assertEqual(self.config["harness"]["version"], "0.4.12")
        self.assertEqual(
            self.config["harness"]["git_revision"],
            "6d642546f4688648fced259eb3302efd36ece5af",
        )
        self.assertEqual(
            self.config["harness"]["group_yaml_sha256"],
            "dcf26c03fadaff36643041bb8a6c16dba04ac0eba33117253a10011895781bcd",
        )
        revisions = {item["revision"] for item in self.config["datasets"].values()}
        self.assertEqual(len(revisions), 6)
        self.assertTrue(all(len(revision) == 40 for revision in revisions))

    def test_retention_thresholds_and_pairing(self):
        self.assertEqual(self.config["protocol"]["context_length"], 16384)
        self.assertEqual(self.config["models"]["w8a8"]["max_batch_size"], 1)
        self.assertEqual(self.config["models"]["bf16"]["max_batch_size"], 1)
        self.assertEqual(self.config["acceptance"]["bootstrap_replicates"], 10000)
        self.assertEqual(self.config["acceptance"]["bootstrap_seed"], 42)
        self.assertEqual(self.config["acceptance"]["macro_min_delta"], -0.02)
        self.assertEqual(self.config["acceptance"]["individual_min_delta"], -0.05)
        paired = [name for name, task in self.config["tasks"].items() if task["paired"]]
        self.assertEqual(paired, ["mmlu_pro", "bbh", "gpqa", "musr"])

    def test_eval_lock_contains_direct_contract(self):
        lock = (ROOT / "docker/eval/requirements.lock").read_text(encoding="utf-8")
        self.assertIn("lm_eval==0.4.12\n", lock)
        self.assertIn("antlr4-python3-runtime==4.11.0\n", lock)
        for line in lock.splitlines():
            if line and not line.startswith("#"):
                self.assertIn("==", line)

    def test_supervisor_smokes_both_models_before_scoring(self):
        supervisor = (ROOT / "scripts/accuracy_eval_supervisor.sh").read_text(
            encoding="utf-8"
        )
        w8a8_smoke = supervisor.index("run_eval_stage w8a8 smoke leaderboard 2")
        bf16_smoke = supervisor.index("run_eval_stage bf16 smoke-bf16 leaderboard 2")
        first_score = supervisor.index("for group in mmlu_pro bbh gpqa math_hard ifeval musr")
        self.assertLess(w8a8_smoke, bf16_smoke)
        self.assertLess(bf16_smoke, first_score)
        self.assertIn("--user 0:0", supervisor)
        self.assertIn("container_eval.sh", supervisor)


class AggregationTests(unittest.TestCase):
    def test_metric_selection_is_finite_and_exact(self):
        try:
            aggregate = load_module("aggregate")
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        self.assertEqual(aggregate.metric_value({"acc,none": 0.75}, "acc"), 0.75)
        with self.assertRaises(ValueError):
            aggregate.metric_value({"acc,none": float("nan")}, "acc")
        with self.assertRaises(ValueError):
            aggregate.metric_value({"other,none": 1.0}, "acc")

    def test_stratified_bootstrap_is_deterministic(self):
        try:
            aggregate = load_module("aggregate")
            import numpy as np
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        strata = {"a": np.asarray([0.0, 1.0]), "b": np.asarray([-1.0, 0.0, 1.0])}
        first = aggregate.stratified_bootstrap(strata, 100, 42)
        second = aggregate.stratified_bootstrap(strata, 100, 42)
        self.assertEqual(first, second)
        self.assertEqual(first[2], 5)


if __name__ == "__main__":
    unittest.main()
