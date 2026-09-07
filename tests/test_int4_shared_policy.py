"""Integration contracts without loading or launching a real model."""
import ast
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SharedPolicyTests(unittest.TestCase):
    def test_real_entrypoint_uses_shared_checkpoint_and_limits(self):
        tree = ast.parse((ROOT / 'quant/quantize_int4.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
        self.assertTrue({'resource_abort_limits', 'Checkpoints', 'run_identity',
                         'register_pipeline'} <= names)
        oneshot = next(node for node in calls if isinstance(node.func, ast.Name)
                       and node.func.id == 'oneshot')
        pipeline = next(kw.value for kw in oneshot.keywords if kw.arg == 'pipeline')
        self.assertIsInstance(pipeline, ast.Name)
        self.assertEqual(pipeline.id, 'pipeline')

    def test_resume_rejected_before_launch_for_small_stage(self):
        result = subprocess.run(['bash', str(ROOT / 'quant/int4.sh'),
                                 'synthetic', '--resume'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--resume requires real-pilot or full', result.stderr)

    def test_unknown_option_rejected_before_launch(self):
        result = subprocess.run(['bash', str(ROOT / 'quant/int4.sh'),
                                 'full', '--invalid'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
