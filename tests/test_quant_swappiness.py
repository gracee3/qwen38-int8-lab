"""Exercise the host wrapper with fake sysctl/sudo; never modify the host."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

WRAPPER = Path(__file__).resolve().parents[1] / "scripts/with_quant_swappiness.sh"


class SwappinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "setting"
        self.state.write_text("60\n")
        scripts = {
            "sysctl": '#!/bin/bash\nif [[ $1 == -n ]]; then cat "$TEST_STATE"; else printf "%s\\n" "${3#*=}" > "$TEST_STATE"; fi\n',
            "sudo": '#!/bin/bash\nshift\nexec "$@"\n',
        }
        for name, body in scripts.items():
            path = self.root / name
            path.write_text(body)
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}",
                        WORK_ROOT=str(self.root), TEST_STATE=str(self.state))

    def test_restores_after_success_and_failure(self):
        for status in (0, 7):
            with self.subTest(status=status):
                result = subprocess.run(["bash", str(WRAPPER), "bash", "-c",
                    f'[[ $(sysctl -n vm.swappiness) == 1 ]] || exit 99; exit {status}'],
                    env=self.env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(self.state.read_text().strip(), "60")

    def test_restores_after_termination(self):
        ready = self.root / "ready"
        process = subprocess.Popen(["bash", str(WRAPPER), "bash", "-c",
            'touch "$WORK_ROOT/ready"; exec sleep 30'], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists())
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=10)
            self.assertEqual(process.returncode, 143)
            self.assertEqual(self.state.read_text().strip(), "60")
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
