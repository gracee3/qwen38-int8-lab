
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quant'))
from quant.resume_checkpoint import Checkpoints, digest


class CheckpointManifestTests(unittest.TestCase):
    def test_identity_and_checksum_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            generation = path / 'generation-test'
            generation.mkdir()
            for name in ('model.pt', 'rng.pt'):
                (generation / name).write_bytes(b'test')
            identity = {'recipe': 'int8'}
            metadata = {'version': 1, 'identity': identity, 'directory': generation.name,
                        'batches': 0, 'files': {name: digest(generation / name)
                                               for name in ('model.pt', 'rng.pt')}}
            (path / 'current.json').write_text(json.dumps(metadata))
            with self.assertRaises(FileExistsError):
                Checkpoints(root, identity).manifest()
            with self.assertRaisesRegex(ValueError, 'identity'):
                Checkpoints(root, {'recipe': 'int4'}, resume=True).manifest()
            cp = Checkpoints(root, identity, resume=True)
            self.assertEqual(cp.verified_dir(cp.manifest()), generation)
            (generation / 'model.pt').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                cp.verified_dir(cp.manifest())

    def test_exclusive_writer(self):
        with tempfile.TemporaryDirectory() as root:
            with Checkpoints(root, {}).locked():
                with self.assertRaises(BlockingIOError):
                    with Checkpoints(root, {}).locked():
                        self.fail('second writer acquired lock')
