
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quant'))
from validation.validate_int4 import offset_norm_roundtrip


class OffsetNormValidation(unittest.TestCase):
    def test_exact_pinned_conversion_and_corruption(self):
        import torch
        from llmcompressor.modeling.offset_norm import CalibrationOffsetNorm
        original = torch.nn.Module()
        original.eps = 1e-6
        original.weight = torch.nn.Parameter(torch.tensor([.001, -.015, .33, -1.1], dtype=torch.bfloat16))
        raw = lambda t: bytes(t.detach().view(torch.uint8).tolist())
        source = raw(original.weight)
        restored = CalibrationOffsetNorm(original, None).restore(original)
        output = raw(restored.weight)
        name = 'model.language_model.layers.0.input_layernorm.weight'
        self.assertNotEqual(source, output)
        self.assertTrue(offset_norm_roundtrip(name, source, output))
        self.assertFalse(offset_norm_roundtrip(name, source, output[:-1] + bytes([output[-1] ^ 1])))
        for excluded in ['mtp.norm.weight', 'model.visual.norm.weight', 'model.language_model.layers.0.linear_attn.norm.weight', 'model.language_model.layers.0.self_attn.q_norm.weight', 'model.language_model.layers.64.input_layernorm.weight']:
            self.assertFalse(offset_norm_roundtrip(excluded, source, output))

    def test_nonfinite_rejected(self):
        self.assertFalse(offset_norm_roundtrip('model.language_model.norm.weight', b'\x80\x7f', b'\x80\x7f'))
