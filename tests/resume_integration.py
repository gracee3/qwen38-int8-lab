"""Run inside quant image, without GPUs: baseline, interrupt, fresh-process resume."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quant/scripts'))
import torch
from datasets import Dataset
from transformers import Qwen3_5ForConditionalGeneration, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from quantize import synthetic_config
from resume_checkpoint import Checkpoints, register_pipeline

p = argparse.ArgumentParser()
p.add_argument('stage', choices=['baseline', 'interrupt', 'resume', 'compare'])
p.add_argument('scheme', choices=['W8A8', 'W4A16'])
p.add_argument('root', type=Path)
a = p.parse_args()
a.root.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(42)
if a.stage == 'compare':
    expected = torch.load(a.root / 'baseline.pt', weights_only=True)
    actual = torch.load(a.root / 'resume.pt', weights_only=True)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    print(f'EXACT_RESUME_MATCH {a.scheme}: {len(expected)} tensors')
    raise SystemExit(0)
config = synthetic_config()
# Group-128 INT4 requires the synthetic GDN out_proj input to be >=128.
config.text_config.linear_value_head_dim = 32
model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
from compressed_tensors.offload import offload_module
for module in model.modules():
    offload_module(module, torch.device('cpu'), torch.device('cpu'))
targets = [name for name, module in model.named_modules()
           if isinstance(module, torch.nn.Linear) and 'language_model.layers.' in name
           and ('.mlp.' in name or '.self_attn.' in name or
                (a.scheme == 'W4A16' and name.endswith(('in_proj_qkv', 'in_proj_z', 'out_proj'))))]
modifier = GPTQModifier(targets=targets, scheme=a.scheme, block_size=128,
                        dampening_frac=0.01, actorder=None)
dataset = Dataset.from_list([{'input_ids': list(range(1, 17)), 'attention_mask': [1]*16},
                            {'input_ids': list(range(17, 33)), 'attention_mask': [1]*16}])
tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
    vocab={'[PAD]': 0, '[UNK]': 1}, unk_token='[UNK]')), pad_token='[PAD]', unk_token='[UNK]')
pipeline = 'sequential'
if a.stage != 'baseline':
    pipeline = register_pipeline(Checkpoints(a.root / 'checkpoints', {'scheme': a.scheme},
        resume=a.stage == 'resume', interval=0.4, stop_after=2 if a.stage == 'interrupt' else None))
try:
    oneshot(model=model, processor=tokenizer, dataset=dataset, recipe=[modifier],
            max_seq_length=16, num_calibration_samples=2, pipeline=pipeline,
            sequential_targets=['Qwen3_5DecoderLayer'])
except RuntimeError as exc:
    if a.stage == 'interrupt' and str(exc) == 'TEST_CHECKPOINT_INTERRUPT':
        print('EXPECTED_CHECKPOINT_INTERRUPT')
        raise SystemExit(0)
    raise
assert a.stage != 'interrupt', 'Expected injected interruption did not occur'
torch.save(model.state_dict(), a.root / f'{a.stage}.pt')
