#!/usr/bin/env python3
"""Bounded W4A16 synthetic quantizer gate; never loads the real 27B model."""
import argparse
import json
import re
from pathlib import Path

from quantize import synthetic_config, PeakMonitor, resource_abort_limits


def main():
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import Qwen3_5ForConditionalGeneration, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from safetensors import safe_open

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/work/scratch').is_dir():
        raise RuntimeError('PeakMonitor requires the dedicated scratch mount at /work/scratch')
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU required')
    torch.manual_seed(42)
    config = synthetic_config()
    # Scale dimensions while retaining 3 GDN / 1 full-attention layers and
    # Marlin-compatible 128-column groups. Production dimensions audited separately.
    text = config.text_config
    text.hidden_size = 512
    text.intermediate_size = 1024
    text.head_dim = 128
    text.linear_key_head_dim = 128
    text.linear_value_head_dim = 128
    config.vision_config.out_hidden_size = 512
    model = Qwen3_5ForConditionalGeneration(config).to(device='cuda:0', dtype=torch.bfloat16)
    pattern = re.compile(r'model\.language_model\.layers\.\d+\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))')
    targets = [name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and pattern.fullmatch(name)]
    assert len(targets) == 25, targets
    preserved = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name.removesuffix('.weight') not in targets}
    backend = Tokenizer(WordLevel(vocab={'[PAD]': 0, '[UNK]': 1, **{f'token_{i}': i for i in range(2,512)}}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', model_max_length=128)
    dataset = Dataset.from_list([{'input_ids': [2 + (i*17+j)%478 for j in range(128)], 'attention_mask':[1]*128} for i in range(4)])
    modifier = GPTQModifier(targets=targets, scheme='W4A16', block_size=128, dampening_frac=0.01, actorder=None)
    print('stage=synthetic_gptq targets=25 dtype=bfloat16 group=128 actorder=None', flush=True)
    with PeakMonitor(abort_limits=resource_abort_limits({})) as monitor:
        oneshot(model=model, processor=tokenizer, dataset=dataset, recipe=[modifier], max_seq_length=128, num_calibration_samples=4, pipeline='sequential', sequential_targets=['Qwen3_5DecoderLayer'])
        for name, old in preserved.items():
            assert torch.equal(old, model.state_dict()[name].cpu()), name
        args.output.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(args.output, save_compressed=True, safe_serialization=True, max_shard_size='1GB')
        tokenizer.save_pretrained(args.output)
        weights = {}
        for shard in args.output.glob('*.safetensors'):
            with safe_open(shard, framework='pt') as handle:
                for name in handle.keys():
                    weights[name] = handle.get_tensor(name)
        for name in targets:
            assert name+'.weight_packed' in weights, name
            assert name+'.weight' not in weights, name
            assert weights[name+'.weight_packed'].dtype == torch.int32, name
        metadata = json.loads((args.output/'config.json').read_text())['quantization_config']
        assert metadata['format'] == 'pack-quantized', metadata
        for group in metadata['config_groups'].values():
            w = group['weights']
            assert w['num_bits']==4 and w['group_size']==128 and w['symmetric'] and w.get('actorder') is None, w
            assert group.get('input_activations') is None, group
        result = {'status':'passed', 'scope':'synthetic_quantization_only_not_runtime_or_real_source', 'targets':targets, 'preserved_count':len(preserved), 'peaks':monitor.report()}
        assert monitor.samples > 0 and monitor.thread.is_alive(), 'Memory monitor failed'
        (args.output/'pilot-result.json').write_text(json.dumps(result, indent=2)+'\n')
        print('stage=synthetic_gptq status=passed; runtime loader/Marlin still required', flush=True)


if __name__ == '__main__':
    main()
