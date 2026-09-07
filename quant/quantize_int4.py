#!/usr/bin/env python3
"""Dedicated W4A16 build entrypoint. Full builds require completed external gates."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from quant.quantize import PeakMonitor, ResourceSafetyError, resource_abort_limits, load_real_inputs, inject_mtp_tensors, copy_processor_configs, package_versions, utc_stamp, git_revision
from quant.resume_checkpoint import Checkpoints, register_pipeline, run_identity
from validation.validate_int4 import validate


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_corpus(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['corpus_sha256']==hashlib.sha256((root/'calibration.parquet').read_bytes()).hexdigest(), 'corpus changed'
    assert not manifest['failures'] and 1300000<=manifest['total_tokens']<=1700000
    assert len(manifest['sources'])==4
    assert all(s['revision'] and s['rows']>0 for s in manifest['sources'])
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/models/source'))
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume the matching durable calibration snapshot')
    p.add_argument('--real-pilot-report',type=Path)
    p.add_argument('--overlap-report',type=Path)
    p.add_argument('--runtime-report',type=Path,required=True)
    a=p.parse_args()
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available < 80*1024**3:
        raise RuntimeError('Real-source execution requires 80 GiB available host RAM')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if processes:
        raise RuntimeError('Another compute process is present; refusing real-source execution')
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    import torch
    if not Path('/work/scratch').is_dir():
        raise RuntimeError('PeakMonitor requires /work/scratch')
    corpus=check_corpus(a.corpus)
    runtime=json.loads(a.runtime_report.read_text())
    assert runtime['status']=='passed' and runtime['profiled_marlin_events'], 'Synthetic runtime gate required'
    assert torch.cuda.device_count()==1
    if a.output.exists(): raise FileExistsError(a.output)
    if not a.full and Path('/run-int4') not in a.output.parents: raise ValueError('Pilot output must be beneath /run-int4')
    if a.full and a.output!=Path('/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1'):
        raise ValueError('Full output must use the dedicated INT4 destination')
    audit=json.loads(a.audit.read_text())
    assert audit['status']=='passed' and len(audit['targets'])==400
    assert audit['source']['shard_hashes_verified'], 'Complete source hash audit required'
    for name,key in [('config.json','config_sha256'),('model.safetensors.index.json','index_sha256')]:
        assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==audit['source'][key]
    for name,expected in audit['source']['shard_sha256'].items():
        assert file_hash(a.source/name)==expected, 'Source changed: '+name
    dataset=Dataset.from_parquet(str(a.corpus/'calibration.parquet'))
    lengths=[len(row['input_ids']) for row in dataset]
    if a.full:
        if a.real_pilot_report is None or a.overlap_report is None:
            raise RuntimeError('Full build requires real pilot and reviewed held-out overlap reports')
        pilot=json.loads(a.real_pilot_report.read_text())
        overlap=json.loads(a.overlap_report.read_text())
        assert pilot['status']=='passed' and pilot['integrity']['status']=='passed'
        assert pilot['corpus_sha256']==corpus['corpus_sha256']
        assert max(pilot['sequence_lengths'])==max(lengths)
        assert pilot['peaks']['samples']>0 and pilot['peaks']['safety_trigger'] is None
        assert overlap['status']=='passed' and overlap['corpus_sha256']==corpus['corpus_sha256']
        assert overlap['unresolved_overlaps']==0 and overlap['heldout_manifest_sha256']
    shortest=sorted(range(len(lengths)),key=lengths.__getitem__)[:4]
    longest=max(range(len(lengths)),key=lengths.__getitem__)
    indices=list(range(len(dataset))) if a.full else list(dict.fromkeys(shortest+[longest]))
    dataset=dataset.select(indices).select_columns(['input_ids','attention_mask'])
    profile={'num_samples':len(dataset),'max_seq_length':max(len(row['input_ids']) for row in dataset)}
    config={'model':{'source':str(a.source)},'memory':{'offload_dir':'/run-int4/offload','max_cpu_gib':80},'calibration':{'corpus_dir':str(a.corpus),'sources':corpus['sources'],'seed':42}}
    import yaml
    recipe_path=Path(__file__).resolve().parents[1]/'recipes/int4-expanded400-v1/quant.yaml'
    config['safety']=yaml.safe_load(recipe_path.read_text())['safety']
    # Reuse the proven source loader, then replace its selected inputs with the
    # explicit short+long pilot selection. The active INT8 implementation is untouched.
    stamp=utc_stamp(); staging=a.output.with_name('.'+a.output.name+'.incomplete-'+stamp)
    result={'status':'failed','profile':'full' if a.full else 'real_source_short_and_long_pilot','packages':package_versions(),'corpus_sha256':corpus['corpus_sha256'],'sequence_lengths':[len(row['input_ids']) for row in dataset]}
    result['git_commit']=git_revision()
    result['source_revision']=audit['source']['revision']
    result['recipe']={'bits':4,'group_size':128,'symmetric':True,'activation_dtype':'bfloat16','actorder':None,'block_size':128,'dampening_frac':.01,'sequential_targets':['Qwen3_5DecoderLayer']}
    checkpoint_dir=a.output.with_name('.'+a.output.name+'.resume')
    result['checkpoint_dir']=str(checkpoint_dir)
    result['resumed']=a.resume
    result['resource_abort_limits']=resource_abort_limits(config)
    with PeakMonitor(abort_limits=resource_abort_limits(config)) as monitor:
        try:
            model,tokenizer,_,shared=load_real_inputs(config,profile)
            names=[t['name'] for t in audit['targets']]
            modules=dict(model.named_modules())
            for t in audit['targets']:
                assert list(modules[t['name']].weight.shape)==t['shape'],t['name']
            modifier=GPTQModifier(targets=names,scheme='W4A16',block_size=128,dampening_frac=.01,actorder=None)
            identity=run_identity(a.source,dataset,{'recipe':result['recipe'],'targets':names,
                'config':config,'profile':profile,'entrypoint_sha256':file_hash(Path(__file__)),
                'loader_sha256':file_hash(Path(__file__).with_name('quantize.py'))})
            pipeline=register_pipeline(Checkpoints(checkpoint_dir,identity,resume=a.resume))
            oneshot(model=model,processor=tokenizer,dataset=dataset,recipe=[modifier],max_seq_length=profile['max_seq_length'],num_calibration_samples=len(dataset),pipeline=pipeline,sequential_targets=['Qwen3_5DecoderLayer'])
            staging.mkdir(exist_ok=False)
            model.save_pretrained(staging,save_compressed=True,safe_serialization=True,max_shard_size='1GB')
            tokenizer.save_pretrained(staging)
            inject_mtp_tensors(a.source,staging); copy_processor_configs(a.source,staging)
            result['integrity']=validate(a.source,staging,audit)
            assert monitor.samples > 0 and monitor.thread.is_alive(), 'Memory monitor failed'
            if not a.full:
                (staging/'EXPERIMENTAL_NON_PRODUCTION.json').write_text(json.dumps({'profile':result['profile'],'production_authorized':False})+'\n')
            staging.rename(a.output)
            result['status']='passed'
        except KeyboardInterrupt:
            if monitor.safety_trigger:
                result['error']='ResourceSafetyError: '+monitor.safety_trigger
                raise ResourceSafetyError(monitor.safety_trigger) from None
            result['error']='External KeyboardInterrupt'
            raise
        except Exception as exc:
            result['error']=type(exc).__name__+': '+str(exc)
            raise
        finally:
            result['peaks']=monitor.report()
            a.output.parent.mkdir(parents=True,exist_ok=True)
            (a.output.parent/('quant-'+result['profile']+'-'+stamp+'.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
