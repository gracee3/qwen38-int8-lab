#!/usr/bin/env python3
"""Fail closed on INT4 scope, packing, metadata, and preserved source bytes."""
import argparse
import array
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

from inspect_model import inspect_checkpoint, safetensors_header


def inventory(root):
    result={}
    for path in sorted(root.glob('*.safetensors')):
        header,length=safetensors_header(path)
        for name,meta in header.items():
            if name=='__metadata__': continue
            if name in result: raise RuntimeError('duplicate tensor '+name)
            result[name]=(path,8+length,meta)
    return result


def tensor_hash(entry):
    path,base,meta=entry
    start,end=meta['data_offsets']; digest=hashlib.sha256()
    with path.open('rb') as handle:
        handle.seek(base+start); remaining=end-start
        while remaining:
            chunk=handle.read(min(1024**2,remaining))
            if not chunk: raise RuntimeError('truncated tensor '+str(path))
            digest.update(chunk); remaining-=len(chunk)
    return digest.hexdigest()


def tensor_bytes(entry):
    path,base,meta=entry
    start,end=meta['data_offsets']
    with path.open('rb') as handle:
        handle.seek(base+start)
        raw=handle.read(end-start)
    if len(raw)!=end-start: raise RuntimeError('truncated tensor')
    return raw


def validate(source, output, audit):
    inspected,ok=inspect_checkpoint(output,False)
    if not ok: raise RuntimeError(inspected['errors'])
    source_tensors=inventory(source); tensors=inventory(output)
    targets={t['name']:t['shape'] for t in audit['targets']}
    if audit['status']!='passed' or len(targets)!=400: raise RuntimeError('bad audit')
    config=json.loads((output/'config.json').read_text())['quantization_config']
    assert config['quant_method']=='compressed-tensors' and config['format']=='pack-quantized' and config['quantization_status']=='compressed'
    declared=[]
    for group in config['config_groups'].values():
        w=group['weights']
        assert w['num_bits']==4 and w['type']=='int' and w['strategy']=='group' and w['symmetric'] and w['group_size']==128 and w.get('actorder') is None
        assert group.get('input_activations') is None and group.get('output_activations') is None
        declared.extend(group['targets'])
    assert set(declared)==set(targets), 'serialized target scope mismatch'
    expected=set(source_tensors)
    for name,shape in targets.items():
        assert source_tensors[name+'.weight'][2]['shape']==shape
        expected.remove(name+'.weight')
        for suffix in ('.weight_packed','.weight_scale','.weight_shape'):
            expected.add(name+suffix)
            assert name+suffix in tensors, name+suffix
        packed=tensors[name+'.weight_packed'][2]; scales=tensors[name+'.weight_scale'][2]
        assert packed['dtype']=='I32' and packed['shape']==[shape[0],shape[1]//8], name
        assert scales['dtype']=='BF16' and scales['shape']==[shape[0],shape[1]//128], name
        assert list(struct.unpack('<qq',tensor_bytes(tensors[name+'.weight_shape'])))==shape, name
        values=array.array('H'); values.frombytes(tensor_bytes(tensors[name+'.weight_scale']))
        if sys.byteorder!='little': values.byteswap()
        assert all(0<v<0x7f80 for v in values), 'Nonpositive or nonfinite scale: '+name
    assert set(tensors)==expected, {'missing':sorted(expected-set(tensors)), 'extra':sorted(set(tensors)-expected)}
    preserved={}
    for name,entry in source_tensors.items():
        if name.removesuffix('.weight') in targets: continue
        other=tensors[name]
        assert entry[2]['dtype']==other[2]['dtype'] and entry[2]['shape']==other[2]['shape'], name
        digest=tensor_hash(entry)
        assert tensor_hash(other)==digest, name
        preserved[name]=digest
    assert len([n for n in preserved if n.startswith('mtp.')])==15
    for name in ('preprocessor_config.json','video_preprocessor_config.json'):
        assert (source/name).read_bytes()==(output/name).read_bytes(),name
    return {'status':'passed','logical_targets':400,'preserved_sha256':preserved,'shard_bytes':inspected['checkpoint']['shard_bytes']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path); p.add_argument('output',type=Path); p.add_argument('audit',type=Path)
    a=p.parse_args(); print(json.dumps(validate(a.source,a.output,json.loads(a.audit.read_text())),indent=2))
