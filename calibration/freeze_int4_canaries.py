#!/usr/bin/env python3
"""Freeze public 10+10 canary definitions and screen the corpus; no task execution."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import hashlib
import argparse
import json
import random
import urllib.request
from pathlib import Path

from calibration.screen_int4_overlap import shingles

SWE_REV='56ff018c04a38e27ada1e9d0a6d5839a51f88f0d'
TB_REV='2fd12b88aafdd04a52c298e3940bcb189f9766d6'
TASKS=['cancel-async-tasks','db-wal-recovery','fix-git','git-multibranch','large-scale-text-editing','log-summary-date-ranges','multi-source-data-merger','nginx-request-logging','openssl-selfsigned-cert','regex-log']


def download(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response: raw=response.read(32*1024**2+1)
        if len(raw)>32*1024**2: raise RuntimeError('Download exceeds 32 MiB cap')
        with path.open('xb') as handle: handle.write(raw)
    return path.read_bytes()


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',default='calibration-candidate')
    parser.add_argument('--output',default='overlap-agent-canaries.json')
    args=parser.parse_args()
    root=Path('/run-int4'); frozen=root/'heldout-agents'; frozen.mkdir(exist_ok=True)
    output=root/args.output
    if output.exists(): raise FileExistsError(output)
    cases=[]; records=[]
    for file in ['rust/BurntSushi__ripgrep_dataset.jsonl','go/grpc__grpc-go_dataset.jsonl']:
        url=f'https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench/resolve/{SWE_REV}/{file}'
        raw=download(url,frozen/Path(file).name)
        rows=[json.loads(line) for line in raw.splitlines() if line]
        rows=[row for row in rows if row.get('f2p_tests') and row.get('fix_patch') and row.get('test_patch')]
        rows.sort(key=lambda r:r['instance_id']); random.Random(42).shuffle(rows)
        if len(rows)<5: raise RuntimeError('Insufficient executable canary definitions: '+file)
        for row in rows[:5]:
            problem='\n'.join(str(issue.get('title',''))+'\n'+str(issue.get('body','')) for issue in row.get('resolved_issues',[]) if isinstance(issue,dict))
            if not problem.strip(): problem=row['title']+'\n'+row['body']
            cases.append({'suite':'Multi-SWE-bench','id':row['instance_id'],'prompt':problem,'code':row['fix_patch']+'\n'+row['test_patch']})
            records.append({'suite':'Multi-SWE-bench','id':row['instance_id'],'repo':row['org']+'/'+row['repo'],'base':row['base'],'dataset_revision':SWE_REV,'source_file':file,'source_sha256':hashlib.sha256(raw).hexdigest()})
    for task in TASKS:
        base=f'https://raw.githubusercontent.com/harbor-framework/terminal-bench-2/{TB_REV}/{task}'
        raw=download(base+'/instruction.md',frozen/(task+'-instruction.md'))
        config=download(base+'/task.toml',frozen/(task+'-task.toml'))
        cases.append({'suite':'Terminal-Bench','id':task,'prompt':raw.decode(),'code':''})
        records.append({'suite':'Terminal-Bench','id':task,'revision':TB_REV,'instruction_sha256':hashlib.sha256(raw).hexdigest(),'task_config_sha256':hashlib.sha256(config).hexdigest()})
    assert len(cases)==20
    tokenizer=AutoTokenizer.from_pretrained('/models/source',local_files_only=True)
    corpus=Dataset.from_parquet(str(root/args.corpus/'calibration.parquet'))
    matches=[]
    for row in corpus:
        text=tokenizer.decode(row['input_ids']); grams=shingles(text)
        for case in cases:
            for field in ('prompt','code'):
                heldout=shingles(case[field])
                common=len(grams & heldout); containment=common/max(1,len(heldout))
                if common>=8 and containment>=.5:
                    matches.append({'suite':case['suite'],'id':case['id'],'source_task':row['task_id'],'field':field,'shingle_containment':containment})
    manifest=json.loads((root/args.corpus/'manifest.json').read_text())
    report={'status':'canary_lexical_matches_need_review' if matches else 'canary_lexical_screen_clear','corpus_sha256':manifest['corpus_sha256'],'cases':records,'matches':matches,'scope':'task-definition and lexical prompt/patch screen; environments and transformed-task provenance still require review','environments_executed':False}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(cases),'matches':len(matches)}),flush=True)


if __name__=='__main__': main()
