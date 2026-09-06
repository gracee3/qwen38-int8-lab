#!/usr/bin/env python3
"""Screen pinned public held-outs; never claim pending canary review has passed."""
import hashlib
import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

SOURCES=[
    ('MultiPL-E-Rust','nuprl/MultiPL-E','28441b6024e71d4a1c1c0f6bf171c935cd5a43f2','humaneval-rs/test-00000-of-00001.parquet'),
    ('HumanEval+','evalplus/humanevalplus','d32357cf319e50e9c8d8dab5ea876c72b0fd321b','test.jsonl'),
]


def sha(raw): return hashlib.sha256(raw).hexdigest()


def shingles(text):
    words=re.findall(r'\w+',text.lower())
    return {tuple(words[i:i+5]) for i in range(max(0,len(words)-4))}


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',default='calibration-candidate')
    parser.add_argument('--output',default='overlap-public-suites.json')
    args=parser.parse_args()
    root=Path('/run-int4'); output=root/args.output
    if output.exists(): raise FileExistsError(output)
    frozen=root/'heldout-public'; frozen.mkdir(exist_ok=True)
    cases=[]; sources=[]
    for label,repo,revision,name in SOURCES:
        path=frozen/(label+Path(name).suffix)
        if not path.exists():
            url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
            with urllib.request.urlopen(url,timeout=60) as response:
                data=response.read(64*1024**2+1)
            if len(data)>64*1024**2: raise RuntimeError('held-out download exceeds 64 MiB cap')
            with path.open('xb') as handle: handle.write(data)
        raw=path.read_bytes()
        rows=Dataset.from_parquet(str(path)) if path.suffix=='.parquet' else [json.loads(line) for line in raw.splitlines() if line]
        for i,row in enumerate(rows):
            cases.append({'suite':label,'id':str(row.get('task_id') or row.get('name') or i),'prompt':row['prompt'],'code':row.get('canonical_solution') or row.get('tests') or ''})
        sources.append({'suite':label,'repo':repo,'revision':revision,'file':name,'sha256':sha(raw),'count':len(rows)})
    ifeval=Dataset.from_file('/heldout-ifeval/instruction-following-eval-train.arrow')
    order=list(range(len(ifeval))); random.Random(42).shuffle(order)
    chosen=order[:100]
    # Screen all 541 available prompts, a conservative superset of the fixed 100.
    for i,row in enumerate(ifeval): cases.append({'suite':'IFEval','id':str(row.get('key',i)),'prompt':row['prompt'],'code':''})
    sources.append({'suite':'IFEval','repo':'wis-k/instruction-following-eval','revision':'5a5661c2a35488308556cf4453dc074d1eba91a0','screened':len(ifeval),'fixed_subset_indices':chosen})
    tokenizer=AutoTokenizer.from_pretrained('/models/source',local_files_only=True)
    corpus=Dataset.from_parquet(str(root/args.corpus/'calibration.parquet'))
    manifest=json.loads((root/args.corpus/'manifest.json').read_text())
    indexed=[(case,field,shingles(case[field])) for case in cases for field in ('prompt','code') if case[field]]
    matches=[]
    for row in corpus:
        text=tokenizer.decode(row['input_ids']); lowered=text.lower(); grams=shingles(text)
        for case,field,heldout in indexed:
            common=len(grams & heldout)
            exact=len(case[field].strip())>=40 and case[field].strip().lower() in lowered
            containment=common/max(1,len(heldout))
            if exact or (common>=8 and containment>=.5):
                matches.append({'source':row['source'],'task_id':row['task_id'],'suite':case['suite'],'heldout_id':case['id'],'field':field,'exact_text':exact,'shingle_containment':containment})
    status='public_suite_matches_need_review' if matches else 'public_suites_clear_canaries_pending'
    report={'status':status,'corpus_sha256':manifest['corpus_sha256'],'sources':sources,'heldout_manifest_sha256':sha(json.dumps(sources,sort_keys=True).encode()),'screened_cases':len(cases),'matches':matches,'unresolved_overlaps':len(matches),'pending':['freeze runnable 10 repository + 10 terminal canary fixtures','review transformed-task and code overlap'],'method':'exact prompt containment or >=8 shared normalized word-5-shingles covering >=50% of heldout prompt'}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':status,'cases':len(cases),'matches':len(matches)}),flush=True)


if __name__=='__main__': main()
