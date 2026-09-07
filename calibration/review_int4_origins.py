#!/usr/bin/env python3
"""Revisit selected upstream identities for repository/terminal overlap review."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
import argparse
import re
from pathlib import Path

from calibration.prepare_int4_corpus import SOURCES, JsonRows, messages_for


def normal(text): return re.sub('[^a-z0-9]','',text.lower())


def main():
    from datasets import Dataset, concatenate_datasets
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',default='calibration-candidate')
    parser.add_argument('--output',default='origin-review.json')
    args=parser.parse_args()
    root=Path('/run-int4'); cache=Path('/hf')
    corpus=Dataset.from_parquet(str(root/args.corpus/'calibration.parquet'))
    heldout=json.loads((root/'overlap-agent-canaries.json').read_text())
    needles={case['id']:normal(case['id']) for case in heldout['cases']}
    records=[]; matches=[]
    for label,repo,revision,relative,split,share in SOURCES:
        selected=[r for r in corpus if r['source']==repo]
        if relative:
            paths=sorted((cache/'datasets'/relative/'0.0.0'/revision).glob('*-'+split+'*.arrow'))
            data=concatenate_datasets([Dataset.from_file(str(p)) for p in paths])
        else:
            data=JsonRows(cache/'hub'/('datasets--'+repo.replace('/','--'))/'snapshots'/revision/'data/interactive_agent.jsonl')
        for selected_row in selected:
            row=data[selected_row['source_index']]
            if not messages_for(row):
                matches.append({'source':repo,'task_id':selected_row['task_id'],'reason':'unsupported_chat_schema'})
            origin={key:row[key] for key in ('trajectory_id','image','crate_name','task','episode','trial_name','uuid') if key in row}
            text=normal(json.dumps(row,sort_keys=True))
            hits=[name for name,value in needles.items() if value in text]
            if hits: matches.append({'source':repo,'task_id':selected_row['task_id'],'heldout_ids':hits})
            records.append({'source':repo,'source_index':selected_row['source_index'],'origin':origin})
    report={'status':'identity_matches_need_review' if matches else 'identity_screen_clear','matches':matches,'origins':records,'method':'normalized heldout task IDs searched in complete selected upstream rows; original metadata retained for review'}
    report['corpus_sha256']=json.loads((root/args.corpus/'manifest.json').read_text())['corpus_sha256']
    (root/args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'matches':len(matches),'rows':len(records)}),flush=True)


if __name__=='__main__': main()
