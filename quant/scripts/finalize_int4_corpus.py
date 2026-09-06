#!/usr/bin/env python3
"""Make final selection after heldouts are frozen, excluding all flagged tasks."""
import hashlib
import json
from pathlib import Path


def main():
    from datasets import Dataset
    root=Path('/run-int4'); source=root/'calibration-candidate'; output=root/'calibration-final'
    if output.exists(): raise FileExistsError(output)
    origin=json.loads((root/'origin-review.json').read_text())
    agents=json.loads((root/'overlap-agent-canaries.json').read_text())
    public=json.loads((root/'overlap-public-suites.json').read_text())
    excluded={m['task_id'] for m in origin['matches']}
    excluded.update(m['source_task'] for m in agents['matches'])
    excluded.update(m['task_id'] for m in public['matches'])
    dataset=Dataset.from_parquet(str(source/'calibration.parquet'))
    rows=[row for row in dataset if row['task_id'] not in excluded]
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['candidate_corpus_sha256']=manifest.pop('corpus_sha256')
    for entry in manifest['sources']:
        selected=[r for r in rows if r['source']==entry['repo']]
        entry['rows']=len(selected)
        entry['tokens']={bucket:sum(len(r['input_ids']) for r in selected if r['bucket']==bucket) for bucket in ('short','long')}
    manifest['total_tokens']=sum(len(r['input_ids']) for r in rows)
    manifest['rows']=len(rows)
    assert 1300000<=manifest['total_tokens']<=1700000
    for entry,share in zip(manifest['sources'],(.35,.30,.25,.10)):
        assert abs(sum(entry['tokens'].values())/manifest['total_tokens']-share)<=.03
    output.mkdir(exist_ok=False)
    Dataset.from_list(rows).to_parquet(str(output/'calibration.parquet'))
    manifest['corpus_sha256']=hashlib.sha256((output/'calibration.parquet').read_bytes()).hexdigest()
    manifest['excluded_task_ids']=sorted(excluded)
    manifest['final_selection']='after freezing public suite and agent canary definitions; conservative removal of all screening flags'
    manifest['heldout_overlap']='final rescreen required'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'final_selection_needs_rescreen','rows':len(rows),'tokens':manifest['total_tokens'],'excluded_tasks':len(excluded)}),flush=True)


if __name__=='__main__': main()
