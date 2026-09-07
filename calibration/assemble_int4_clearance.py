#!/usr/bin/env python3
"""Assemble matching completed overlap screens; never replace missing evidence."""
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    root=Path('/run-int4'); out=root/'overlap-clearance.json'
    if out.exists(): raise FileExistsError(out)
    manifest=json.loads((root/'calibration-final/manifest.json').read_text())
    fingerprint=hashlib.sha256((root/'calibration-final/calibration.parquet').read_bytes()).hexdigest()
    assert fingerprint==manifest['corpus_sha256']
    reports={}
    for name in ('overlap-final-public.json','overlap-final-agents.json','origin-final-schema.json'):
        d=json.loads((root/name).read_text())
        assert d['corpus_sha256']==fingerprint and not d['matches'],name
        reports[name]=d
    public=reports['overlap-final-public.json']; agents=reports['overlap-final-agents.json']; origin=reports['origin-final-schema.json']
    assert public['status']=='public_suites_clear_canaries_pending' and public['screened_cases']==861
    assert agents['status']=='canary_lexical_screen_clear'
    assert Counter(case['suite'] for case in agents['cases'])=={'Multi-SWE-bench':10,'Terminal-Bench':10}
    assert origin['status']=='identity_screen_clear' and len(origin['origins'])==manifest['rows']
    identities={'public':public['sources'],'agents':agents['cases']}
    result={'status':'passed','corpus_sha256':fingerprint,'heldout_manifest_sha256':hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),'unresolved_overlaps':0,'screened_cases':881,'selected_calibration_rows':manifest['rows'],'excluded_task_ids':manifest['excluded_task_ids'],'evidence_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in reports},'scope':'exact text, normalized task identities and >=50% word-5-shingle containment (>=8 shared shingles) of prompts, available canonical code/tests and repository patches; reviewed origin metadata; final selection follows frozen fixtures','limitations':['A negative lexical/identity screen cannot guarantee absence of every semantic transformation.','Benchmark environments have not been executed; suite execution prerequisites remain an evaluation gate.']}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','rows':manifest['rows'],'cases':881,'tokens':manifest['total_tokens']}),flush=True)


if __name__=='__main__': main()
