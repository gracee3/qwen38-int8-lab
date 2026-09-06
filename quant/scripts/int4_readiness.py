#!/usr/bin/env python3
"""Read-only readiness report. Never launches a build or creates passing gates."""
import argparse
import json
import subprocess
import shutil
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    root=args.run_root
    checks={}
    corpus_hash=None
    for name,file in [('source_audit','target-audit-hashed.json'),('runtime','runtime-result.json'),('corpus','calibration-final/manifest.json')]:
        path=root/file
        if not path.is_file(): checks[name]='missing'; continue
        d=json.loads(path.read_text())
        checks[name]=d.get('status','unknown')
        if name=='source_audit' and not d.get('source',{}).get('shard_hashes_verified'): checks[name]='missing_shard_verification'
        if name=='corpus' and d.get('failures'): checks[name]='failed'
        if name=='corpus': corpus_hash=d.get('corpus_sha256')
    pilots=list(root.glob('quant-real_source_short_and_long_pilot-*.json'))
    checks['real_source_pilot']='not_passed'
    for p in pilots:
        d=json.loads(p.read_text())
        if d.get('status')=='passed' and corpus_hash and d.get('corpus_sha256')==corpus_hash and d.get('peaks',{}).get('samples',0)>0:
            checks['real_source_pilot']='passed'
    checks['heldout_overlap']='not_passed'
    for p in root.glob('overlap-*.json'):
        d=json.loads(p.read_text())
        if d.get('status')=='passed' and corpus_hash and d.get('corpus_sha256')==corpus_hash and d.get('unresolved_overlaps')==0 and d.get('heldout_manifest_sha256'):
            checks['heldout_overlap']='passed'
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    checks['host_ram']='passed' if available>=80*1024**3 else 'requires_80_GiB_available'
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    checks['exclusive_compute']='passed' if not processes else 'another_workload_active'
    checks['disk_headroom']='passed' if shutil.disk_usage(root).free>=64*1024**3 else 'requires_64_GiB_free'
    checks['output_absent']='passed' if not Path('/data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1').exists() else 'output_already_exists'
    # Corpus preparation intentionally produces a candidate until overlap is reviewed.
    passed=all(value=='passed' for key,value in checks.items() if key!='corpus') and checks['corpus']=='candidate_needs_overlap_review'
    print(json.dumps({'ready':passed,'checks':checks,'available_ram_gib':available/2**30,'automatic_launch':False},indent=2))
    return 0 if passed else 2


if __name__=='__main__': raise SystemExit(main())
