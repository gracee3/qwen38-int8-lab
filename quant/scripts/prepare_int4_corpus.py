#!/usr/bin/env python3
"""Prepare a separate pinned corpus from existing caches, without overwriting v2.

Writes a candidate only. Held-out overlap review remains a separate required gate.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

SOURCES = [
    ('CoderForge', 'togethercomputer/CoderForge-Preview', '060fca96cf723b2ebab3181e9e59fafd273df3cb', 'togethercomputer___coder_forge-preview/trajectories', 'filtered_reward1', .35),
    ('Nemotron-Terminal', 'nvidia/Nemotron-Terminal-Corpus', 'a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9', 'nvidia___nemotron-terminal-corpus/skill_based_mixed', 'train', .30),
    ('Strandset-Rust', 'Fortytwo-Network/Strandset-Rust-v1', '0a8d223302712a2b34a6ad4ce1fd679031894b3d', 'Fortytwo-Network___strandset-rust-v1/default', 'train', .25),
    ('Nemotron-Agentic', 'nvidia/Nemotron-Agentic-v1', '650d590978ca35c8f1ecea2faf136e5fac421b62', None, 'interactive_agent', .10),
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def messages_for(row):
    raw = decode(row.get('messages') or row.get('conversations'))
    if raw is None and row.get('input_data') and row.get('output_data'):
        raw = [{'role':'user','content':row['input_data']}, {'role':'assistant','content':row['output_data']}]
    if not isinstance(raw,list):
        return []
    if any(not isinstance(m,dict) or m.get('role') not in ('system','user','assistant','tool') or m.get('function_call') for m in raw):
        return []
    # Select known chat fields; separately provided reasoning is never rendered.
    cleaned=[{k:v for k,v in m.items() if k in ('role','content','tool_calls','tool_call_id','name') and v is not None} for m in raw]
    for message in cleaned:
        calls=[]
        for call in message.get('tool_calls',[]):
            function=dict(call.get('function',{}))
            try: arguments=decode(function.get('arguments',{}))
            except (ValueError,TypeError): return []
            if not isinstance(arguments,dict): return []
            function['arguments']=arguments
            calls.append({**call,'function':function})
        if calls: message['tool_calls']=calls
    return cleaned


def coherent_prefixes(messages):
    pending = set()
    for i,m in enumerate(messages):
        if m.get('role')=='tool':
            identity=m.get('tool_call_id')
            if identity not in pending:
                return
            pending.remove(identity)
        for call in m.get('tool_calls',[]):
            identity=call.get('id')
            if not identity or identity in pending:
                return
            pending.add(identity)
        if not pending and m.get('role') in ('assistant','tool'):
            yield messages[:i+1]


def shingles(text):
    words=re.findall(r'\w+',text.lower())
    return {tuple(words[i:i+5]) for i in range(max(0,len(words)-4))}


class JsonRows:
    def __init__(self, path):
        self.path=path
        self.offsets=[]
        with path.open('rb') as handle:
            while True:
                position=handle.tell()
                if not handle.readline(): break
                self.offsets.append(position)
    def __len__(self): return len(self.offsets)
    def __getitem__(self,index):
        with self.path.open('rb') as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


def main():
    from datasets import Dataset, concatenate_datasets
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-rows-per-source',type=int,default=100000)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer=AutoTokenizer.from_pretrained(args.source,local_files_only=True,trust_remote_code=False)
    rows=[]; source_reports=[]; seen=set(); prompts=[]
    for label,repo,revision,cache,split,share in SOURCES:
        print('loading='+label,flush=True)
        snapshot=args.cache/'hub'/('datasets--'+repo.replace('/','--'))/'snapshots'/revision
        card=snapshot/'README.md'
        if cache:
            paths=sorted((args.cache/'datasets'/cache/'0.0.0'/revision).glob('*-'+split+'*.arrow'))
            if not paths: raise RuntimeError(f'No cached Arrow shards: {label}')
            data=concatenate_datasets([Dataset.from_file(str(p)) for p in paths])
        else:
            # Avoid Arrow schema inference across heterogeneous JSON tool arguments.
            data=JsonRows(snapshot/'data/interactive_agent.jsonl')
        order=list(range(len(data))); random.Random(42).shuffle(order)
        # Long allocation is global, not per source. Rust exercises are naturally
        # shorter; allocate their long-token share to coherent agent trajectories.
        long_targets={'CoderForge':210000,'Nemotron-Terminal':150000,'Strandset-Rust':0,'Nemotron-Agentic':15000}
        totals={'short':0,'long':0}; targets={'short':int(1500000*share)-long_targets[label],'long':long_targets[label]}
        accepted=0
        for examined,index in enumerate(order[:args.max_rows_per_source]):
            if examined%250==0: print(f'source={label} examined={examined} accepted={accepted} tokens={totals}',flush=True)
            if all(totals[k]>=targets[k] for k in totals): break
            row=data[index]
            if label=='CoderForge' and row.get('license') not in ('MIT','Apache-2.0','Apache 2.0','BSD-3-Clause'):
                continue
            messages=messages_for(row)
            if not messages: continue
            prompt=next((m.get('content','') for m in messages if m.get('role')=='user'),'')
            if not isinstance(prompt,str): continue
            task_id=str(row.get('trajectory_id') or row.get('uuid') or row.get('task') or digest(prompt))
            identity=repo+':'+task_id
            if identity in seen: continue
            psh=shingles(prompt)
            if psh and any(len(psh & old)/len(psh | old) >= .85 for old in prompts if old): continue
            tools=decode(row.get('tools')) or None
            candidates={}
            for bucket,lo,hi in [('short',2048,4096),('long',8192,16384)]:
                if totals[bucket]>=targets[bucket]: continue
                render_messages=[m for m in messages if m.get('role')!='system'] if bucket=='short' else messages
                render_tools=None if bucket=='short' else tools
                for prefix in coherent_prefixes(render_messages):
                    encoded=tokenizer.apply_chat_template(prefix, tools=render_tools, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=True)
                    tokens=encoded['input_ids']
                    if len(tokens)>hi: break
                    if len(tokens)>=lo: candidates[bucket]=(tokens,prefix,render_tools)
            if not candidates: continue
            bucket=max(candidates,key=lambda k:(targets[k]-totals[k])/targets[k])
            tokens,prefix,render_tools=candidates[bucket]
            rows.append({'input_ids':tokens,'attention_mask':[1]*len(tokens),'source':repo,'revision':revision,'split':split,'source_index':index,'task_id':task_id,'rendered_sha256':digest({'messages':prefix,'tools':render_tools}), 'prompt_sha256':digest(prompt),'bucket':bucket})
            seen.add(identity); prompts.append(psh); totals[bucket]+=len(tokens); accepted+=1
        source_reports.append({'name':label,'repo':repo,'revision':revision,'split':split,'rows':accepted,'tokens':totals,'targets':targets,'dataset_card_sha256':hashlib.sha256(card.read_bytes()).hexdigest()})
        print(json.dumps(source_reports[-1]),flush=True)
        del data
    total=sum(len(row['input_ids']) for row in rows)
    failures=[]
    for source in source_reports:
        for bucket in ('short','long'):
            if source['tokens'][bucket]<source['targets'][bucket]*.90: failures.append(source['name']+':'+bucket+':insufficient_tokens')
    if not 1300000<=total<=1700000: failures.append('total_tokens_out_of_range')
    random.Random(42).shuffle(rows)
    args.output.mkdir(parents=True,exist_ok=False)
    Dataset.from_list(rows).to_parquet(str(args.output/'calibration.parquet'))
    manifest={'status':'candidate_needs_overlap_review' if not failures else 'failed','full_build_authorized':False,'seed':42,'total_tokens':total,'rows':len(rows),'sources':source_reports,'failures':failures,'heldout_overlap':'not_checked','selection':'seeded row order; coherent prefixes; task dedup; user prompt word-5-shingle Jaccard >=0.85 rejected', 'tokenizer_sha256':hashlib.sha256((args.source/'tokenizer.json').read_bytes()).hexdigest(),'corpus_sha256':hashlib.sha256((args.output/'calibration.parquet').read_bytes()).hexdigest()}
    manifest['rendering']={'template':'source tokenizer chat template; enable_thinking=False; no generation prompt', 'short':'omit system messages and tool definitions; preserve remaining message bodies and call-result associations', 'long':'retain system messages and tool definitions', 'tokens':'actual input_ids from BatchEncoding; never padding/truncation'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('status='+manifest['status'],flush=True)
    if failures: raise SystemExit(2)


if __name__=='__main__':
    main()
