"""Versioned serving catalog. Import from a pinned checkout; no Docker side effects."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_KEYS = {
    'tensor_parallel_size', 'pipeline_parallel_size', 'max_model_len', 'dtype',
    'kv_cache_dtype', 'kv_cache_memory_bytes', 'cpu_offload_gb', 'enforce_eager',
    'language_model_only', 'max_num_batched_tokens', 'max_num_seqs',
    'enable_prefix_caching', 'enable_chunked_prefill', 'speculative_decoding',
}
PROFILE_KEYS = {'schema_version', 'id', 'model_id', 'recipe', 'checkpoint_name',
                'image', 'hardware', 'runtime', 'generation', 'server', 'environment', 'evidence'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{label} requires exactly {sorted(keys)}')


def positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f'{label} must be a positive integer')


def validate(profile):
    exact_keys(profile, PROFILE_KEYS, 'profile')
    if type(profile['schema_version']) is not int or profile['schema_version'] != 1:
        raise ValueError('unsupported profile schema_version')
    for key in ('id', 'model_id', 'recipe', 'checkpoint_name', 'image'):
        if not isinstance(profile[key], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', profile[key]):
            raise ValueError(f'invalid {key}')
    h = profile['hardware']
    exact_keys(h, {'gpu_count', 'gpu_memory_gib', 'compute_capability'}, 'hardware')
    positive(h['gpu_count'], 'gpu_count')
    positive(h['gpu_memory_gib'], 'gpu_memory_gib')
    if h['compute_capability'] != '8.6':
        raise ValueError('v1 profiles target the recorded SM86 hardware')
    r = profile['runtime']
    exact_keys(r, RUNTIME_KEYS, 'runtime')
    for key in ('tensor_parallel_size', 'pipeline_parallel_size', 'max_model_len',
                'kv_cache_memory_bytes', 'max_num_batched_tokens', 'max_num_seqs'):
        positive(r[key], key)
    for key in ('enforce_eager', 'language_model_only', 'enable_prefix_caching',
                'enable_chunked_prefill', 'speculative_decoding'):
        if type(r[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if r['dtype'] != 'bfloat16' or r['kv_cache_dtype'] not in ('bfloat16', 'fp8'):
        raise ValueError('unsupported dtype')
    if r['speculative_decoding'] or r['cpu_offload_gb'] != 0 or type(r['cpu_offload_gb']) is not int:
        raise ValueError('v1 requires no speculation and no CPU offload')
    if not r['language_model_only'] or r['max_model_len'] > 262144 or r['max_num_seqs'] != 1:
        raise ValueError('v1 requires text-only, one sequence, context <= 262144')
    if r['tensor_parallel_size'] * r['pipeline_parallel_size'] != h['gpu_count']:
        raise ValueError('parallelism must match hardware.gpu_count')
    g = profile['generation']
    exact_keys(g, {'seed', 'enable_thinking', 'generation_config'}, 'generation')
    if type(g['seed']) is not int or not 0 <= g['seed'] < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if g['enable_thinking'] is not False or g['generation_config'] != 'vllm':
        raise ValueError('v1 requires non-thinking and vllm generation defaults')
    s = profile['server']
    exact_keys(s, {'served_model_name', 'enable_auto_tool_choice', 'tool_call_parser', 'request_logging'}, 'server')
    if not isinstance(s['served_model_name'], str) or not s['served_model_name']:
        raise ValueError('served_model_name must be nonempty')
    if type(s['enable_auto_tool_choice']) is not bool or type(s['request_logging']) is not bool:
        raise ValueError('server switches must be boolean')
    if s['tool_call_parser'] != 'qwen3_xml':
        raise ValueError('unsupported tool parser')
    if profile['environment'] != {'VLLM_USE_FLASHINFER_SAMPLER': '0'}:
        raise ValueError('unsupported environment settings')
    e = profile['evidence']
    exact_keys(e, {'status', 'reports', 'origin', 'required_log_evidence'}, 'evidence')
    if not isinstance(e['status'], str) or not isinstance(e['reports'], list):
        raise ValueError('invalid evidence')
    exact_keys(e['origin'], {'repository', 'commit', 'path'}, 'evidence.origin')
    if not re.fullmatch('[0-9a-f]{40}', e['origin']['commit']):
        raise ValueError('evidence origin requires an exact commit')
    if not isinstance(e['required_log_evidence'], dict):
        raise ValueError('invalid required_log_evidence')
    return profile


def discover(root=ROOT):
    """Return profiles indexed by declared ID, rejecting duplicates/unknown settings."""
    root = Path(root)
    result = {}
    images = json.loads((root / 'environments/images.lock.json').read_text())['images']
    for path in sorted((root / 'serving/profiles').glob('*.yaml')):
        p = validate(yaml.safe_load(path.read_text()))
        if p['id'] in result:
            raise ValueError(f'duplicate profile: {p["id"]}')
        if path.stem != p['id']:
            raise ValueError('profile filename must match its declared ID')
        if p['image'] not in images:
            raise ValueError('unknown image identity')
        recipe = root / 'recipes' / p['recipe'] / 'manifest.json'
        manifest = json.loads(recipe.read_text())
        if (manifest['model_id'], manifest['checkpoint_name']) != (p['model_id'], p['checkpoint_name']):
            raise ValueError('profile is incompatible with recipe model identity')
        for report in p['evidence']['reports']:
            target = (root / report).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                raise ValueError(f'missing or external evidence: {report}')
        result[p['id']] = p
    if not result:
        raise ValueError('no serving profiles found')
    return result


def resolve(profile_id, *, root=ROOT, model_id=None, model_path=None,
            gpu_devices=None, overrides=None):
    """Resolve bindings and explicit runtime overrides into JSON-serializable data.

    Consumers pin this checkout, reject source.dirty and freeze the whole result.
    Backend-specific benchmark generation and image overlays stay in the consumer.
    """
    root = Path(root).resolve()
    profiles = discover(root)
    if profile_id not in profiles:
        raise ValueError(f'unknown profile: {profile_id}')
    p = copy.deepcopy(profiles[profile_id])
    if model_id is not None and model_id != p['model_id']:
        raise ValueError(f'{profile_id} is incompatible with model {model_id}')
    overrides = dict(overrides or {})
    # Freeze the performance envelope; changing parallelism needs a separate profile.
    allowed = {'max_model_len', 'kv_cache_memory_bytes', 'max_num_batched_tokens'}
    if set(overrides) - allowed:
        raise ValueError(f'runtime overrides support only {sorted(allowed)}')
    p['runtime'].update(overrides)
    validate(p)
    devices = list(gpu_devices) if gpu_devices is not None else [str(i) for i in range(p['hardware']['gpu_count'])]
    if len(devices) != p['hardware']['gpu_count'] or len(set(devices)) != len(devices):
        raise ValueError('GPU bindings must be distinct and match the profile GPU count')
    if any(not isinstance(d, str) or not re.fullmatch(r'(?:[0-9]+|GPU-[0-9a-fA-F-]+)', d) for d in devices):
        raise ValueError('GPU bindings must be indices or UUIDs')
    model = Path(model_path or Path('/data/models') / p['checkpoint_name']).expanduser().absolute()
    if ',' in str(model) or '\n' in str(model):
        raise ValueError('unsupported model mount path')
    lock = root / 'environments/images.lock.json'
    image = json.loads(lock.read_text())['images'][p['image']]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip())
    profile_path = 'serving/profiles/' + profile_id + '.yaml'
    return {'schema_version': 1, 'profile': p, 'overrides': overrides,
            'bindings': {'model_path': str(model), 'gpu_devices': devices}, 'image': image,
            'source': {'commit': commit, 'dirty': dirty, 'profile_path': profile_path,
                       'profile_sha256': sha256(root / profile_path),
                       'recipe_manifest_sha256': sha256(root / 'recipes' / p['recipe'] / 'manifest.json'),
                       'images_lock_sha256': sha256(lock)}}


def vllm_args(resolved):
    """vLLM engine kwargs common to HTTP serving and an in-process evaluator."""
    r = dict(resolved['profile']['runtime'])
    r.pop('speculative_decoding')  # Disabled by omission in vLLM 0.27.1.
    return r | {'seed': resolved['profile']['generation']['seed']}


def server_args(resolved):
    p = resolved['profile']
    args = ['/model']
    for key, value in vllm_args(resolved).items():
        flag = '--' + key.replace('_', '-')
        if type(value) is bool:
            if value:
                args.append(flag)
            elif key in ('enable_prefix_caching', 'enable_chunked_prefill'):
                args.append('--no-' + key.replace('_', '-'))
        else:
            args += [flag, str(value)]
    s = p['server']
    args += ['--served-model-name', s['served_model_name'], '--host', '0.0.0.0', '--port', '8000',
             '--generation-config', p['generation']['generation_config'],
             '--default-chat-template-kwargs', json.dumps({'enable_thinking': p['generation']['enable_thinking']})]
    if s['enable_auto_tool_choice']:
        args += ['--enable-auto-tool-choice', '--tool-call-parser', s['tool_call_parser']]
    if not s['request_logging']:
        args += ['--no-enable-log-requests', '--disable-uvicorn-access-log']
    return args
