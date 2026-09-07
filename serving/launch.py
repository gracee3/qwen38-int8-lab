#!/usr/bin/env python3
"""Discover, resolve, preview, or serve a profile using existing local images."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving.catalog import ROOT, discover, resolve, server_args


def docker_command(resolved, *, work_root, cuda_toolkit, port):
    if not 1 <= port <= 65535:
        raise ValueError('port must be between 1 and 65535')
    for path in (work_root, cuda_toolkit):
        if not Path(path).is_absolute() or ',' in str(path) or '\n' in str(path):
            raise ValueError('mount bindings must be absolute paths without commas/newlines')
    b = resolved['bindings']
    devices = 'device=' + ','.join(b['gpu_devices'])
    if len(b['gpu_devices']) > 1:
        devices = '"' + devices + '"'  # Docker parses the --gpus value as CSV.
    args = ['docker', 'run', '--rm', '--pull', 'never', '--gpus', devices, '--ipc=host',
            '-p', f'127.0.0.1:{port}:8000', '--mount', f'type=bind,src={b["model_path"]},dst=/model,readonly',
            '--mount', f'type=bind,src={work_root},dst=/work',
            '--env', 'VLLM_API_KEY', '--env', 'VLLM_CACHE_ROOT=/work/cache/vllm']
    for key, value in resolved['profile']['environment'].items():
        args += ['--env', key + '=' + value]
    if resolved['profile']['runtime']['kv_cache_dtype'] == 'fp8':
        args += ['--mount', f'type=bind,src={cuda_toolkit},dst=/usr/local/cuda,readonly',
                 '--env', 'CUDA_HOME=/usr/local/cuda']
    return args + ['--entrypoint', 'vllm', resolved['image']['local_image_id'], 'serve'] + server_args(resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list')
    listing.add_argument('--json', action='store_true')
    for name in ('resolve', 'serve'):
        p = sub.add_parser(name)
        p.add_argument('profile')
        p.add_argument('--model', default=os.environ.get('MODEL_PATH'))
        p.add_argument('--model-id')
        p.add_argument('--gpus', default=os.environ.get('GPU_DEVICES'))
        p.add_argument('--override', action='append', default=[], metavar='KEY=JSON')
        if name == 'serve':
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8000')))
            p.add_argument('--work-root', default=os.environ.get('WORK_ROOT', '/data/qwen38-int8-lab'))
            p.add_argument('--cuda-toolkit', default=os.environ.get('CUDA_TOOLKIT_ROOT', '/usr/local/cuda-13.3'))
    a = parser.parse_args()
    try:
        if a.command == 'list':
            profiles = discover()
            if a.json:
                print(json.dumps(profiles, indent=2))
            else:
                for key, p in profiles.items():
                    print(f'{key}\t{p["model_id"]}\t{p["evidence"]["status"]}')
            return
        overrides = {}
        for item in a.override:
            key, value = item.split('=', 1)
            if key in overrides:
                raise ValueError(f'duplicate override: {key}')
            overrides[key] = json.loads(value)
        resolved = resolve(a.profile, model_id=a.model_id, model_path=a.model,
                           gpu_devices=a.gpus.split(',') if a.gpus is not None else None,
                           overrides=overrides)
        if a.command == 'resolve':
            print(json.dumps(resolved, indent=2))
            return
        command = docker_command(resolved, work_root=a.work_root, cuda_toolkit=a.cuda_toolkit, port=a.port)
        if a.dry_run:
            print(shlex.join(command))
            return
        if not os.environ.get('VLLM_API_KEY'):
            raise ValueError('set VLLM_API_KEY in the environment before serving')
        if not (Path(resolved['bindings']['model_path']) / 'config.json').is_file():
            raise ValueError('model binding must contain config.json')
        if not Path(a.work_root).is_dir():
            raise ValueError('work root must already exist')
        if resolved['profile']['runtime']['kv_cache_dtype'] == 'fp8' and not (Path(a.cuda_toolkit) / 'bin/nvcc').is_file():
            raise ValueError('FP8 profiles require the existing CUDA toolkit with nvcc')
        # Exact local content ID; no pull, retag, or build fallback.
        subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', resolved['image']['local_image_id']],
                       check=True, stdout=subprocess.DEVNULL)
        raise SystemExit(subprocess.call(command))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
