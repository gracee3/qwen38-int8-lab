"""Atomic rolling checkpoints for pinned, single-process sequential GPTQ.

Snapshots are internal calibration state, not deployable compressed models.
Only checkpoints created locally by this code with matching identity are loaded.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import uuid


def run_identity(source, dataset, recipe):
    """Bind a resume point to source bytes, actual token rows, recipe and code."""
    import importlib.metadata
    source = Path(source)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()) | {'config.json', 'model.safetensors.index.json'})
    source_hashes = {}
    for name in names:
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source.resolve(strict=True)):
            raise ValueError('Source shard escapes source directory')
        source_hashes[name] = digest(path)
    rows = hashlib.sha256()
    for row in dataset:
        rows.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
        rows.update(b'\n')
    import torch
    return {'source': source_hashes, 'dataset': rows.hexdigest(), 'recipe': recipe,
            'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'adapter': digest(__file__),
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'llmcompressor', 'compressed-tensors')}}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def tensor_bytes(value):
    """Conservative per-tree estimate; repeated references may be counted twice."""
    import dataclasses
    import torch
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    if dataclasses.is_dataclass(value):
        return sum(tensor_bytes(getattr(value, field.name)) for field in dataclasses.fields(value))
    return 0


class Checkpoints:
    def __init__(self, root, identity, resume=False, interval=0.2, stop_after=None):
        if not 0 < interval <= 1:
            raise ValueError('checkpoint interval must be in (0, 1]')
        self.root = Path(root)
        self.identity = identity
        self.resume = resume
        self.interval = interval
        self.stop_after = stop_after  # Test-only fault injection at durable boundary.

    @contextlib.contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError('Checkpoint root must not be a symlink')
        with (self.root / 'writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def manifest(self):
        pointer = self.root / 'current.json'
        if not pointer.exists():
            if self.resume:
                raise FileNotFoundError(f'No resume checkpoint: {pointer}')
            return None
        if not self.resume:
            raise FileExistsError(f'Checkpoint exists; explicitly request resume: {pointer}')
        metadata = json.loads(pointer.read_text())
        if metadata.get('version') != 1:
            raise ValueError('Unsupported checkpoint version')
        if metadata['identity'] != self.identity:
            raise ValueError('Checkpoint identity differs: source, recipe, corpus, or software changed')
        return metadata

    def verified_dir(self, metadata):
        name = metadata['directory']
        if Path(name).name != name or not name.startswith('generation-'):
            raise ValueError('Invalid checkpoint directory')
        directory = self.root / name
        if directory.is_symlink():
            raise ValueError('Checkpoint symlinks are not allowed')
        required = {'model.pt', 'rng.pt'} | {f'batch-{i:06d}.pt' for i in range(metadata['batches'])}
        if set(metadata['files']) != required:
            raise ValueError('Incomplete checkpoint manifest')
        for name, expected in metadata['files'].items():
            if Path(name).name != name or (directory / name).is_symlink():
                raise ValueError('Invalid checkpoint file')
            if digest(directory / name) != expected:
                raise ValueError(f'Checkpoint checksum mismatch: {name}')
        return directory

    def save(self, model, activations, next_index, total):
        import torch
        from compressed_tensors.offload import disable_onloading
        from llmcompressor.pipelines.cache import IntermediateValue
        self.root.mkdir(parents=True, exist_ok=True)
        with disable_onloading():
            state = model.state_dict()
            if any(t.device.type != 'cpu' for t in state.values()):
                raise ValueError('Checkpoint weights must be CPU offloaded')
            # Check disk headroom for weights; activation files are checked per batch.
            size = sum(t.numel() * t.element_size() for t in state.values())
            size += tensor_bytes(activations.batch_intermediates)
            if shutil.disk_usage(self.root).free < size + 8 * 1024**3:
                raise RuntimeError('Insufficient disk for next checkpoint plus 8 GiB reserve')
            directory = self.root / ('generation-' + uuid.uuid4().hex)
            directory.mkdir()
            torch.save(state, directory / 'model.pt')
        del state
        files = ['model.pt']
        # Preserve exact cached intermediates, including intended onload devices.
        for index, batch in enumerate(activations.batch_intermediates):
            name = f'batch-{index:06d}.pt'
            if shutil.disk_usage(self.root).free < 8 * 1024**3:
                raise RuntimeError('Insufficient disk for calibration cache')
            torch.save(batch, directory / name)
            with torch.serialization.safe_globals([IntermediateValue]):
                torch.load(directory / name, weights_only=True, mmap=True)
            files.append(name)
        torch.save({'torch': torch.get_rng_state(), 'python': random.getstate(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                   directory / 'rng.pt')
        files.append('rng.pt')
        for name in files:
            with (directory / name).open('rb') as stream:
                os.fsync(stream.fileno())
        metadata = {'version': 1, 'identity': self.identity, 'directory': directory.name,
                    'next_index': next_index, 'total': total, 'batches': len(activations),
                    'files': {name: digest(directory / name) for name in files}}
        self.verified_dir(metadata)
        pointer = self.root / 'current.json'
        previous = json.loads(pointer.read_text()) if pointer.exists() else None
        temporary = self.root / 'current.json.tmp'
        with temporary.open('w') as stream:
            json.dump(metadata, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        # Persist the generation before publishing its pointer.
        for path in (directory,):
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.replace(temporary, pointer)
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        print(f'CHECKPOINT_SAVED completed_stages={next_index}/{total} path={directory}', flush=True)
        if previous:
            old = self.verified_dir(previous)
            if old != directory:
                shutil.rmtree(old)
        if self.stop_after == next_index:
            raise RuntimeError('TEST_CHECKPOINT_INTERRUPT')

    def load(self, model, metadata, offload_device):
        import torch
        from compressed_tensors.offload import disable_onloading
        from llmcompressor.pipelines.cache import IntermediatesCache, IntermediateValue
        directory = self.verified_dir(metadata)
        state = torch.load(directory / 'model.pt', weights_only=True, mmap=True)
        with disable_onloading():
            # Observer/qparam shapes may be initialized lazily. Restore registered
            # tensors directly, avoiding a second full CPU allocation.
            for name, value in state.items():
                parent, _, attr = name.rpartition('.')
                module = model.get_submodule(parent) if parent else model
                if attr in module._parameters:
                    module._parameters[attr] = torch.nn.Parameter(value.clone(), requires_grad=False)
                elif attr in module._buffers:
                    module._buffers[attr] = value.clone()
                else:
                    raise ValueError(f'Unknown checkpoint tensor: {name}')
        with torch.serialization.safe_globals([IntermediateValue]):
            batches = [torch.load(directory / f'batch-{i:06d}.pt', weights_only=True)
                       for i in range(metadata['batches'])]
        rng = torch.load(directory / 'rng.pt', weights_only=True)
        torch.set_rng_state(rng['torch'])
        random.setstate(rng['python'])
        if rng['cuda']:
            torch.cuda.set_rng_state_all(rng['cuda'])
        print(f'CHECKPOINT_RESUMED next_stage={metadata["next_index"] + 1}', flush=True)
        return IntermediatesCache(batches, offload_device)


def register_pipeline(checkpoints):
    """Pinned adapter matching llmcompressor 0.13.0's sequential GPTQ loop."""
    import importlib.metadata
    if importlib.metadata.version('llmcompressor') != '0.13.0':
        raise RuntimeError('Resume adapter requires llmcompressor 0.13.0')
    import torch
    from compressed_tensors.offload import disable_offloading, set_onload_device
    from llmcompressor.core import LifecycleCallbacks, active_session
    from llmcompressor.modifiers.utils.hooks import HooksMixin
    from llmcompressor.pipelines.cache import IntermediatesCache
    from llmcompressor.pipelines.registry import CalibrationPipeline
    from llmcompressor.pipelines.sequential.pipeline import _get_batches, trace_subgraphs
    from llmcompressor.utils.dev import get_main_device
    from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
    from llmcompressor.utils.pytorch.module import infer_sequential_targets

    @CalibrationPipeline.register('resumable_gptq')
    class ResumableGPTQ(CalibrationPipeline):
        @staticmethod
        def __call__(model, dataloader, dataset_args):
            with checkpoints.locked():
                return ResumableGPTQ.run(model, dataloader, dataset_args)

        @staticmethod
        def run(model, dataloader, dataset_args):
            session = active_session()
            modifiers = session.lifecycle.recipe.modifiers
            if len(modifiers) != 1 or type(modifiers[0]).__name__ != 'GPTQModifier':
                raise ValueError('Resume supports one GPTQModifier only')
            if torch.distributed.is_initialized() or getattr(dataset_args, 'use_loss_mask', False):
                raise ValueError('Distributed/masked resume is unsupported')
            if not dataset_args.propagate_error:
                raise ValueError('Resume requires propagated quantization error')
            metadata = checkpoints.manifest()
            onload = get_main_device()
            offload = torch.device(dataset_args.sequential_offload_device)
            set_onload_device(model, onload)
            subgraphs = trace_subgraphs(model, next(iter(dataloader)),
                infer_sequential_targets(model, dataset_args.sequential_targets),
                dataset_args.tracing_ignore, dataset_args.sequential_targets_per_subgraph)
            total = len(subgraphs)
            if metadata and (metadata['total'] != total or metadata['batches'] != len(dataloader)
                             or not 0 < metadata['next_index'] <= total):
                raise ValueError('Resume graph or batch count differs')
            LifecycleCallbacks.calibration_start()
            with calibration_forward_context(model), DisableQuantization(model):
                activations = checkpoints.load(model, metadata, offload) if metadata else \
                    IntermediatesCache.from_dataloader(dataloader, onload, offload)
                start = metadata['next_index'] if metadata else 0
                session.state.loss_masks = None
                session.state.sequential_prefetch = False
                every = max(1, math.ceil(total * checkpoints.interval))
                for index in range(start, total):
                    subgraph = subgraphs[index]
                    with disable_offloading():
                        for batch_index, inputs in _get_batches(activations, len(dataloader),
                                subgraph.input_names, f'({index+1}/{total}): Calibrating'):
                            session.state.current_batch_idx = batch_index
                            subgraph.forward(model, **inputs)
                        LifecycleCallbacks.sequential_epoch_end(subgraph.submodules(model))
                        with HooksMixin.disable_hooks():
                            for batch_index, inputs in _get_batches(activations, len(dataloader),
                                    subgraph.input_names, f'({index+1}/{total}): Propagating'):
                                outputs = subgraph.forward(model, **inputs)
                                if index < total - 1:
                                    activations.update(batch_index, outputs)
                                    activations.delete(batch_index, subgraph.consumed_names)
                    if (index + 1) % every == 0 or index == total - 1:
                        if modifiers[0]._hessians or modifiers[0]._num_samples:
                            raise RuntimeError('Cannot checkpoint unfinished GPTQ statistics')
                        checkpoints.save(model, activations, index + 1, total)
                LifecycleCallbacks.calibration_end()
    return 'resumable_gptq'
