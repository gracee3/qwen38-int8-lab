#!/usr/bin/env python3
"""Install a fail-closed revision policy around Hugging Face dataset loading."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def install(pins: dict[str, str], *, offline: bool) -> Callable[..., Any]:
    import datasets

    original = datasets.load_dataset

    def pinned_load_dataset(*args: Any, **kwargs: Any) -> Any:
        path = kwargs.get("path") or (args[0] if args else None)
        if path not in pins:
            raise RuntimeError(f"unapproved dataset requested: {path!r}")
        requested = kwargs.get("revision")
        expected = pins[path]
        if requested not in (None, expected):
            raise RuntimeError(
                f"dataset revision mismatch for {path}: expected {expected}, got {requested}"
            )
        kwargs["revision"] = expected
        if offline:
            kwargs.setdefault("download_mode", datasets.DownloadMode.REUSE_DATASET_IF_EXISTS)
        return original(*args, **kwargs)

    datasets.load_dataset = pinned_load_dataset
    return original
