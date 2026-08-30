#!/usr/bin/env python3
"""Shared, content-safe helpers for the standardized accuracy workflow."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path("/app/eval/config/leaderboard-v2.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    selected = Path(path or os.environ.get("EVAL_SUITE_CONFIG", CONFIG_PATH))
    with selected.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected = {
        "version": "0.4.12",
        "git_revision": "6d642546f4688648fced259eb3302efd36ece5af",
        "upstream_group": "leaderboard",
        "group_yaml_sha256": "dcf26c03fadaff36643041bb8a6c16dba04ac0eba33117253a10011895781bcd",
    }
    if config["harness"] != expected:
        raise ValueError("unexpected harness identity")
    return config


def dataset_pins(config: dict[str, Any]) -> dict[str, str]:
    return {item["path"]: item["revision"] for item in config["datasets"].values()}


def atomic_json(path: str | Path, payload: Any, mode: int = 0o600) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(temporary, mode)
    temporary.replace(destination)


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()
