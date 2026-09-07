#!/usr/bin/env python3
"""Run lm-eval only after installing the suite's fail-closed dataset pins."""

from __future__ import annotations

import os

from common import dataset_pins, load_config
from pinned_datasets import install


def main() -> None:
    config = load_config()
    install(dataset_pins(config), offline=os.environ.get("HF_DATASETS_OFFLINE") == "1")
    import lm_eval

    original_simple_evaluate = lm_eval.simple_evaluate

    def simple_evaluate_without_unpaired_bootstrap(*args, **kwargs):
        # The published uncertainty is the suite's deterministic 10,000-replicate
        # paired bootstrap. Avoid the harness's unrelated 100,000-replicate stderr.
        kwargs["bootstrap_iters"] = 0
        return original_simple_evaluate(*args, **kwargs)

    lm_eval.simple_evaluate = simple_evaluate_without_unpaired_bootstrap
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
