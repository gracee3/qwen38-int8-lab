# syntax=docker/dockerfile:1.7

ARG VLLM_IMAGE=qwen38-int8-lab/vllm:0.1.0
FROM ${VLLM_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    NLTK_DATA=/opt/nltk_data

WORKDIR /opt/qwen38-int8-lab-eval
COPY docker/eval/requirements.in docker/eval/requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --prefer-binary --no-deps -r requirements.lock && \
    python -m pip check
RUN python - <<'PY'
import hashlib
from importlib.metadata import version
from pathlib import Path

import lm_eval

expected = {"lm_eval": "0.4.12", "vllm": "0.27.1", "torch": "2.13.0+cu130"}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise SystemExit(f"{package}: expected {wanted}, got {actual}")
    print(f"{package}=={actual}")
leaderboard = Path(lm_eval.__file__).parent / "tasks/leaderboard/leaderboard.yaml"
actual = hashlib.sha256(leaderboard.read_bytes()).hexdigest()
expected_hash = "dcf26c03fadaff36643041bb8a6c16dba04ac0eba33117253a10011895781bcd"
if actual != expected_hash:
    raise SystemExit(f"leaderboard task group hash: expected {expected_hash}, got {actual}")
print(f"leaderboard.yaml sha256={actual}")
PY
RUN python - <<'PY'
import nltk

if not nltk.download("punkt_tab", download_dir="/opt/nltk_data", quiet=True):
    raise SystemExit("failed to install pinned IFEval tokenizer data")
PY

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="qwen38-int8-lab accuracy evaluation" \
      org.opencontainers.image.description="lm-eval 0.4.12 Open LLM Leaderboard v2 evaluation overlay" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.lm-eval.revision="6d642546f4688648fced259eb3302efd36ece5af" \
      org.opencontainers.image.base.digest="sha256:60508d8dcbbb0a985955e9cf2f66e561a66c3f1c99bd7ec8fa5020e991a0ef4d"

WORKDIR /app
ENTRYPOINT ["python"]
