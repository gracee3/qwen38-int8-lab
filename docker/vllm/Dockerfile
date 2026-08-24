# syntax=docker/dockerfile:1.7

ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
FROM ${PYTORCH_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/qwen38-int8-lab
COPY docker/vllm/requirements.in docker/vllm/requirements.lock ./

# Reuse the same immutable PyTorch/CUDA base as the quant image. The precompiled
# vLLM wheel is much smaller than pulling the 9.11 GB official image and still
# carries the release's compiled CUDA extensions.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip uninstall -y spin && \
    python -m pip install --prefer-binary -r requirements.lock && \
    python -m pip check

RUN python - <<'PY'
from importlib.metadata import version

expected = {
    "vllm": "0.27.1",
    "torch": "2.13.0",
    "compressed-tensors": "0.17.0",
}
for package, wanted in expected.items():
    actual = version(package)
    if package == "torch":
        valid = actual == wanted or actual == f"{wanted}+cu130"
    else:
        valid = actual == wanted
    if not valid:
        raise SystemExit(f"{package}: expected {wanted}, got {actual}")
    print(f"{package}=={actual}")
PY

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="qwen38-int8-lab vLLM environment" \
      org.opencontainers.image.description="Qwen3.8 W8A8 compressed-tensors TP2 inference environment" \
      org.opencontainers.image.revision="${VCS_REF}"

WORKDIR /app
ENTRYPOINT ["vllm", "serve"]
