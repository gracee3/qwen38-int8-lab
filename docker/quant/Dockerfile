# syntax=docker/dockerfile:1.7

ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/qwen38-int8-lab

# Keep the expensive dependency layer independent of application scripts. The
# lock is generated only after this intent set passes the API and smoke gates.
COPY docker/quant/requirements.in docker/quant/requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip uninstall -y spin && \
    python -m pip install --prefer-binary -r requirements.lock && \
    python -m pip check

RUN python - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "llmcompressor": "0.13.0",
    "compressed-tensors": "0.18.0",
    "datasets": "5.0.1",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
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
LABEL org.opencontainers.image.title="qwen38-int8-lab quantization environment" \
      org.opencontainers.image.description="Qwen3.8 calibrated W8A8 compressed-tensors environment" \
      org.opencontainers.image.revision="${VCS_REF}"

WORKDIR /app
CMD ["/bin/bash"]
