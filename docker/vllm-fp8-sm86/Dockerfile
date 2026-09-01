# syntax=docker/dockerfile:1.7

ARG VLLM_IMAGE=qwen38-int8-lab/vllm:0.1.0
FROM ${VLLM_IMAGE}

# FlashInfer's FP8 KV compatibility path on SM86 JIT-compiles one CUDA kernel.
# nvcc is supplied by a read-only host-toolkit mount; keep only the small host
# C++ compiler dependency in this experimental image.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      g++=4:13.2.0-7ubuntu1 \
      g++-13=13.3.0-6ubuntu2~24.04.1 \
      g++-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
      g++-x86-64-linux-gnu=4:13.2.0-7ubuntu1 \
      libstdc++-13-dev=13.3.0-6ubuntu2~24.04.1 && \
    rm -rf /var/lib/apt/lists/*

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="qwen38-int8-lab vLLM SM86 FP8 JIT environment" \
      org.opencontainers.image.description="Experimental PP2 W8A8 inference with FlashInfer FP8 KV JIT support" \
      org.opencontainers.image.revision="${VCS_REF}"
