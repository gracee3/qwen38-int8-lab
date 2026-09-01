# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.3.0-runtime-ubuntu24.04@sha256:789e629e49401647e22b7054ae9c6c4f6427dba68010ba428deb4cc6b063676e

FROM ${CUDA_DEVEL_IMAGE} AS builder

ARG LLAMA_CPP_REVISION=c845263f8b7d60113e213a3bd2d5cc6472ccf204
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libssl-dev

WORKDIR /src/llama.cpp
RUN git init . && \
    git remote add origin https://github.com/ggml-org/llama.cpp.git && \
    git fetch --depth=1 origin "${LLAMA_CPP_REVISION}" && \
    git checkout --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${LLAMA_CPP_REVISION}"

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=86-real \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF && \
    cmake --build build --config Release -j "$(nproc)" \
        --target llama-server llama-cli llama-bench && \
    install -d /opt/llama/bin && \
    cp -a build/bin/. /opt/llama/bin/ && \
    printf '%s\n' "${LLAMA_CPP_REVISION}" > /opt/llama/REVISION && \
    short_revision="$(printf '%s' "${LLAMA_CPP_REVISION}" | cut -c1-9)" && \
    /opt/llama/bin/llama-server --version | grep -F "commit ${short_revision}"

FROM ${CUDA_RUNTIME_IMAGE}

ARG LLAMA_CPP_REVISION=c845263f8b7d60113e213a3bd2d5cc6472ccf204
ARG VCS_REF=unknown
ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/opt/llama/bin:/usr/local/cuda/lib64

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        libgomp1 \
        libssl3 \
        tini

COPY --from=builder /opt/llama /opt/llama

RUN test "$(cat /opt/llama/REVISION)" = "${LLAMA_CPP_REVISION}" && \
    /opt/llama/bin/llama-server --version && \
    ! ldd /opt/llama/bin/llama-server | grep -F 'not found'

LABEL org.opencontainers.image.title="qwen38-int8-lab llama.cpp environment" \
      org.opencontainers.image.description="Pinned CUDA SM86 llama.cpp shared-library server for Qwen3.5 GGUF" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.llama-cpp.revision="${LLAMA_CPP_REVISION}" \
      org.opencontainers.image.base.devel.digest="sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52" \
      org.opencontainers.image.base.runtime.digest="sha256:789e629e49401647e22b7054ae9c6c4f6427dba68010ba428deb4cc6b063676e"

WORKDIR /app
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/llama/bin/llama-server"]
