# Container environments — 2026-08-24

Both images share this immutable base:

```text
pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime
sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
```

Application code is bind-mounted read-only at `/app`; it is not copied into dependency layers. Each image installs its resolved lock file while the adjacent `requirements.in` explains direct intent. BuildKit pip caches are retained, and revision labels are applied after dependency installation so a new Git commit does not invalidate the expensive layer.

The locked rebuild and runtime gates were executed at Git `5bfa399f561fcd935e8af883df1c58d8afe97610`. Local image manifest IDs were `sha256:7c0ac3089184466e7348dd98ba0219311e69fa32b7395a7674956467a9e02088` (quant) and `sha256:60508d8dcbbb0a985955e9cf2f66e561a66c3f1c99bd7ec8fa5020e991a0ef4d` (vLLM); both carry that revision label.

## Quantization image

Direct compatibility contract:

- PyTorch 2.13.0+cu130
- Transformers 5.14.1
- LLM Compressor 0.13.0
- compressed-tensors 0.18.0
- Accelerate 1.14.0
- Datasets 5.0.1
- Safetensors 0.8.0
- Hugging Face Hub 1.28.0

LLM Compressor 0.13.0 constrains Transformers to 5.14.1 and compressed-tensors to 0.18.0. The complete resolver result is `docker/quant/requirements.lock`.

## Inference image

Direct compatibility contract:

- vLLM 0.27.1
- PyTorch 2.13.0+cu130
- Transformers 5.15.1 (resolved by vLLM)
- compressed-tensors 0.17.0

The precompiled PyPI release reuses the already-cached PyTorch/CUDA base. It avoids a separate pull of the substantially larger official serving image while retaining the release's compiled CUDA extensions. The complete resolver result is `docker/vllm/requirements.lock`.

The environments intentionally remain separate because their tested compressed-tensors versions differ. Cross-version checkpoint compatibility still requires validation with the completed artifact.
