# Validation without rebuilding images

`just test` runs the full unit suite inside the existing quant image, with no
network, no GPU devices, two CPUs, and a 4 GiB memory limit. The checkout is mounted
read-only; `--pull never` prevents image downloads. No host virtual environment or
new dependencies are required.

For lightweight catalog checks on the host with existing PyYAML:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_serving.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_layout.py -v
python3 serving/launch.py list --json
python3 serving/launch.py serve int4-v1-96k-fp8-tp1 --dry-run
```

The full suite additionally needs the pinned Torch/NumPy packages in the existing
container. Do not install those large packages on the host just to run tests.

The baseline fixture records original recipe settings, profile values, and
SHA-256 hashes of dependency files. Tests reject retuning, unrecognized fields,
incompatible models, invalid GPU bindings, and silent overrides. They check
shared engine/HTTP values and credential-free command previews.

## Optional tiny CPU recovery integration

Run explicitly in the existing quant image, with no GPU or network, two CPUs and
an 8 GiB memory limit. Mount the repository read-only at `/app`, set the workdir to
`/app`, and use a disposable container `/tmp` root. For each of `W8A8` and `W4A16`,
run these in separate Python processes inside that container:

```sh
python -B tests/resume_integration.py baseline W8A8 /tmp/resume-W8A8
python -B tests/resume_integration.py interrupt W8A8 /tmp/resume-W8A8
python -B tests/resume_integration.py resume W8A8 /tmp/resume-W8A8
python -B tests/resume_integration.py compare W8A8 /tmp/resume-W8A8
```

Repeat with `W4A16` and `/tmp/resume-W4A16`. The interrupted process must print
`EXPECTED_CHECKPOINT_INTERRUPT`; compare must report `EXACT_RESUME_MATCH`.
Use `docker run --rm` so synthetic tensors and temporary logs disappear with the
container. This exercises shared calibration/recovery behavior without loading
the 27B source. It does not validate GPU kernels, full-model resource use, or
resume compatibility with historical snapshots.

Shell scripts also pass `bash -n` and ShellCheck. `just --list`, command previews,
Markdown link checks, Python AST parsing, and `git diff --check` verify the moved
entrypoints and documentation. See the [release evidence](../reports/layout-release-validation-2026-09-07.md).
