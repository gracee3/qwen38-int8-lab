# Contributing

Use a focused branch from current remote `main`. Preserve dirty work in other
checkouts. Personal checkouts belong under `~/projects`; linked worktrees under
`~/worktrees`. Open a PR with exact validation and resource limitations.

This layout release freezes quantization behavior, recipe settings, and dependency
versions. Dependency updates, new quantization policies, new model builds, and
release publication need separate deliberate work. Do not rebuild or pull images
as a side effect of documentation, discovery, or tests.

Run the checks in [validation](docs/validation.md). Profile changes require tests
of effective settings and CLI output. Quantization changes require synthetic
coverage and explicit scheduling of any real-model GPU validation. Never infer
quality from a structural check or a two-example smoke.

Keep model weights, calibration records, raw runs, credentials, and caches outside
Git. Preserve historical evidence identities and limitations. Update maintained
docs instead of adding agent handoffs. Read [architecture](docs/architecture.md)
before changing checkpoint/resume code and [profiles](docs/profiles.md) before
changing the public discovery contract.
