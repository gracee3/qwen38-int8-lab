# Evidence index

Reports describe the dated run and protocol they measured. Repository links follow
the new layout; historical raw paths, identities, measurements, and limitations
remain historical. Raw data and weights stay outside Git.

| Question | Primary evidence |
| --- | --- |
| Original INT8 artifact and native kernel | [Quality candidate](quality-candidate-2026-08-25.md) |
| INT8-v2 completed build | [Quantization milestone](agentic-w8a8-v2-quantized-2026-09-06.md) |
| INT4 completed build | [INT4 milestone](int4-v1-milestone-2026-09-07.md) |
| Completed 2,440-evaluation comparison, corrected IFEval pairing, external context | [Full local results](paired-16k-complete-2026-09-08.md) |
| Paired 16K scored baseline | [Eight examples per benchmark](paired-16k-calibration-baseline-2026-09-07.md) |
| Original serving throughput | [Inference measurements](inference-performance-2026-08-30.md) |
| INT4 TP1/TP2 96K capacity | [Inference profiles](int4-v1-inference-profiles-2026-09-07.md) |
| INT8-v2 262K admission smoke | [Imported eval evidence](int8-v2-262k-smoke-2026-09-07.md) |
| Target/source identity and calibration clearance | [Source audit](int4-expanded400-source-audit-2026-09-06.json), [overlap clearance](int4-overlap-clearance-2026-09-06.json) |
| Recovery behavior | [Resumable quantization](resumable-quant-2026-09-06.md), [pilot recovery](int4-pilot-recovery-2026-09-07.md) |
| Older six-group evaluation status | [Attempt ledger](evaluation-and-agent-status-2026-08-29.md) |
| This layout release | [Validation](layout-release-validation-2026-09-07.md) |

Other retained host/source/scaling reports support the original build provenance.
They are not current host instructions. A structural pass, kernel pass, capacity
measurement, and benchmark score answer different questions; do not combine them
into an unsupported quality claim.
