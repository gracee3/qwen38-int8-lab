# Publishing the first layout release

The existing `baseline-2026-09-07` tag is the pre-layout checkpoint/evaluation
baseline at `52de78d4c7fc2672a118802e02f742b701f02b37`. Preserve it. This one PR
prepares the source layout release without rebuilding or downloading images.

The proposed new source tag is `v0.1.0`; it is distinct from existing Docker tags.
Do not move the old baseline tag or retag images to simulate validation.

1. Review the single layout PR, recipe manifests, profile evidence statuses,
   migration map, and validation report.
2. Confirm the intended scope: inherited checkpoint/runtime evidence plus CPU
   layout/contract/recovery validation; no new full-model or GPU validation.
3. Merge only after review. Record the exact merged commit and create an annotated
   source tag on that commit. This PR does not auto-merge or publish the release.
4. Release notes link all three recipe manifests, `environments/images.lock.json`,
   the paired baseline, and the validation report. State the INT8-v2 provenance
   limitation and the unvalidated imported INT4 262K preset.
5. Update local-agent-evals to pin that commit and consume catalog discovery,
   preserving existing frozen runs and benchmark/image overrides. This is a
   consumer change, not a prerequisite for using the new serving launcher.

A tag freezes repository content, not local mutable paths or Docker tag names.
The recorded content IDs and retained payload hashes establish the existing
identities. These model artifacts are functional candidates with bounded evidence;
“stable source release” does not imply universal model quality or hardware support.
If GPU revalidation is desired before the tag, schedule it explicitly as a later
resource-consuming gate. It is not started by this cleanup.
