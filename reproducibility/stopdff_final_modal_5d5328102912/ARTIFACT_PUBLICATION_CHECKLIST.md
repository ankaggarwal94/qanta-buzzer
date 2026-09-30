# Exact-artifact publication checklist

The repository source and run identities are public, but the exact historical evidence package is not. Complete these steps after confirming redistribution rights and double-blind timing.

## Package the certified directory

Use the certified directory without regenerating, renaming, or editing anything inside it:

```text
verified_export/
  aggregate.json
  bootstrap_plan.json
  cells/
  evidence/
  figures/
  reports/
  run_manifest.json
  run_spec.json
  SHA256SUMS
  ...
```

1. Run `sha256sum -c SHA256SUMS` from inside `verified_export/`.
2. Create an outer archive that preserves names and bytes.
3. Compute the outer archive's SHA-256 and byte size.
4. Record both in `run_identity.json` under a new `published_archive` block.
5. Attach the archive to a GitHub Release or a DOI-backed archive. If it exceeds the host's per-asset limit, split it losslessly and publish a manifest for the parts.
6. Download the public asset into a fresh machine and verify:
   - outer archive SHA-256;
   - every line of the internal `SHA256SUMS`;
   - the standalone repository validator;
   - `verify_expected_results.py`.
7. Attach the verification transcript and machine/environment description to the release notes.

## Raw inputs

The exact ten-file raw-input bundle is approximately one gigabyte before outer compression. Publish it either:

- inside the certified evidence archive, if redistribution and asset-size constraints permit; or
- as a separate checksummed asset referenced by `raw_input_bundle_id`.

Do not silently substitute regenerated files with matching row counts. The hashes in `raw_inputs.sha256` define the certified bytes.

## Model snapshot

The model is pinned to a concrete Hugging Face revision. Publishing the historical snapshot is useful for exact verification but must comply with the model's license. If the snapshot is omitted, document that the reproduction downloads the pinned revision and may not be byte-identical if upstream hosting changes.

## Release notes must state

- constructed QA-reference claim boundary;
- exact source commit and tree;
- all content-addressed IDs;
- exact environment;
- whether the release supports exact verification, scientific re-execution, or both;
- lack of a cross-hardware bitwise guarantee;
- any files excluded for licensing or size reasons.
