# Reproducibility status

**Assessment date:** 2026-08-30  
**Certified source commit:** `0017b89da921e85a6960cd8a22f1969176aed079`

## Status matrix

| Component | On GitHub? | Sufficient now? | Notes |
|---|---:|---:|---|
| Exact source commit | Yes | Yes | Main currently points at the certified source commit. |
| Source dataset `questions.csv` | Yes | Mostly | Exact file and SHA-256 are recorded. Upstream redistribution terms still need clarification. |
| Dataset construction code/config | Yes | Yes for scientific re-execution | Train-only answer profiles, split seed, K, distractor strategy, and guards are present. |
| Final v5 implementation | Yes | Yes | Local and Modal runners, schemas, mutation tests, and standalone validator are present. |
| Exact final environment | Previously no; captured here | Yes for the v5 stage | The top-level `requirements.txt` is stale for this run. |
| Expected final results | Previously no; captured here | Yes | `expected_results.json` plus an independent reducer. |
| Exact final raw-input bundle | No | **No** | Large processed JSON files are ignored and were retained only outside Git. |
| Exact certified `verified_export/` package | No | **No** | No GitHub release exists. This prevents external exact-package verification. |
| Model revision | Yes | Mostly | Revision is pinned; the full historical snapshot is not a GitHub asset. |
| Cross-hardware bitwise guarantee | No | Not promised | Cosine values are rounded, but the historical contract does not guarantee byte identity across hardware. |
| Public redistribution license | No | **No** for reuse | README states “Not licensed for redistribution.” Contributor consent/provenance must be resolved before adding an OSS license. |
| Anonymous review artifact | No | **No** | The repository owner and history identify the author. Do not link it during double-blind review. |

## Bottom line

A collaborator can inspect the implementation and run a scientifically comparable experiment from GitHub. A collaborator cannot yet verify the exact certified run from GitHub alone because the final evidence package and its large raw inputs are not publicly archived. “The code is there” and “the published result is exactly independently verifiable” are, regrettably, different statements.

## Required closure sequence

1. Package the exact certified `verified_export/` directory without modifying it.
2. Verify its internal `SHA256SUMS`.
3. Compute a SHA-256 for the outer archive.
4. Publish it as a GitHub Release asset or DOI-backed archive.
5. Publish the exact ten-file raw-input bundle, or place it in the same archive if size/terms permit.
6. Add the public asset URLs and outer hashes to this directory.
7. Resolve code and dataset licensing before inviting reuse.
8. Re-run the independent verifier from a fresh clone and attach the transcript to the release.
9. After TAE notification, link the de-anonymized repository from the final paper/project page.
