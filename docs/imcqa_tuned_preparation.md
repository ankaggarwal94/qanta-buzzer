# Fresh IMCQA input preparation

The independently tuned policy lock in `tuned_policies/selection_v2` plans 850
fresh questions and 34,000 plain MCQA contexts. No fresh cohort or model scores
have been produced. The original full source snapshot could not be recovered:
its final single-file download returned HTTP 502. The locally retained original
5,000 questions are all excluded, the 1,100 reservoir components remain
distractors, and the 370 reserved 2026 questions are outside this historical
test's 2010–2025 frame.

`scripts/prepare_imcqa_tuned.py` is ready to run when the original normalized
source JSONL, original exclusions JSON and curated identity overrides JSON are
restored. Their SHA256 values are hard-coded in `INPUT_PINS`; a different source
snapshot cannot silently replace them. The original dataset and reservoir are
also hash-pinned. Reconstructing the old grouping and partition must reproduce
every original component identity before fresh sampling proceeds.

The source audit reported 15,106 unused historical components before the new
0.8 text screen. This is an upper bound on current capacity, not a verified
850-question fresh cohort. The builder excludes all old component source IDs
and aliases and applies inclusive normalized word-5gram Jaccard >=0.8 against
original excluded texts, all original 5,000 texts, and reservoir texts. It
greedily deduplicates the entire fresh candidate pool in frozen hash order,
then allocates the locked N to the original raw-source category/difficulty
weights using bounded proportional allocation. No answer correctness or model
outcome influences this sampling. Original menu construction and answer
semantics are retained as the experiment's authorized setup assumption.

The exact local command below expects three environment variables containing
the actual restored filenames. These filenames are not known until the source
archive is recovered; the command fails immediately if any is missing.

```bash
cd /workspace/scratch/cb54ff173a76/qanta-buzzer
.venv/bin/python scripts/prepare_imcqa_tuned.py \
  --source-jsonl "${IMCQA_SOURCE_JSONL:?set to the restored normalized source JSONL}" \
  --exclusions "${IMCQA_EXCLUSIONS:?set to the original exclusions JSON}" \
  --identity-overrides "${IMCQA_IDENTITY_OVERRIDES:?set to the original identity overrides JSON}" \
  --prior-dataset /workspace/scratch/cb54ff173a76/info_sheet_work/sources/ACL_5000_Analysis_2026-10-03/verified_frozen/evaluator/main_dataset.json \
  --reservoir /workspace/scratch/cb54ff173a76/info_sheet_work/sources/ACL_5000_Analysis_2026-10-03/verified_frozen/evaluator/reservoir.json \
  --policy-lock /workspace/scratch/cb54ff173a76/tuned_policies/selection_v2/frozen_policies.json \
  --sample-size-planning /workspace/scratch/cb54ff173a76/tuned_policies/selection_v2/sample_size_planning.json \
  --selection-receipt /workspace/scratch/cb54ff173a76/tuned_policies/selection_v2/selection_receipt.json \
  --out-dir /workspace/scratch/cb54ff173a76/tuned_experiment/fresh
```

The output directory is create-once. Only `public/pilot.json` is needed by the
GPU scorer. Its `main_dataset_sha256` binds the separate selected evaluator
file, `policy_lock_sha256` binds the frozen policies, and
`sample_size_planning_sha256` binds N. The original selection receipt must bind
the same policy and planning files. The public package includes only the
allowlisted inference fields; real gold labels are absent. `manifest.json`
binds all generated files. `preparation_receipt.json` reports
`prepared_not_scored`; it is not an experimental result.

The model is pinned to Qwen/Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Prompts, option mappings and reward
schedule retain the previous plain-answerer setup. Five prefixes × four
rotations × two menus yield 40 jobs per question. Shared dataset validation and
job building accept an optional `required_splits` keyword so the fresh cohort
can contain only test questions; their existing default still requires all
three original splits.

Validation: synthetic preparation exercises original-source reconstruction,
component and near-text exclusion, quotas, full grid rendering, leakage
rejection and manifest/hash binding. The original saved 5,000-question dataset
also passes unchanged default validation. No synthetic score is scientific
evidence.
