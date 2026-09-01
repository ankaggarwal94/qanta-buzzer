# Final StopDFF run: reproduction capsule

This directory documents the exact StopDFF result set used by the final workshop manuscript.

## Claim boundary

This run compares:

- a **constructed QA-reference trajectory**: cosine similarity between each cumulative clue prefix and the gold-answer string; and
- a **fixed-option MC trajectory**: the maximum cosine similarity between that same prefix and four answer options, with the selected option defined by the similarity argmax.

Both are produced with the same frozen sentence-transformer. The QA side is not an observed generative open-ended model and has no independently observed error labels. The analysis is therefore a **constructed-reference sensitivity audit**, not proof that an MC benchmark preserves the stopping policy of a deployed open-ended QA system.

## Certified run

| Field | Value |
|---|---|
| Run label | `final_modal_5d5328102912` |
| Source commit | `0017b89da921e85a6960cd8a22f1969176aed079` |
| Source tree | `d51ebfee68741e2e6b476faf4d438146a6d41b07` |
| Protocol | `stopdff_bucketed_dp_paired_v2`, schema 2 |
| Backend | Modal |
| Cells | 96 requested, 96 completed, 0 failed, 0 skipped |
| Release status | `VALID` |
| Family verdict | `PASS` |
| Evaluation items | 3,037 paired test items |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Model revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Bootstrap | 1,000 common item resamples, NumPy PCG64, seed 1 |
| Primary statistic | maximum across cells of median absolute prefix-index shift |

All content-addressed identities, exact package versions, and raw-input hashes are recorded in `run_identity.json`. Expected numerical summaries are in `expected_results.json`.

## Three different reproduction claims

These should not be conflated.

### 1. Exact package verification

Given the archived `verified_export/` directory from the certified run, verify every file against its `SHA256SUMS`, then run the repository's standalone validator. This checks the exact historical evidence rather than recomputing a conveniently similar result.

**Current publication status:** the exact certified package is not yet a public GitHub Release asset. Until it is attached to a release or another durable public archive, exact package verification is unavailable to an external collaborator using GitHub alone.

### 2. Re-execution from the archived raw-input bundle

Given the ten exact raw inputs listed in `run_identity.json`, check out the source commit, install the recorded environment, and run the local or Modal v5 driver. This is the most direct computational reproduction of the 96-cell sweep.

### 3. End-to-end regeneration from `questions.csv`

The source commit contains `questions.csv` and all code needed to recreate split-specific MC datasets, calibration, the myopic prerequisite, and the final bucketed-DP sweep. This is the strongest pipeline-level test, but it is not presently a bit-for-bit historical reconstruction because the environment used to generate the ten pre-v5 raw inputs was not captured as a complete closed lock. Compare every regenerated raw-input hash with the `raw_inputs` block in `run_identity.json`; a mismatch means a new scientific run, not the certified historical run.

## Environment

Use Python 3.11.12. For re-execution from the ten archived raw inputs, install the exact identity-bound package set plus the validator dependency:

```bash
git clone https://github.com/ankaggarwal94/qanta-buzzer.git
cd qanta-buzzer
git checkout 0017b89da921e85a6960cd8a22f1969176aed079

python3.11 -m venv .venv-stopdff-final
source .venv-stopdff-final/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  huggingface_hub==1.29.0 \
  matplotlib==3.11.1 \
  numpy==2.4.6 \
  pandas==3.0.5 \
  scikit-learn==1.9.0 \
  scipy==1.17.1 \
  sentence-transformers==6.0.0 \
  torch==2.13.0 \
  transformers==5.16.1 \
  "jsonschema>=4.18,<5"
python -m pip install -e . --no-deps
```

For an end-to-end rebuild beginning with `questions.csv`, additional data-pipeline packages are needed. The best available reconstruction is the repository's pinned 2026-05-26 build stack with the final v5 scientific packages substituted where the certified run recorded later versions:

```bash
python -m pip install \
  datasets==4.8.5 gymnasium==1.2.3 huggingface_hub==1.29.0 \
  jsonlines==4.0.0 "jsonschema>=4.18,<5" matplotlib==3.11.1 \
  numpy==2.4.6 pandas==3.0.5 PyYAML==6.0.3 scikit-learn==1.9.0 \
  scipy==1.17.1 seaborn==0.13.2 sentence-transformers==6.0.0 \
  sentencepiece==0.2.1 stable-baselines3==2.8.0 torch==2.13.0 \
  tqdm==4.67.3 transformers==5.16.1
```

The top-level `requirements.txt` reflects an earlier development environment and is **not** the final run lock. The broader command above is a best-effort reconstruction of the pre-v5 raw-input build environment, not a historical identity record.

## Rebuild the raw inputs

The canonical source CSV at the certified commit is `questions.csv`, SHA-256 `da24b029f6d5186b8b9328a4ac3704f9581e84972e3f7cd467c895fadb81916a`, size 14,968,106 bytes. The frozen construction uses four choices, SBERT-profile distractors, 0.70/0.15/0.15 split ratios, and seed 789685.

```bash
python scripts/build_mc_dataset.py \
  --config configs/cs321m_final.yaml \
  --output-dir data/processed

python scripts/compute_prefix_calibration.py \
  --data-dir data/processed \
  --output paper_exports/calibration.json \
  --fit-split val

python scripts/compute_stopdff.py \
  --data-dir data/processed \
  --calibration paper_exports/calibration.json \
  --output paper_exports/stopdff.json \
  --report-output stopdff_report.json
```

Verify all ten generated files against `run_identity.json` before running v5. A mismatch means the build did not recreate the certified input bytes.

## Re-run StopDFF v5 locally

Place output outside the Git checkout because the runner requires a clean worktree:

```bash
REPRO_ROOT="$(cd .. && pwd)/stopdff-final-reproduction"

python scripts/run_stopdff_v5_local.py \
  --data-dir data/processed \
  --paper-exports paper_exports \
  --out-dir "$REPRO_ROOT" \
  --variant smoke

FINAL_ROOT="$(cd .. && pwd)/stopdff-final-reproduction-full"

python scripts/run_stopdff_v5_local.py \
  --data-dir data/processed \
  --paper-exports paper_exports \
  --out-dir "$FINAL_ROOT" \
  --variant final
```

The final driver performs source staging, model-snapshot freezing, a two-build adapter determinism gate, FVI parameter selection, a common bootstrap plan, smoke and mutation gates, all 96 cells, package creation, and standalone validation. CPU execution is supported. Matching the historical Modal hardware is required only for a claim of byte-identical embeddings and downstream artifacts.

## Validate the numerical results

```bash
python reproducibility/stopdff_final_modal_5d5328102912/verify_expected_results.py \
  /path/to/verified_export
```

The independent reducer checks all per-item shifts in all 96 cells, including the primary statistic, common-bootstrap family histogram, zero-shift proportions, mean-shift ranges, calibrator-family counts, and representative cell. The repository's standalone validator remains authoritative for artifact identities, schemas, evidence bindings, coverage gates, and release validity.

## Interpretation

The primary family statistic is zero with a finite-item bootstrap stability interval of `[0, 0]`. This does **not** establish equivalence or no effect. Every cell has a majority of exact-zero item shifts, which pins its median to zero, while a minority of items can move substantially. Signed means, absolute means, zero-shift fractions, and calibrator breakdowns are secondary distributional analyses.

## Remaining publication blockers

GitHub source alone is not yet a complete archival reproduction record. See `REPRODUCIBILITY_STATUS.md`. Required closure includes:

1. publish the exact certified `verified_export/` directory as a checksummed Release asset or DOI-backed archive;
2. publish or otherwise make accessible the exact ten-file raw-input bundle;
3. resolve repository and dataset redistribution licensing;
4. provide an anonymized code supplement or anonymous mirror during double-blind review.

Do not link this author-identifying repository from an anonymous TAE submission or rebuttal. The named repository is appropriate after author notification.
