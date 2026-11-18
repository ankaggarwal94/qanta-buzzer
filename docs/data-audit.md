# Auditing question data

Run the auditor before using an existing split directory or comparing runs that used different source exports. From a checkout with the project dependencies installed, the same commands work on Windows and POSIX.

```console
python scripts/audit_dataset.py --splits-dir data/processed --report artifacts/split-audit.json
python scripts/audit_dataset.py --input questions.csv --report artifacts/source-audit.json --quarantine artifacts/rows-for-review.jsonl
```

The command exits with 0 for a clean audit, 1 for data findings, and 2 if an input cannot be audited or the report cannot be written. Pass both input options to produce a combined report. The command reads source data and writes only the requested report and optional quarantine file. Quarantine is a JSONL copy of rejected raw records, with their one-based record positions and reasons. Existing checkpoints, results and dataset files are independent of these outputs.

## Persisted train, validation and test splits

Supply a directory containing `train_dataset.json`, `val_dataset.json` and `test_dataset.json`. The auditor accepts question arrays and the wrapped `{"questions": [...]}` form. It binds each file to its SHA-256 digest and checks question IDs, normalized clue text and answer labels. Clue comparison uses the same Unicode NFKC normalization, case folding, and whitespace handling as the split builder.

Inspect `split_integrity.cross_split_text_groups` for the affected groups. The count at `row_overlap.test.train.rows` is the number of test rows whose clue occurs in training. Each test row counts once even when training has several copies. Calculate the validation overlap separately. A repeated clue within a single split with consistent labels is reported for inspection.

A finding in historical files applies to those specific inputs. To obtain metrics on disjoint splits, resolve any label conflicts, rebuild the splits into a fresh directory, and rerun the affected training and evaluation against that directory.

## Run paper metrics on a new dataset

Use the preflight entry point to run CSLI, prefix calibration, StopDFF, or StopDFF DP on new inputs. The metric's own parser supplies its default data directory, fit split, and evaluation split. Place its usual options after the metric name. Point outputs to a new directory when you want to keep historical results.

```console
python scripts/run_audited_metric.py compute_csli --data-dir artifacts/main --models tfidf --output artifacts/main/csli.json
python scripts/run_audited_metric.py compute_stopdff_dp --data-dir artifacts/main --fit-split fold_a --split fold_b --out artifacts/main/dp.json --out-md artifacts/main/dp.md --out-tex artifacts/main/dp.tex
```

The preflight checks the complete selected files and available canonical peers before passing control to the metric, including on smoke runs. The four original producer scripts remain available for replaying the hash-bound paper release. Invoking those scripts directly uses their historical validation behavior. The current learned-value trainer, learned-value evaluator, and DP sweep check their effective split inputs directly. For checkpoints that record the training directory and split names, learned-value evaluation checks the recorded cohorts against the selected evaluation split. When a training directory has moved, provide the recorded split files in the evaluation data directory.

## Raw CSV, QANTA and QBReader exports

```console
python scripts/audit_dataset.py --input data/raw/qanta.json --format qanta --qanta-answer-field page --report artifacts/qanta-audit.json
python scripts/audit_dataset.py --input data/raw/qbreader.json --format qbreader --report artifacts/qbreader-audit.json
```

The raw audit checks the source ID, clue text and answer used for grouping. Its report includes the source digest, rejected row positions, repeated identifiers, and groups of equivalent clues with conflicting answers. Every member of a conflicting group is included for review. Resolve label conflicts against the source before changing a training corpus. Training loader defaults remain in effect when running training commands.

For native QANTA JSON, `auto` selects mapped `page` labels throughout an export if any record contains that field. Use `--qanta-answer-field answer` to audit tournament answerlines. For QBReader, the auditor reads tossup arrays, packet and query envelopes, and exported JSONL. It removes display markup and preserves answerline instructions. The audit covers the records present in the supplied export.
