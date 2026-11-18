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

## Raw CSV, QANTA and QBReader exports

```console
python scripts/audit_dataset.py --input data/raw/qanta.json --format qanta --qanta-answer-field page --report artifacts/qanta-audit.json
python scripts/audit_dataset.py --input data/raw/qbreader.json --format qbreader --report artifacts/qbreader-audit.json
```

The raw audit checks the source ID, clue text and answer used for grouping. Its report includes the source digest, rejected row positions, repeated identifiers, and groups of equivalent clues with conflicting answers. Every member of a conflicting group is included for review. Resolve label conflicts against the source before changing a training corpus. Training loader defaults remain in effect when running training commands.

For native QANTA JSON, `auto` selects mapped `page` labels throughout an export if any record contains that field. Use `--qanta-answer-field answer` to audit tournament answerlines. For QBReader, the auditor reads tossup arrays, packet and query envelopes, and exported JSONL. It removes display markup and preserves answerline instructions. The audit covers the records present in the supplied export.
