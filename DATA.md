# Data and Split Provenance

## Data source

This project uses quizbowl tossup questions because they are naturally
incremental: clues are revealed from obscure to obvious, so both correctness
and answer timing carry measurement signal.

The CS321M audit treats QANTA-style multiple-choice reformulation as a
benchmark translation problem. The source task is open-ended incremental QA;
the proxy task is multiple-choice answer-or-abstain over matched prefixes.

## Why quizbowl / QANTA

Quizbowl is appropriate for this project because:

- each item is already ordered by evidence state;
- early vs late answerability is part of the construct;
- the same item can be evaluated at multiple prefixes;
- answer timing can be audited as a decision boundary;
- answer choices can introduce a measurable intervention.

The project-origin thread framed the task as converting each clue into a
multiple-choice step where the model can answer or abstain and receive the
next clue, with diminishing rewards for later correct answers. It also
identified distractor construction, fixed-vs-dynamic distractors, and wrong
human/computer answers as central design questions.

## Split contract

`build_mc_dataset.py` must construct all MC artifacts from a split-safe
pipeline:

1. load raw questions;
2. create raw train/val/test splits;
3. fit answer profiles from raw train only;
4. construct MC train/val/test questions using the train reference corpus;
5. write split-specific artifacts and `build_metadata.json`.

The script already performs the raw split before profile fitting and writes
`mc_dataset.json`, `train_dataset.json`, `val_dataset.json`,
`test_dataset.json`, and `build_metadata.json`. See
`scripts/build_mc_dataset.py`.

## Required provenance fields

Every full MC build should record:

```json
{
  "schema_version": 2,
  "fresh_split": {
    "seed": 789685,
    "created_at_utc": "...",
    "script_sha256": "...",
    "git_commit": "...",
    "git_dirty": false
  },
  "mc_build": {
    "created_at_utc": "...",
    "script_sha256": "...",
    "git_commit": "...",
    "git_dirty": false,
    "constructed_after_fresh_split": true
  },
  "split_hashes": {
    "raw_train_qids_sha256": "...",
    "raw_val_qids_sha256": "...",
    "raw_test_qids_sha256": "...",
    "mc_train_qids_sha256": "...",
    "mc_val_qids_sha256": "...",
    "mc_test_qids_sha256": "..."
  },
  "retention": {
    "train": {"raw_count": 0, "retained_count": 0, "retention_rate": 0.0},
    "val": {"raw_count": 0, "retained_count": 0, "retention_rate": 0.0},
    "test": {"raw_count": 0, "retained_count": 0, "retention_rate": 0.0}
  }
}
```

## Retained-subset warning

If val/test MC retention falls below the frozen threshold and the run proceeds
with an override, all downstream artifacts must identify the result as
**retained-subset**. A retained-subset audit can be scientifically useful, but
it is not a clean full raw-test PASS.

## Known limitation

A historical artifact can be stale if `fresh_split.py` was run after
`build_mc_dataset.py`, because the test split may be fresh while distractors
were built against an older train reference pool. Final evidence must either
prove MC construction happened after the final split or label the result as
legacy/stale-provenance.
