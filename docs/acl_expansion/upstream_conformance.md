# Historical ACL freeze interoperability

The fixture exercises schema compatibility using synthetic source rows. It contains no model predictions, human judgments or scientific finding. The historical writer hard-codes its dataset scope to `scientific`. The fixture wrapper records its actual scope as synthetic and retains the emitted dataset verbatim.

The generator reads the historical Python blobs directly from git commit `c5981d913165f8dc24173c3ec8615b221cfc944d`. In a temporary directory, an isolated Python process runs the unmodified historical `scripts.prepare_acl_expansion.prepare`, including candidate filtering, source-component grouping, selection, menu construction with `build_dataset`, and output publication. After publication it separately runs the pinned `qb_data.jane_paired.validate_dataset` on the emitted JSON. The golden file retains the semantic contents of `evaluator/main_dataset.json` and `evaluator/reservoir.json` plus the six upstream source hashes. The source input has 13 rows in two categories and 12 eligible components after deduplication. The selected freeze contains one calibration, one selection and two test questions plus six disjoint reservoir donors. Each question has ten exact prefixes and two frozen menus. The four questions exercise all four gold positions.

| Canonical JSON | SHA-256 |
| --- | --- |
| Entire fixture | `3f820bf5f6a92b551b66e1a02155a8bdde9473d9e89cd20d834ea7c89ab5921f` |
| Historical dataset | `cc615de03d8caa260eb10c7c7abf3b3161effb639e4e6b70f4d1df2096b09aec` |
| Historical reservoir | `adb9989d21ab68d3bd7013e4ba9306708f7d2d6f2495d4c63f8824c86f9c7e38` |

From the repository root in PowerShell or a POSIX shell, the first command checks current compatibility, including installations whose git history is shallow. The second regenerates in temporary storage and compares the decoded golden contents. It requires the pinned upstream object in local git history.

```sh
python -m pytest tests/test_acl_upstream_conformance.py -q
python -m scripts.acl_upstream_conformance
```

Both independent-pool and same-category baselines must retain the historical ordered labels, gold slot, donor IDs and all prefixes. The declared crosswalk sends calibration to train, selection to val and test to test. The held-out TF-IDF check requires every selected distractor source to belong to the separate reservoir and rejects a reservoir row contaminated with a target component member. The test-only class map uses the synthetic source labels and explicit aliases. It supplies no evidence for adjudicating a real answerline.
