# Seven-question real-data engineering subset

Seven Geography questions were selected from the privately held October 2, 2026 ACL freeze for unambiguous full-question referents, simple target answerlines and recoverable baseline provenance. The original calibration, selection and test splits supply three training, two validation and two test questions respectively. The materializer keeps exact original source rows, qids, duplicate groups, member IDs, prefix text, fraction labels, original order and menu order. It marks the derived dataset's scope `engineering_smoke`. Selection intentionally favors easy-to-review data and supplies no population estimate or held-out performance claim.

Two agents independently reviewed each target’s original answerline and each donor’s identity and source-specific rules. Their recorded judgments cover seven accepted classes and twenty disjoint donor classes. The 21 original distractor slots use 20 distinct donor records. Four Geography donor classes support the generated K=4 category arm. Full-question acceptability, early-prefix plausibility and population performance are separate claims.

The published review envelope contains the original selection, registry, provenance and reviewer rationales under authenticated encryption. The materializer derives its AES-SIV key from the complete private archive bytes using domain-separated SHA-512. Its public SHA-256 identity alone cannot recover the envelope. The materializer checks the original selection and registry digests and writes the exact original reviewer file bytes to the ignored input directory. Keep recovered files with the private evaluator. [Materializer](../../scripts/acl_reviewed_subset.py)

## Reproduction

From the repository root on Windows or POSIX, pass the locally held archive to the materializer.

```sh
python -m pip install -e ".[acl-private-review]"
python -m scripts.acl_reviewed_subset --archive "<path-to-archive>/ACL_Dataset_5000_Frozen_Inputs_2026-10-02.zip"
python -m pytest -q tests/test_acl_reviewed_subset.py
```

The script checks the full archive, the evaluator dataset and reservoir bytes, each selected row, source group membership and the derived datasets. It writes the original selection, reviewed registry, private dataset, donor reservoir, provenance receipt and question-specific decisions to ignored `artifacts/acl_expansion/real_inputs/`. For each of seven questions the selection specifies the accepted class and twenty rejected donor classes. The 147 judgments use the full scientific configuration’s `inputs.class_review` format. The materializer checks their projection against the independent baseline slot judgments. A fresh checkout can check authentication and rejection of other archives using synthetic inputs. The real-source tests require the private materialization. [Materializer](../../scripts/acl_reviewed_subset.py) · [PPO/T5 smoke configuration](../../configs/acl_expansion/reviewed_geography_smoke.yaml)

Both learners completed all 18 single-factor conditions with these inputs at source revision `50283a793cc0193e719f5cfaa952395ebdf01708`. PPO recorded 72 training decisions and T5 recorded 144. Every held-out run has separate validation and test episode hashes and a verified checkpoint digest. The ten realized menu arms share the same underlying source decisions and have ten distinct slot-review digests. [Verified run identities and per-menu reviews](evidence/reviewed_geography_smokes.json)

During initial development, the scientific review preflight regenerated all ten menu arms at seeds 1, 2 and 3 on this subset. Each of those 30 combinations had exactly one reviewed acceptable slot on every question. The refreshed learner receipts above cover seed 1. [Historical three-seed coverage receipt](evidence/reviewed_geography_matrix_preflight.json)
