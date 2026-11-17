# Source-bound answer review

The scientific configuration requires both a global equivalence registry and private source-specific adjudication. The registry supplies identities for retrieval. The private review records whether a displayed class is acceptable for that individual full question. The upstream Jane compatibility review records distractor plausibility at individual prefixes.

## Reusable decisions across menu arms

`inputs.class_review` points to a single set of (question ID, canonical class) decisions. The standard matrix, training, export and scoring commands project it to each realized menu. Receipts bind the dataset, registry, reservoir, split roles, individual source rows, decision file and ordered menu. A new seed or generator can reuse previously reviewed classes for that question. Every newly displayed class needs an explicit decision. [Production configuration](../../configs/acl_expansion/full.yaml) · [Verifier](../../qb_data/acl_scientific_review.py)

After compiling the reviewed global registry, collect the candidate pairs for the intended menus and seeds. Repeat the `--menu` and `--seed` arguments for their Cartesian product. To extend an existing review, use `--decisions existing.json` and a fresh `--out` path. Existing judgments are preserved and new pairs start unresolved. These commands work on Windows and POSIX.

```sh
python -m scripts.acl_scientific_review template --config configs/acl_expansion/full.yaml --menu baseline --menu uniform_global --seed 1 --seed 2 --out data/acl_expansion/review/question_class_template.json
```

Save a separate private working copy as `data/acl_expansion/review/question_class_decisions.json`. Inspect each qid’s full question and raw answerline in the evaluator dataset, along with the candidate class referent and donor sources. Supply `judgment`, `reviewer` and a substantive `rationale` for each displayed class. Mark genuinely acceptable alternatives `accepted`, including alternatives distinct from the primary entity. Use `prompt`, `conditional` or `unresolved` when appropriate. A menu can proceed only when its gold is accepted and every displayed distractor is explicitly rejected. A second acceptable class stops that run without redrawing or dropping the question. Review provenance identifies the supplied adjudication. Its semantic quality depends on the reviewer.

The compile command can preflight the same normal matrix combinations. The private template builder prepares candidate lists before decisions exist. It runs no learner or model-policy prediction.

```sh
python -m scripts.acl_scientific_review compile --config configs/acl_expansion/full.yaml --menu baseline --menu uniform_global --seed 1 --seed 2 --decisions data/acl_expansion/review/question_class_decisions.json --out data/acl_expansion/review/coverage.json
```

The registry can declare source-dependent spellings per qid under `source_review.context_only_by_qid`, bound to the exact dataset. A source shorthand can remain outside the global choice catalogue. An independently registered entity keeps its identity when its spelling is acceptable in another question. Display labels always require global coverage.

## Regression from the real archive

All 5,000 archived questions have empty extracted `prompt` and `rejected` lists. After removing HTML tags and case-folding, 1,485 raw answerlines contain “prompt on” and 1,815 contain “accept ”. A regenerated uniform-global menu displayed a source-specific alternative that the raw answerline accepted alongside the designated gold. The structural registry assigned them distinct global classes. The frozen fixed menu used a different draw. The individual row and answers remain in the private evaluator.

Before the scientific review gate, a full-profile regression prepared a corresponding menu from a global registry alone. The test expected rejection and observed successful preparation. The regression suite now covers missing review settings, multiple accepted choices, prompt/conditional/unresolved decisions, scoped terms absent from the global catalogue, stale slot order and reuse of question/class decisions across seeds. [Tests](../../tests/test_acl_scientific_review.py)

## Fixed-menu sidecars

Start a review with the actual prepared freeze, retaining the original dataset dictionary passed to preparation. The template includes the source question, available raw/accepted/rejected/prompt rules and every menu slot. Every judgment initially has status `unresolved`. The reviewer inspects the original source material, assigns a source-specific `accepted` or `rejected` judgment and a rationale to each slot, sets `reviewer` to their identifier and sets each completed question’s status to `reviewed`. Leave ambiguous questions unresolved for adjudication.

```python
from qb_data.acl_oracle_review import review_template, verify_review
from qb_data.acl_protocol import publish_json
from pathlib import Path

publish_json(Path("review-template.json"), review_template(prepared, dataset))
# After a reviewer completes a separate copy, load the private decisions.
from scripts.stopdff_v5.identity import load_json_strict
receipt = verify_review(prepared, dataset, load_json_strict("review-completed.json"))
```

The verifier checks the full dataset and realized menu digests, source row digest, exact split and question text, answer rules, labeled slot order, coverage, reviewer and rationale. Every gold must be judged acceptable and every distractor rejected. A prompted answer appearing as a selectable option requires redesign or resolution in the original source freeze. Conflicting Jane answer rules and changed source or menu inputs are rejected. Reuse a completed review across reward-only runs when the dataset and menus are identical. The receipt preserves the dataset's declared scope, including synthetic fixture status. Reviewer identity is a supplied attribution. The checker verifies record binding and consistency. It cannot attest that the named person performed the review or that their interpretation is scientifically correct.

The template, answer rules, menu key and judgments are evaluator-only. Keep the sidecar outside public policy prompts. The review command and data paths use the same syntax on Windows and POSIX.
