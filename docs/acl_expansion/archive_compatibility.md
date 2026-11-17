# Structural compatibility of the 5,000-question historical archive

On 2026-11-17 Pacific, the local audit recomputed the zip digest, the hash and byte length of all 63 files listed by the outer manifest, and the nested frozen, code and tokenizer manifests. It ran `qb_data.acl_protocol.prepare` twice on the actual historical evaluator dataset and its separate reservoir. No source records, aliases or per-question judgments were exported. [Aggregate receipt](archive_compatibility_receipt.json)

| Check | Result |
| --- | --- |
| Questions and source splits | 5,000 total, comprising 1,000 calibration, 1,000 selection and 3,000 test |
| Separate reservoir | 1,100 donors, exactly 100 in each of 11 target categories |
| Component membership | 30,163 target source IDs and 3,642 donor source IDs, with zero shared IDs |
| Frozen independent-pool baseline | 50,000 prefixes, 15,000 ordered distractor IDs and all labels/gold slots matched |
| Frozen same-category baseline | 50,000 prefixes, 15,000 ordered distractor IDs and all labels/gold slots matched |
| Mechanical label association | 6,100 canonical strings, 6,688 distinct normalized labels and zero conflicting owners under the runtime's label-key function |
| Human review queue | 5,000 pending, zero reviewed answer-type fields |

The structural check formed its temporary class map from frozen canonical strings and explicitly extracted aliases. The required reviewer field held a sentinel declaring **no reviewer**. The map stayed in memory. Zero normalized string collisions establish compatibility with this parser. They supply no semantic equivalence evidence.

All structured rejected and prompt lists were empty. The preserved raw answerlines contained the word “prompt” in 1,512 questions, “reject” in 99 and “before” in 447. These lexical counts identify work for a source-aware reviewer. They do not parse the instructions or establish that any specific distractor is acceptable. The queue contains 5,000 pending records bound to the matching answerlines and splits.

Inspection of the 63-member manifest and its review-related records found a priority fragment-review CSV, a pending per-question review queue, nine curated source-component identity groups and the optional type-candidate validation receipt. That receipt covers 342 candidate menus and declares human or semantic validation false. The archive includes no completed global answer-equivalence registry or completed source-bound semantic adjudication for the expansion. The historical writer's scope declaration and byte hashes identify the original material. They do not authenticate third-party review.

Run the following from the repository root in PowerShell or a POSIX shell, replacing the archive argument with the locally available file. The audit opens the zip read-only. The receipt contains only aggregate counts and digests.

```sh
python -m scripts.acl_archive_compatibility --archive "<archive-path>" --receipt docs/acl_expansion/archive_compatibility_receipt.json
```

The archive SHA-256 is `cd2f72b9a1fd774ccd1da30aefca418b60c5c8c35ffaf88b625773dd13f82cc3`. The canonical JSON receipt SHA-256 is `61f2d0f8d76b518a892bb2730e7d50de69d09742219c12ba48bfdd0b28c00f5a` before its final newline. A local rerun with the same bytes and code must agree with the stored receipt. No model inference or scientific comparison took place.

## Materialize the full configuration and hand off review

From the repository root, supply the locally held archive path. The command verifies its pinned SHA-256 and manifests, writes the original evaluator dataset and reservoir bytes at the paths expected by `configs/acl_expansion/full.yaml`, and writes a private readable draft. Repeating it accepts identical files and rejects a changed destination. The three files live under the ignored `data/acl_expansion/` directory.

```sh
python -m scripts.acl_full_inputs materialize --archive "<archive-path>"
```

The local materialization produced `evaluator/main_dataset.json` (73,545,887 bytes), `evaluator/reservoir.json` (3,660,892 bytes), and `review/draft.json` (6,100 unresolved source records). The draft has no compiled class map. Both the class loader and review compiler reject it. A second materialization reproduced it.

A reviewer should save a **separate working copy** as `data/acl_expansion/review/completed.json`. Use these commands once to start that copy.

PowerShell:

```powershell
[System.IO.File]::Copy((Resolve-Path data/acl_expansion/review/draft.json), (Join-Path (Get-Location) data/acl_expansion/review/completed.json), $false)
```

POSIX:

```sh
cp -n data/acl_expansion/review/draft.json data/acl_expansion/review/completed.json
```

For each target or reservoir record, inspect its retained raw answerline and the full question in the indicated evaluator file using its source ID. Inspect across records for equivalence. Set `canonical_class` to the reviewed global display identity. List every alternative judged globally interchangeable in `global_aliases`, including frozen source spellings when different from that display identity. Record ambiguous or conditional names in `context_only` and explain the decision in `rationale`. A source-local form never establishes global equivalence to that question’s gold. A label already registered for another independently reviewed entity keeps that identity. An otherwise ambiguous term can remain absent from the global map, bound only to this question and dataset digest. Leave unresolved decisions pending. When all source records have been adjudicated, set each completed row's `status` to `reviewed`, enter the responsible reviewer's name in the document-level `reviewer` field, and change the document-level `status` to `reviewed`. The immutable source fields and digests must match the retained files.

```sh
python -m scripts.acl_full_inputs compile --decisions data/acl_expansion/review/completed.json
```

Compilation checks the pinned file hashes, every source identity, decision coverage, reviewed aliases and rationales. It runs both historical baseline menu conditions against the resulting registry. A source may accept a different entity or a locally meaningful shorthand without merging globally distinct classes. The compiler preserves each target’s declared source-local forms and checks that the source canonical still belongs to the reviewed gold class. The run’s question/class review must resolve each displayed option. Displaying a second accepted class, a prompted class or an unresolved class fails the scientific gate. If such a class is absent from the menu, the original question and its acceptance rule can still be used unchanged. A successful compile writes `data/acl_expansion/reviewed_classes.json` atomically. That file contains the review identity and the input digests. A name, rationale and digest give provenance for a supplied judgment. They do not independently establish that a person performed a substantive review.

The full configuration next requires the private question/class decisions described in the source review guide. A separate fixed-menu review is also available after class compilation. Select the actual run's menu and seed. The command uses the source-bound sidecar schema and leaves each slot unresolved. [Reusable question/class review workflow](source_review.md)

```sh
python -m scripts.acl_full_inputs menu-draft --config configs/acl_expansion/full.yaml --menu baseline --seed 1 --out data/acl_expansion/review/baseline-menu-draft.json
```

For fixed-menu mode, replace `inputs.class_review` with `inputs.review` pointing to a separately completed copy. The runner validates the exact dataset, menu, source and slot judgments. Changed menus require corresponding evidence. [Source-bound review](source_review.md)

The full private source material and unresolved draft were materialized locally. No real registry was compiled. The tests exercise draft rejection, source binding, missing or conflicting decisions, repeated publication, a completed synthetic registry and an optional source-bound menu template.

```sh
python -m pytest tests/test_acl_full_inputs.py -q
```
