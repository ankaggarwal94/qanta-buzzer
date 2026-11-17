# ACL distractor menus

`qb_data/acl_menus.py` provides the opt-in generator. The policy sees the displayed option sequence. Keep gold indices, class IDs, source IDs, registry hashes, sampling probabilities and provenance in the evaluator.

## Inputs and equivalence

Each reservoir row identifies a training or separately reserved reference example by source ID and duplicate-group ID and supplies its answer, category and profile text. Supply a stable unique source ID when a group contains multiple rows. Training, validation and test partitions must already be disjoint by source and duplicate group in the upstream manifest. The oracle accepts only `train` and `reference` rows. The adapter checks reserved reference source/component and group IDs against all target splits. The oracle excludes every contribution from the target group before fitting or scoring profiles. A profile concatenates all surviving texts of the same reviewed answer class, sorted by source ID.

Supply a versioned JSON registry with a reviewer and disjoint classes. Its canonical JSON digest binds the exact registry to each menu. The reviewer field records an attribution supplied by the caller. Only a substantive review of the input answerlines and their context establishes the validity of a real registry. This synthetic illustration has no corpus coverage.

```json
{
  "version": "synthetic-elements-v1",
  "reviewer": "synthetic example",
  "classes": {
    "Gold": ["Au"],
    "Silver": ["Ag"],
    "Copper": ["Cu"]
  }
}
```

The keys are canonical display labels. Each alias maps to exactly one key, including each key itself. Lookup uses NFC, case folding and whitespace normalization. Punctuation, articles, digits and diacritics can distinguish answers. The existing `qb_data.text_utils.normalize_answer` removes punctuation and leading articles and could conflate answers such as C and C++. Missing or ambiguous registry entries raise an error. Context-dependent matches such as a surname shared by two people require source-level disambiguation before using this label-based interface. Treat prompt, reject and conditional instructions as separate fields in the source review.

## Sampling protocol

Let K be the **total** number of options, d = K − 1, and n the number of eligible non-gold answer classes. Every generated menu has one representative per class. A smaller pool raises an error. There is no replacement, K reduction, category relaxation or cross-split fallback.

| Strategy | Eligible population or rule |
| --- | --- |
| `baseline` | Replay the exact K explicit labels and their order. Check class distinctness, exactly one gold, eligible distractors and any supplied ordered source IDs. |
| `uniform_global` | All eligible non-gold training classes. |
| `uniform_category` | Classes with at least one eligible source in the supplied category. Multi-category classes count once. |
| `tfidf_hard`, `sbert_hard` | Highest max(d, ⌈n/2⌉) eligible ranks. |
| `tfidf_easy`, `sbert_easy` | Lowest max(d, ⌈n/2⌉) eligible ranks. The bands can overlap for odd n or large K. |
| `mixed` | Use TF-IDF rank, alternate highest/lowest remaining rank starting highest. At each step prefer the first class that contributes an unrepresented source category. With all categories represented, take the directional extreme. Choose d this way. |
| `prefix_tfidf`, `prefix_sbert` | Top d classes for the supplied observed clue prefix. |

All ranked modes query with the caller's observed prefix. The scorer receives no gold identifier or held-out question suffix. TF-IDF uses lowercase word tokens including single letters, eligible-donor-only IDF and L2-normalized dot products. An empty token vocabulary or an out-of-vocabulary query gives all equal scores. SBERT accepts the existing `models.likelihoods.SBERTLikelihood.score` interface and checks score shape and finiteness. Supply a pinned model snapshot and record the revision, runtime and backend in the enclosing run manifest. Ordering is descending similarity, then ascending canonical class ID. Gold is removed after scoring, and its rank is never a feature. These are *retrieval similarity bands*. Human difficulty and the strength of choices-only cues require empirical measurements.

The evaluation adapter should draw **once** at the first available observation and bind that realized ordered menu across the episode. For a separately declared prefix-varying design, explicitly bind the prefix at which each menu is created and retain its identity per step. A caller can use `draw_id` to distinguish questions and replicates within a group. The stream seed is SHA-256 of canonical JSON containing the sampler version, integer seed, target group, draw ID and strategy.

For a generated pool of size m, draw d successive uniform indices in ranges m, m−1, …, m−d+1 and remove each chosen class. Insert the gold class at an independent uniform index among K slots. Each reachable ordered menu has probability 1/[K × m × (m−1) × … × (m−d+1)]. A fixed prefix-top or mixed set has m=d and probability 1/K! per ordering. A replayed baseline is a point mass conditional on its explicit input. These are probabilities under ideal uniform draws conditional on the inputs. Replaying a particular seed is deterministic. The exposed log probability concerns the entire ordered menu. `Menu.digest` binds its realized content and audit metadata, while the outer experiment manifest must bind source corpus, draw ID, seed, prefix and scorer.

## Reuse and evidence

The repository's `qb_data/mc_builder.py` at `aef3ca43e36c682ff33ed32217ab1bbab4e2408e` already supports TF-IDF/SBERT, anti-artifact guards, training reference sets and variable K. Its gold-profile ranking and full-question guards serve a different experiment. These ACL prefix-only sampling laws reuse the project's SBERT scoring interface and scikit-learn's TF-IDF implementation. The historical ACL builder at `c5981d913165f8dc24173c3ec8615b221cfc944d:scripts/prepare_acl_expansion.py` freezes source IDs, alias components, option order and gold slot. The historical reservoir is disjoint from the sampled calibration, selection and test questions. Use its evaluator/reservoir.json records as separately declared reference donors and preserve the frozen menu's candidate source IDs. A training-questions-only reconstruction can fail eligibility for the authentic historical options. Its recorded syntactic disjointness is insufficient to establish semantic equivalence; the published preparation explicitly records semantic review as false.

Pinafore's QANTA guesser provides a training-answer TF-IDF retrieval precedent. Balepur, Ravichander and Rudinger's choices-only evaluation motivates inspecting menu artifacts, including correct-option position and class-conditioned distributions. Both were design references. No source code from either project was imported. [Pinafore TF-IDF guesser, pinned](https://github.com/Pinafore/qb/blob/a5216348fadc0f2bee7d0d7919068a34a311e2cf/qanta/guesser/tfidf.py) · [Balepur et al. implementation, pinned](https://github.com/nbalepur/mcqa-artifacts/tree/9f1981b036a0f570f5610aa442de8018de44a36d) · [ACL 2024 paper](https://aclanthology.org/2024.acl-long.555/)

One reproducible encoder snapshot is `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The CPU oracle tests exercise actual TF-IDF and an injected SBERT protocol scorer to test query/provenance isolation. The scorer test does not measure model quality or certify a run of that checkpoint. [Pinned model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

Run `python -m pytest -q tests/test_acl_menus.py` from the repository with the test dependencies installed. The first test run on the base checkout failed at collection because the new oracle module was absent. A later behavior-level red run raised `ValueError: invalid baseline source ID list` for an unseen gold with valid frozen distractor IDs, with 14 other tests passing. Allowing an empty source list solely for the gold slot fixed the replay. The suite also exhausts all 24 shrinking-pool draws at (4,3), checks rank band and mixed-set membership, and changes corpus order, excluded rows, gold aliases, registry version and provenance.
