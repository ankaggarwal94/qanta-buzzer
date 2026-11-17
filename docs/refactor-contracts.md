# Refactor contracts and verification

Baseline commit `a3195eaf3c4d647978c126c9edbeb69c3515527d`. This inventory describes the production contracts used to select and verify the changes. The three root StopDFF contracts and the ACL proof challenges remain the authorities for their protocols.

| Boundary | Invariant and business rule | Consumers and dependencies |
| --- | --- | --- |
| Questions and answer profiles | Split question groups without leakage. Fit profiles to the reference corpus. Preserve source order, multiplicity, leave-one-out exclusion by question ID, whitespace tokenization, positive token caps and answer-text fallback. | `qb_data` loads and groups tossups, constructs menus and profiles. `models.answer_profiles` and `qb_env` expose compatibility imports. |
| Menus and likelihoods | Option index, gold index and response profile identify the same choice. A scorer returns one raw score per option in input order. Dense embedding rows correspond to the requested texts including repeats. Cached float32 vectors belong to the fitted representation. | The TF-IDF backend uses scikit-learn. SBERT, T5 and opt-in OpenAI/DSPy backends share model and cache interfaces. Temperature belongs to the caller. |
| Environment and agents | Action 0 waits, action i+1 commits option i. Mask padding beyond local K. Reset and cached inference refer to the same sampled row. Beliefs and agent traces preserve prefix order. Threshold agents buzz on the first qualifying step or the last available step. | Gymnasium environment consumes question data, features and likelihoods. SB3, text wrappers and baseline agents consume observations and action masks. |
| Policy training | Joint action probabilities factor into stop and conditional answer probabilities. WAIT has no answer gradient contribution. Preserve selected action log probabilities, tokenizer padding, valid-token masks, seeded sampling and bootstrapping semantics. | T5 supervised and PPO trainers use PyTorch and Transformers. Stop-only and variable-K paths have distinct action spaces. |
| Rewards and evaluation | Legacy horizon behavior follows `end_mode`. ACL has a single terminal settlement, matching shaping and learner discount, and zero terminal potential. At step t, first-buzz mass is the conditional hazard times the probability of surviving earlier steps. Expected Wins combines fractional correctness and opponent survival. | Runtime rewards feed policies. `evaluation.metrics` consumes dict or dataclass traces. Calibration excludes forced commitments. CLI entrypoints use separate train, validation and held-out test splits. |
| StopDFF v5 | Canonical identity JSON has sorted NFC keys and ordered arrays. Scientific decimals use strings. Row files have their own canonical JSONL and deterministic gzip format. Stage complete files before atomic publication. A failing row producer must preserve the old destination. | Adapter, manifests, sweep and writers feed a checker that recomputes gates. Seeds, paired bootstrap samples, checksums, schemas, package namespace and artifact bytes are observable. |
| ACL experiments and delivery | Scientific menus require reviewed answer evidence. Policy-facing text uses available prefixes. Checkpoint identity binds configuration, source, inputs and model assets. Existing receipts describe their original revisions. | CLI workflows compose data, environment, learner and grading components. Lean challenges cover specified sampler, transition and reward properties. Python and native numeric libraries remain separate trust boundaries. |

The verification sequence is contract inventory, failing regression or resource test, implementation, cross-domain review, then independent production-path checks. For numerical kernels, compare the Python results against an independently specified recurrence and state the arithmetic model used in proofs. For artifacts, compare complete bytes. For optional dependencies, report which integrations actually ran.

## Internal dependencies

Static imports in the production packages, including deferred and type-only imports, give the following edges. Some edges are bidirectional because the historical compatibility modules re-export implementations and the environment shares Bayesian math with the agents.

| Package | Imports from this repository |
| --- | --- |
| `agents` | `models`, `qb_data`, `qb_env` |
| `evaluation` | `agents`, `qb_data` |
| `models` | `qb_data` |
| `qb_data` | `models`, `qb_env` |
| `qb_env` | `agents`, `models`, `qb_data` |
| `training` | `models`, `qb_data`, `qb_env` |
| `scripts` | All six packages above |
