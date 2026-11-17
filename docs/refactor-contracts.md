# Refactor contracts and verification

Baseline commit `a3195eaf3c4d647978c126c9edbeb69c3515527d`. This inventory describes the production contracts used to select and verify the changes. The three root StopDFF contracts and the ACL proof challenges remain the authorities for their protocols.

| Boundary | Invariant and business rule | Consumers and dependencies |
| --- | --- | --- |
| Questions and answer profiles | Split question groups without leakage. Fit profiles to the reference corpus. Preserve source order, multiplicity, leave-one-out exclusion by question ID, whitespace tokenization, positive token caps and answer-text fallback. | `qb_data` loads and groups tossups, constructs menus and profiles. `models.answer_profiles` and `qb_env` expose compatibility imports. |
| Menus and likelihoods | Option index, gold index and response profile identify the same choice. A scorer returns one raw score per option in input order. Dense embedding rows correspond to the requested texts including repeats. Cached float32 vectors belong to the fitted representation. | The TF-IDF backend uses scikit-learn. SBERT, T5 and opt-in OpenAI/DSPy backends share model and cache interfaces. Temperature belongs to the caller. |
| Environment and agents | Action 0 waits, action i+1 commits option i. Mask padding beyond local K. Reset and cached inference refer to the same sampled row. Beliefs and agent traces preserve prefix order. Threshold agents buzz on the first qualifying step or the last available step. | Gymnasium environment consumes question data, features and likelihoods. SB3, text wrappers and baseline agents consume observations and action masks. |
| Policy training | Joint action probabilities factor into stop and conditional answer probabilities. The log probability of a WAIT action depends only on the stop decision. Preserve selected action log probabilities, tokenizer padding, valid-token masks, seeded sampling and bootstrapping semantics. | T5 supervised and PPO trainers use PyTorch and Transformers. Stop-only and variable-K paths have distinct action spaces. |
| Rewards and evaluation | Legacy horizon behavior follows `end_mode`. ACL has a single terminal settlement, matching shaping and learner discount, and zero terminal potential. At step t, first-buzz mass is the conditional hazard times the probability of surviving earlier steps. Expected Wins combines fractional correctness and opponent survival. | Runtime rewards feed policies. `evaluation.metrics` consumes dict or dataclass traces. Calibration excludes forced commitments. CLI entrypoints use separate train, validation and held-out test splits. |
| StopDFF v5 | Canonical identity JSON has sorted NFC keys and ordered arrays. Scientific decimals use strings. Row files have their own canonical JSONL and deterministic gzip format. Stage complete files before atomic publication. A failing row producer must preserve the old destination. | Adapter, manifests, sweep and writers feed a checker that recomputes gates. Seeds, paired bootstrap samples, checksums, schemas, package namespace and artifact bytes are observable. |
| COLM AIMS release | Source verification checks package structure. Release verification additionally binds external expectations, the claim ledger and the selected run. Scientific rendering requires a canonical passing release verdict. | The standalone verifier, finalizer and renderer under `reproducibility/colm_aims_2026` share schemas and identity checks. The inference replay requires the pinned NumPy version. |
| Legacy T5 application | The root `main.py` supports supervised, PPO, full and evaluation modes with the root config, dataset, environment and checkpoint conventions. | Root `model.py`, `train_supervised.py`, `train_ppo.py` and `metrics.py` serve this separate entrypoint. |
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

## Review and regression evidence

Domain reviewers inspected the baseline independently. Implementers then wrote failing tests against production entrypoints. A different reviewer tried to falsify each implementation using baseline code, independently generated traces, or the historical wire format. Integration checks used the merged baseline and candidate on Linux with the same offline dependencies.

| Boundary | Regression or independent challenge | Outcome |
| --- | --- | --- |
| Embedding identity | Repeated texts, malformed model responses, failed TF-IDF refit and a late malformed array in an existing NPZ | The backend receives unique misses in order. Outputs restore input order and multiplicity. Each batch or archive is validated before updating the cache. A failed refit preserves the prior estimator and cache. |
| Profile semantics | 14,280 differential inputs covering Unicode whitespace, duplicate IDs, blank text, exclusion, fallback, and extreme token limits | Outputs matched the baseline. The positive cap bounds tokenization even for a single very long question. |
| Baseline decisions | 1,008 live and cached episodes against the original, plus all 720 combinations in the short-trace production replay test | The first qualifying step, tie-breaking, rewards and trace lengths matched. Peer review exposed an adaptive subclass that changed its threshold and reward while scoring. Its regression test passes. |
| Environment and trainer | Duplicate question IDs across 192 cached transitions, singleton optimization, and 2,304 padding comparisons | Position-based sampling gives cached and live callers the same question. A one-step rollout preserves the raw advantage. The packaged trainer retains input dtypes and padding values, including when token rows have different dtypes. |
| Evaluation | Three Lean proofs over arbitrary finite rational traces, 81 four-step numerical comparisons, and a read-only slotted dataclass with opaque metadata | The first-buzz masses plus terminal survival conserve total mass. Their weighted sum equals the backward conditional expectation. Changing the survival sign or dropping a continuation weight caused proof failure. Empty hazard traces still score zero. Nonempty S_q inputs with different step counts now fail explicitly. Expected Wins retains its common-prefix convention. |
| StopDFF | Compare gzip bytes with the original encoder, abort publication mid-stream, enumerate 768 small paired resamples, and run the package checker | Streamed output matches the historical encoding. A failed writer preserves the previous file. Batched means, medians, intervals and replicate order matched. The checker accepted the reference package and rejected all 48 mutations. |

### Results at the verified source revision

Source commit `4ac770022bc5d357b670e9a97f7525cd6e2bb9d2` was built and tested from a fresh Linux checkout. The Windows checks used the project's NumPy 2.4.6 pin.

| Check | Result |
| --- | --- |
| Original Linux suite | 3,671 passed, 38 skipped |
| Refactored Linux suite | 3,739 passed, 38 skipped. All 3,709 original test IDs retained their outcomes. All 68 added cases passed. |
| Supplemental verification | 294 Windows cases passed, including the 35 Windows-specific Linux skips. Two DSPy import checks passed with DSPy 3.5.0. The opt-in four-stage pipeline test passed. Across these runs every one of the 3,777 distinct cases passed. |
| Red–green replay | All 31 selected regression witnesses failed on the unmodified baseline and passed against the final source using the same test files. Additional characterization and adversarial tests check parity. |
| Static analysis | Ruff's E9, F63, F7 and F82 checks passed across all 333 tracked Python files. Broader Pyflakes analysis of the modified files found 11 existing diagnostics and zero new ones. Mypy passed on seven selected numerical, agent and I/O modules with silent import following and missing third-party stubs ignored. Python bytecode compilation and wheel build succeeded. The wheel supplied all six core package imports in an isolated smoke test. |
| Formal and runtime checks | Lean 4.35.0 accepted three first-buzz proofs and the existing paranoid comparators accepted 33 ACL theorems. The first-buzz proofs use only `propext`, `Classical.choice` and `Quot.sound`. The ACL reward tests exercise 126 legal Gym paths in the declared bounds. All eight reward mutants were detected, and both new first-buzz proof mutations were rejected. |

An initial Windows platform run used ambient NumPy 2.4.4. Four release-verification cases failed their version-bound checks. Repeating under 2.4.6 passed all four.

### Resource measurements

These are local measurements of individual workloads. Peak allocations use Python's `tracemalloc`. Expected Wins timings are medians of five batches in one Windows CPU process.

| Workload | Baseline | Refactor |
| --- | ---: | ---: |
| File gzip, 640 generated rows of random hex | 30.75 MB | 0.41 MB |
| In-memory gzip, 640 highly compressible rows | 21.02 MB | 0.38 MB |
| Paired bootstrap, 1,024 items and 1,001 replicates, excluding the shared index matrix | 24.66 MB | 3.53 MB |
| Take three words from a 250,000-token question | 13.31 MB | Under 4 KB |
| Expected Wins at 5 / 50 / 4,096 steps | 6.21 / 14.08 / 746.58 μs | 7.68 / 10.38 / 258.60 μs |

The proof in `formal/FirstBuzz.lean` covers exact rational algebra. The numerical tests exercise the NumPy implementation in float64. The existing ACL proofs cover their declared sampler, transition and reward properties. Filesystem crash durability, third-party libraries, model quality and the complete learning system fall outside these proofs.

## Reproduce the checks

From the repository root, run these commands in the project environment. The Python and Lean invocations use the same syntax in PowerShell and POSIX shells. Install the project dev dependencies for the test suite. The Lean command uses the 4.35.0 toolchain also selected by the opt-in verification workflow.

```sh
python -m pytest tests -q
python -m scripts.validate_stopdff_bucketed_sweep self-test
lean -DwarningAsError=true formal/FirstBuzz.lean
```

For the end-to-end smoke, use a disposable checkout with an empty smoke output directory. These stages write generated artifacts there.

```sh
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

The smoke used 36 train, one validation and two test questions. Threshold selection used validation. PPO collected 3,008 transitions for its configured 3,000-step budget. Both baseline and refactor used the training split to fit TF-IDF. All 1,204 numeric JSON values matched the original commit. Nineteen of 21 generated files matched byte for byte, including the complete final report. The remaining differences were run IDs, elapsed time and process addresses in SB3 checkpoint metadata. Policy and optimizer tensors were byte-identical. This run checks wiring and reproducibility. It has only two test questions.
