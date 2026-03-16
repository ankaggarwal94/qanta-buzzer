# ChatGPT Prompt: Full-Codebase Review with DSPy Deep-Dive

**Attach these three files:**

1. `repomix/repomix-code.md` (18,876 lines — full source, configs, tests — **with line numbers**)
2. `repomix/repomix-docs.md` (21,907 lines — all .planning/, docs — **with line numbers**)
3. `repomix/repomix-smoke.md` (31,933 lines — current smoke artifact JSON/CSV — **with line numbers**)

All files include stable line numbers (`NNN: `) on every source line. Cite them as `file.py:NNN`.

---

**Copy everything below this line into ChatGPT.**

---

You are conducting a comprehensive code review of a Stanford CS234 final project codebase called **qanta-buzzer** at commit `fd34e25a`. The three attached Markdown files are line-numbered Repomix snapshots containing the complete repository: every source file, test, config, planning doc, and current smoke artifact.

This review has two layers:
1. **Whole-codebase pass** — check everything for correctness, contract violations, and test quality
2. **DSPy deep-dive** — give special scrutiny to the six DSPy-related files and their integration with the rest of the system

## Anti-hallucination rules

These rules are non-negotiable. Violating any of them invalidates the entire review.

1. **Every claim must cite `file.py:LINE`** using the line numbers in the attached files. "Line 42 of foo.py" is acceptable. "Somewhere in foo.py" is not.

2. **Quote the exact code** that is wrong. Do not paraphrase. Copy the line(s) from the attachment.

3. **If you cannot find evidence in the attachments, do not report it.** Speculation without a code citation is forbidden.

4. **Distinguish VERIFIED ISSUE from POTENTIAL CONCERN.** A verified issue has a specific code path and a provably wrong output. A potential concern is a design risk you can describe but cannot prove from the code alone.

5. **Do not re-report known issues.** The files `.planning/quick/patch-audit-issues.md` and `.planning/quick/extensions-master-run.md` document known risks. If you find something already listed there, put it in the ALREADY KNOWN section and move on.

6. **Style preferences are not bugs.** Only report issues where behavior is wrong, contracts are violated, or tests are provably weak.

## Repo context

- Quiz bowl RL buzzer with belief-feature MLP and T5 end-to-end policy tracks
- Three opt-in extensions just added: Expected Wins, Variable-K, DSPy
- 318 tests pass, 3 skipped (optional extras not installed)
- Smoke pipeline and T5 smoke both green
- Previous review already fixed: DSPyLikelihood inheritance, score shape validation, unused config key, stale config comment, misleading docstring, two weak tests

## Part 1: Whole-Codebase Review

For every package (`agents/`, `evaluation/`, `models/`, `qb_data/`, `qb_env/`, `training/`, `scripts/`, `tests/`, `configs/`), check:

### A. Contract fidelity
- Does every function do what its docstring says?
- Does every factory return the type it promises?
- Are all abstract methods implemented by all concrete subclasses?

### B. Error paths
- Are there code paths that silently produce wrong results instead of raising?
- Are there bare `except` or `except Exception` blocks that swallow real errors?
- Are there missing input validations that could produce confusing downstream failures?

### C. Numerical correctness
- Are the math formulas in `evaluation/metrics.py` correctly implemented?
- Does `expected_wins_score()` use the continuous V_self formula (not binary branching on g_trace)?
- Are softmax computations numerically stable (subtract max before exp)?
- Are there division-by-zero risks in belief normalization?

### D. Config-code alignment
- For every key in `configs/default.yaml` and `configs/smoke.yaml`: is there code that reads it?
- For every `config.get("key")` call in Python: is that key defined in at least one YAML file?
- Do config comments accurately describe the accepted values?

### E. Test quality
- Are there assertions that would pass for both correct AND incorrect implementations?
- Are there production code paths with zero test coverage that should have tests?
- Do test names accurately describe what they test?

## Part 2: DSPy Deep-Dive

Give special attention to these six files and their integration points:

### Production code
1. **`models/dspy_likelihood.py`** — The core DSPy scorer wrapper
2. **`qb_data/dspy_answer_profiles.py`** — Optional LM-augmented answer profiles
3. **`scripts/optimize_dspy.py`** — Offline DSPy compile/optimize workflow

### Test code
4. **`tests/test_dspy_likelihood.py`**
5. **`tests/test_dspy_answer_profiles.py`**
6. **`tests/test_dspy_optimize.py`**

### Integration points
7. **`models/likelihoods.py`** — the `build_likelihood_from_config()` factory, specifically the `model_name == "dspy"` branch
8. **`tests/test_factories.py`** — `TestDSPyFactoryIntegration`
9. **`configs/default.yaml`** — the `dspy:` section
10. **`pyproject.toml`** — the `[project.optional-dependencies]` `dspy` extra

For each DSPy file, answer these specific questions:

#### `models/dspy_likelihood.py`
- Does `DSPyLikelihood` correctly inherit `LikelihoodModel`? (This was just fixed — verify the fix is sound.)
- Does `score()` enforce the `(K,)` shape contract? (Also just fixed — verify.)
- Is `_score_cache_key()` collision-resistant? Could two different `(clue, options, fingerprint)` triples produce the same key?
- Does `save_cache()` / `load_cache()` correctly handle the `_score_cache` dict where keys are SHA-256 hex strings? (Note: `np.savez_compressed` uses keys as numpy array names — are SHA-256 hex strings valid numpy identifier-style keys?)
- Does `embed_and_cache()` raising `NotImplementedError` cause any problems for callers that call `embed_and_cache` on a generic `LikelihoodModel`? Grep for all call sites of `embed_and_cache` and `precompute_embeddings` to check.
- Is the `cache_memory_bytes` property consistent with the parent class implementation?

#### `qb_data/dspy_answer_profiles.py`
- Does `build_dspy_profiles()` actually preserve leave-one-out discipline, or does it just claim to?
- If the DSPy LM call fails for one answer, does the fallback to `existing_profiles` work correctly?
- Is the `max_answers` cap applied correctly (first N get augmentation, rest get extractive)?

#### `scripts/optimize_dspy.py`
- Is `compile_dspy_scorer()` actually functional, or is it a skeleton that would crash on real input?
- Does the `metric` lambda in the optimizer initialization actually measure anything useful, or is it a placeholder `lambda: 1.0`?
- Is the `program_fingerprint` derivation deterministic and stable across runs with the same config?
- Could `build_dspy_trainset()` produce training examples that leak test-set information?

#### Factory integration
- Does the placeholder scorer in `build_likelihood_from_config()` match the `score()` contract?
- If someone sets `likelihood.model: dspy` in config but does NOT have the `dspy` package installed, what happens? Trace the exact error path.
- If someone sets `likelihood.model: dspy` and also sets `dspy.cache_dir: null`, does `DSPyLikelihood.__init__` handle that correctly?

#### Test quality for DSPy
- Does `test_changed_fingerprint_invalidates` now actually prove what it claims? (Was just rewritten — verify the new version.)
- Is `test_persistence_roundtrip` robust, or could it pass even if persistence were broken?
- Does `test_score_shape_validation` cover all failure modes (wrong length, wrong ndim, empty)?
- Are there any DSPy code paths that have ZERO test coverage?

## Part 3: Cross-Cutting Concerns

### A. LikelihoodModel hierarchy after DSPy addition
- List every concrete subclass of `LikelihoodModel` and verify each implements `score()` and `_embed_batch()`.
- For `DSPyLikelihood` specifically: it inherits `embed_and_cache()` from the base class but overrides it to raise. Is the base class's `__init__` (which initializes `self.embedding_cache = {}`) wasteful but harmless, or does it cause real problems?
- Does `DSPyLikelihood.save_cache()` shadow the base class `save_cache()`? If so, does it have compatible semantics?

### B. Variable-K + DSPy interaction
- If `variable_K=True` and `likelihood.model=dspy`, does the DSPy scorer receive the correct per-question option count?
- Does shape validation in `DSPyLikelihood.score()` still work when K varies per question?

### C. Expected Wins + DSPy interaction
- If `reward_mode=expected_wins` and `likelihood.model=dspy`, does the env correctly consume DSPy scores?
- Does `DSPyLikelihood.score()` output integrate correctly with `_softmax()` and `_compute_belief()` in `tossup_env.py`?

## Output format

```
## VERIFIED ISSUES
### [SEVERITY] Title
- **File:** path/to/file.py:LINE
- **Code:** `exact quoted code`
- **Problem:** what it does vs what it should do
- **Fix:** one-line description

## POTENTIAL CONCERNS
### [SEVERITY] Title
- **File:** path/to/file.py:LINE
- **Evidence:** what you observed
- **Risk:** when this could become a real problem

## VERIFIED CLEAN
Areas examined and found correct, with brief evidence.

## ALREADY KNOWN
Issues found that are documented in .planning/ files.

## DSPy-SPECIFIC FINDINGS
Separate section for DSPy deep-dive results, organized by file.
```

Be thorough but honest. A short report with two real bugs is worth more than a long report padded with speculation. If the codebase is mostly correct, say so clearly and explain what you verified.
