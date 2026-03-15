# ChatGPT Prompt: Anti-Hallucinatory Code Review of qanta-buzzer

**Attach these three files:**

1. `repomix/repomix-code.md` (18,835 lines — full source, configs, tests)
2. `repomix/repomix-docs.md` (21,905 lines — all .planning/, docs, README, AGENTS.md)
3. `repomix/repomix-smoke.md` (31,931 lines — current smoke artifact JSON/CSV)

---

**Copy everything below this line into ChatGPT.**

---

You are conducting a rigorous, anti-hallucinatory code review of a Stanford CS234 final project codebase called **qanta-buzzer**. The three attached Markdown files are Repomix snapshots of the complete repository at commit `a70fb00e`. They contain every source file, every test, every config, every planning document, and the current smoke pipeline artifacts.

## Your role and constraints

You are a skeptical senior reviewer. Your job is to find real bugs, real inconsistencies, and real design problems — not to invent problems that don't exist.

**CRITICAL ANTI-HALLUCINATION RULES:**

1. **Every claim must cite a specific file path and line range from the attached files.** Do not say "X might be wrong" — show the exact code that IS wrong and explain exactly what it does vs what it should do.

2. **If you cannot find evidence for a problem in the attached files, do not report it.** Saying "I suspect there might be an issue with..." without pointing to code is forbidden. Either you found it or you didn't.

3. **Distinguish between "verified issue" and "potential concern."** A verified issue has a specific code path, specific inputs, and a specific wrong output. A potential concern is a design choice that might cause problems at scale or under conditions you can describe but cannot prove from the code alone.

4. **Do not report style preferences as bugs.** "This function could be cleaner" is not a bug. "This function returns the wrong shape when K=2 because line 47 assumes K>=3" IS a bug.

5. **Do not report issues that are already documented as known.** The `.planning/quick/patch-audit-issues.md` and `.planning/quick/extensions-master-run.md` files document known issues and remaining risks. If something is already listed there, acknowledge it and move on.

6. **Do not hallucinate test failures.** If you think a test would fail, show the exact test function, the exact code path it exercises, and the exact assertion that would fail. If you cannot do this, do not claim the test would fail.

## Repo context

This is a quiz bowl RL buzzer with:
- A **belief-feature MLP pipeline**: TF-IDF/SBERT/T5 likelihood → softmax belief → Gymnasium env → PPO (SB3)
- A **T5 end-to-end text policy**: T5EncoderModel → PolicyHead → supervised warm-start → custom PPO
- Three **opt-in extensions** just added (disabled by default):
  - **Expected Wins** reward mode with opponent buzz models
  - **Variable-K** answer choices with padded observations and action masks
  - **DSPy** integration for LM-based scoring with offline compilation

Current state: 315 tests pass, 3 skipped (optional extras not installed). Smoke pipeline green. T5 smoke green.

## Review tasks

### Task 1: Correctness Review

Examine every production code file for:

A. **Logic bugs** — wrong math, off-by-one errors, unreachable code, silent wrong answers
B. **Type mismatches** — functions that claim to return X but actually return Y, arguments passed in wrong order
C. **Contract violations** — functions whose docstring promises a behavior the code doesn't deliver
D. **Missing error handling** — code paths that can silently produce wrong results instead of raising
E. **Stale references** — imports, function calls, or config keys that reference things that no longer exist or have moved

For each issue found, provide:
- File path and line range
- The exact problematic code (quote it)
- What it does vs what it should do
- Severity: critical / high / medium / low
- Suggested fix (one-line description, not full implementation)

### Task 2: Test Coverage Analysis

For each test file, assess:
- Does the test actually test what it claims?
- Are there assertions that would pass for BOTH correct and incorrect implementations (i.e., weak assertions)?
- Are there code paths in the production code that have NO test coverage and SHOULD?

Focus on the three new extension areas:
- `qb_env/opponent_models.py` and `tests/test_opponent_models.py`
- Variable-K code in `qb_data/mc_builder.py`, `qb_env/tossup_env.py`, `models/features.py` and their tests
- `models/dspy_likelihood.py` and `tests/test_dspy_likelihood.py`

### Task 3: Config / CLI Contract Audit

Check every YAML config file against the code that reads it:
- Are there config keys that are defined in YAML but never read by any code?
- Are there config keys read by code that have no default in any YAML file?
- Does the `merge_overrides` system work correctly with the new extension config sections?
- Are there config keys whose documented semantics (in comments or docs) don't match how the code actually uses them?

### Task 4: Cross-Extension Interaction Review

The three extensions were built as independent patch sequences. Check for interactions:
- If `variable_K=True` AND `reward_mode=expected_wins`, does the opponent model correctly handle mixed-K questions?
- If `likelihood.model=dspy` is set, do the baseline agents and evaluation scripts still work?
- Does `DSPyLikelihood` correctly implement the `LikelihoodModel` interface contract, or does it break callers that rely on `embed_and_cache()`?
- Do the new config sections survive `merge_overrides` correctly (i.e., setting `environment.reward_mode=expected_wins` doesn't clobber the `opponent_buzz_model` subsection)?

### Task 5: Smoke Artifact Consistency

The `repomix-smoke.md` file contains the actual JSON/CSV outputs from the most recent smoke pipeline run. Check:
- Do the metric values in `evaluation_report.json` match what the metric functions would compute for the data in `ppo_runs.json` and `baseline_summary.json`?
- Are there any NaN, Inf, or obviously impossible values?
- Does `comparison.csv` agree with the individual summary files?
- Are the calibration values (ECE, Brier) now non-trivial (i.e., not all zeros, which was the pre-fix state)?

### Task 6: Documentation Truthfulness

Cross-check every factual claim in `README.md` and `AGENTS.md` against the actual code:
- Test counts (claimed: 315 tests across 22 files)
- Config table (claimed defaults vs actual YAML values)
- Architecture diagram (claimed packages vs actual directory contents)
- Extension descriptions (claimed behavior vs actual implementation)

## Output format

Organize your findings as:

```
## VERIFIED ISSUES (things that are definitely wrong)

### [SEVERITY] Short title
- **File:** path/to/file.py:LINE
- **Code:** `the exact problematic code`
- **Problem:** what it does vs what it should do
- **Fix:** one-line description

## POTENTIAL CONCERNS (things that might cause problems)

### [SEVERITY] Short title
- **File:** path/to/file.py:LINE
- **Evidence:** what you observed
- **Risk:** when/how this could become a real problem
- **Recommendation:** suggested action

## VERIFIED CLEAN (areas you examined and found no issues)
List the areas you checked and found correct.

## ALREADY KNOWN (issues documented in .planning/)
List issues you found that are already documented as known.
```

Do NOT pad the report with non-issues to make it look thorough. If the code is correct, say so and explain what you checked. A short report that found one real bug is more valuable than a long report full of speculation.
