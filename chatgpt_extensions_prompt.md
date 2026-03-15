# ChatGPT Prompt: qanta-buzzer Codebase Audit + Three Extensions

**Attach these four files when pasting into ChatGPT:**

1. `repomix/repomix-code.md` (17,063 lines, 580K — full source code, configs, tests)
2. `repomix/repomix-docs.md` (21,808 lines, 976K — all docs and .planning/)
3. `repomix/repomix-smoke.md` (31,931 lines, 1.0M — current smoke artifacts)
4. `2026-03-13-195118-qanta-buzzer-optimization-cc-transcript3.txt` (4,024 lines — Claude Code optimization + audit remediation transcript)

---

**Prompt starts below this line. Copy everything from here to the end of the file.**

---

You are receiving the complete source code, documentation, planning history, current smoke-test artifacts, and the full engineering decision transcript of a Stanford CS234 final project called **qanta-buzzer**. It is a quiz bowl RL buzzer with two policy tracks:

1. **Belief-feature MLP pipeline:** TF-IDF/SBERT/T5 likelihood → softmax belief → Gymnasium env (Box(K+6) obs, Discrete(K+1) actions) → PPO (Stable-Baselines3) → evaluation (S_q, ECE, Brier, per-category)
2. **T5 end-to-end text policy:** T5EncoderModel → PolicyHead → supervised warm-start → custom PPO → TextObservationWrapper

The codebase has just been through a full optimization campaign (7 ranked items: precomputed beliefs, embedding cache persistence, baseline sweep collapse, profile memoization, top-M argpartition, TF-IDF cache unification, shuffle control precomputation) followed by an evidence-verified audit remediation pass. 261 tests pass, the smoke pipeline and T5 smoke are green, calibration metrics correctly use `top_p_trace` (not binary `g_trace`), dataset splits are deterministic via `hashlib.md5`, and legacy prototype files have been moved to `_legacy/`.

## Attached Files

The four attached files provide complete context:

- **repomix-code.md**: All source code (agents/, evaluation/, models/, qb_data/, qb_env/, training/, scripts/, tests/, configs/), plus README.md, AGENTS.md, CLAUDE.md, pyproject.toml. This is the canonical code snapshot.

- **repomix-docs.md**: All .planning/ files (STATE.md, ROADMAP.md, codebase analysis, phase plans, quick task summaries, audit remediation checklist), plus README.md, AGENTS.md, walkthrough.md. This is the complete planning and decision record.

- **repomix-smoke.md**: Current smoke artifact JSON/CSV files (mc_dataset.json, baseline_summary.json, ppo_summary.json, evaluation_report.json, comparison.csv). These show actual metric values from the corrected pipeline.

- **transcript3.txt**: The Claude Code engineering transcript covering the full optimization campaign and audit remediation. It contains every profiling measurement, every design decision rationale, every rejected alternative, every verification command and its output, and measured timing/memory numbers. This is the authoritative record of WHY each design decision was made. Use it to understand constraints and verified invariants before proposing changes that might violate them.

## Context from the Cursor Remediation Session

After the Claude Code optimization campaign, a separate Cursor session performed the audit remediation. Key facts from that session:

**Calibration fix (P0-1):** `calibration_at_buzz()` in `evaluation/metrics.py` was using `g_trace[buzz_step]` as "confidence" — but in baseline agents, `g_trace` is binary (1.0 if argmax==gold, else 0.0), not a probability. Fixed to use `top_p_trace` (max belief probability) with fallback to `c_trace`. `PPOEpisodeTrace` in `agents/ppo_buzzer.py` gained a `top_p_trace: list[float]` field populated from `max(self.env.belief)` at each step.

**Split reproducibility (P0-2):** `dataset_splits.py` used `hash(category)` which is randomized by PYTHONHASHSEED. Fixed with `hashlib.md5`. Cross-process determinism test confirms identical splits across PYTHONHASHSEED=0 and PYTHONHASHSEED=12345.

**Compare policies honesty (P0-3):** The MLP path uses config-driven env settings; the T5 path hardcodes `wait_penalty=0.1`. S_q semantics differ (belief-sigmoid vs wait-head probability). The docstring and README now state these caveats honestly instead of claiming "identical metrics."

**CI robustness (P0-4):** `scripts/ci.sh` auto-activates `.venv/` if present. `pyproject.toml` has `testpaths = ["tests"]`. 261/261 tests pass in 75s.

**Memory measurements:** TF-IDF embedding cache: 1.87 MB for 44 questions, projected ~42 MB for 1000 questions. Precomputed beliefs: 3.5 KB for 44 questions. `cache_memory_bytes` property added to `LikelihoodModel`.

**Known remaining issues:**
- `parse_overrides` in `build_mc_dataset.py` creates nested dicts that clobber parent config sections when merged (pre-existing bug, not introduced by remediation)
- Full 100k PPO training run not verified end-to-end
- SBERT/T5-large likelihood paths not exercised locally (require large model downloads)
- compare_policies S_q/reward comparisons are qualitative across architectures

## Current Reward System

The `TossupMCEnv._buzz_reward()` method supports three modes today:
- `simple`: +1.0 correct, -1.0 incorrect
- `time_penalty`: +buzz_correct/-buzz_incorrect with per-step wait_penalty and optional early_buzz_penalty scaled by progress
- `human_grounded`: 0.0 if agent buzzes after sampled human position; otherwise +buzz_correct/-buzz_incorrect

The environment already has `MCQuestion.human_buzz_positions` (list of (position, count) tuples from QANTA data) and `_sample_human_buzz()` for opponent modeling.

## Current Architecture Constraints

- `MCQuestion.options` is `List[str]` with fixed length K per dataset (default K=4)
- `TossupMCEnv` observation space is `Box(K+6)`, action space is `Discrete(K+1)` — both set at construction
- The `LikelihoodModel` ABC requires `score(clue_prefix, option_profiles) → np.ndarray` of shape (K,)
- Configs use flat YAML with `data`, `likelihood`, `environment`, `ppo`, `evaluation`, `bayesian`, `supervised` top-level sections

---

## Task 1: Codebase Audit

Before proposing any changes, analyze the attached codebase for:
- Remaining correctness issues, dead code, or inconsistencies
- Config/CLI contract mismatches (especially the known `parse_overrides` bug)
- Test coverage gaps that matter
- Architectural bottlenecks that will block the extensions below

Produce a prioritized issue list before proceeding.

---

## Task 2: Design Three Extensions

Design concrete, implementation-ready plans for these three extensions. For each, specify: (a) exact files to create or modify, (b) new classes/functions with signatures and docstrings, (c) config schema additions, (d) tests, (e) integration points with existing code.

### Extension A: "Expected Wins" Reward Function

Jordan Boyd-Graber's QANTA project uses an "Expected Wins" (EW) scoring metric that rewards buzzing optimally relative to an opponent model. The key idea: you get +10 for buzzing first and correctly, -5 for buzzing first and incorrectly, and the opponent buzzes with some known probability distribution over positions.

Design a new reward mode `expected_wins` for `TossupMCEnv` that:
1. Accepts an opponent buzz-position distribution (empirical histogram from human data, or a parametric model)
2. Computes reward as: R(t, correct) = P(opponent hasn't buzzed by t) × [+10 if correct, -5 if incorrect] + P(opponent buzzed before t) × [opponent_expected_value]
3. Supports configuration via `environment.opponent_buzz_model` in YAML configs
4. Integrates with the existing `MCQuestion.human_buzz_positions` field for empirical opponent modeling
5. Has an S_q-compatible trace structure
6. Includes a standalone `expected_wins_score()` function in `evaluation/metrics.py` for offline scoring

Reference repos for EW semantics:
- https://github.com/Pinafore/qb (original QANTA expected wins implementation)
- https://github.com/Pinafore/qanta-codalab (competition scoring)
- https://github.com/qanta-challenge/qanta25-starter (2025 starter kit)

### Extension B: Variable-K Answer Choices

Currently `MCQuestion` and `TossupMCEnv` hardcode K=4 answer options. The Gymnasium observation space is `Box(K+6)` and action space is `Discrete(K+1)`, both fixed at env construction.

Design a system that supports arbitrary K (2 to N) per question, where:
1. `MCBuilder` accepts `K` as a per-question or global parameter
2. `TossupMCEnv` handles variable-K across questions in the same pool (padding or dynamic reshaping)
3. The PPO policy (SB3 MLP) and T5 policy both handle variable-K
4. Evaluation metrics and baseline agents generalize cleanly
5. The observation space strategy is explicit: either (a) pad to max-K with masking, or (b) use a variable-length observation wrapper
6. Action masking for padded options is integrated

Reference repos:
- https://github.com/nbalepur/mcqa-artifacts (MCQA artifact analysis — how K affects difficulty)
- https://github.com/EleutherAI/lm-evaluation-harness (handles variable answer counts across benchmarks)
- https://github.com/Farama-Foundation/Gymnasium (variable action/observation spaces)

### Extension C: DSPy Integration for Rapid Iteration

Integrate DSPy (https://github.com/stanfordnlp/dspy) to allow declarative specification of the likelihood scoring, answer generation, and evaluation pipelines. The goal is to enable rapid iteration on prompts, few-shot examples, and chain-of-thought strategies without rewriting Python each time.

Design a system where:
1. A DSPy `Signature` replaces or wraps the `LikelihoodModel.score()` interface, allowing LM-based scoring with optimizable prompts
2. A DSPy module can generate answer profiles from question text (replacing or augmenting `AnswerProfileBuilder`)
3. DSPy's `BootstrapFewShot` or `MIPROv2` optimizers can tune the scoring prompts against the S_q metric
4. The existing TF-IDF/SBERT/T5 models remain available as non-DSPy baselines
5. A new config section `dspy` controls model selection, optimization strategy, and prompt caching
6. The DSPy-optimized scorer plugs into the existing environment and PPO training loop without changes to the RL side

Reference repos:
- https://github.com/stanfordnlp/dspy (core framework — v2.5+, `dspy.Signature`, `dspy.Module`, `dspy.BootstrapFewShot`, `dspy.MIPROv2`)
- https://github.com/huggingface/transformers (model backend)
- https://github.com/UKPLab/sentence-transformers (embedding models)
- https://github.com/qbreader/python-module (quiz bowl question data API — can provide training examples for DSPy optimization)
- https://github.com/huggingface/datasets (dataset loading patterns)
- https://github.com/DLR-RM/stable-baselines3 (RL training — must interoperate)

---

## Task 3: Implementation Plan

For each extension, produce:
1. A phased implementation plan (what to build first, what depends on what)
2. A dependency analysis (new pip packages, version constraints, optional vs required)
3. A test plan (unit tests, integration tests, smoke tests)
4. A config schema (YAML additions with defaults and validation)
5. Migration notes (what existing code breaks, what stays compatible)

Order the three extensions by implementation priority and explain why.

---

## Task 4: Cross-Extension Integration

Explain how the three extensions interact:
- Does variable-K change the EW reward computation?
- Can DSPy optimize the EW opponent model?
- How does DSPy's prompt optimization interact with the PPO training loop?
- What shared infrastructure (config, testing, evaluation) do all three need?

Produce a unified architecture diagram showing the current system and where each extension plugs in.

---

## Constraints

- All designs must be backward-compatible: existing smoke pipeline, 261 tests, and T5 smoke must continue to work unchanged when extensions are not activated.
- New dependencies must be optional (extras in pyproject.toml) unless they're already in the dependency tree.
- Every new code path must have at least one test.
- Config additions must have sensible defaults that preserve current behavior.
- Do not hand-wave implementation details. If a design requires a tricky Gymnasium space, show the space definition. If it needs a DSPy signature, write the signature class. If it changes reward math, write the formula with all terms defined.
- Respect the verified invariants from the transcript: calibration uses `top_p_trace`, splits use `hashlib.md5`, TF-IDF cache is vocab-specific (save_cache is a no-op), alias control must re-score live.
