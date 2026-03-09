# Phase 4: PPO Training Pipeline - Research

**Researched:** 2026-02-25
**Domain:** PPO training with Stable-Baselines3, pipeline orchestration, smoke testing
**Confidence:** HIGH

## Summary

Phase 4 implements the PPO training pipeline with SB3 on belief feature observations from the TossupMCEnv built in Phase 2-3. The domain has well-established patterns: Stable-Baselines3 provides production-ready PPO with MlpPolicy for vectorized observations, and the qb-rl reference implementation demonstrates the exact integration pattern. The core challenge is orchestrating four pipeline scripts (build_mc_dataset, run_baselines, train_ppo, evaluate_all) with shared configuration and consistent artifact management.

The qb-rl codebase provides the complete reference implementation. Key patterns: (1) PPOBuzzer wrapper class around SB3's PPO model that adds episode trace generation for S_q computation, (2) shared _common.py module with config loading, JSON serialization, and path management, (3) consistent artifact structure under artifacts/smoke/ and artifacts/main/ directories, (4) smoke test mode with --smoke flag that reduces dataset size and hyperparameters for <2 minute validation.

**Primary recommendation:** Port qb-rl's four-script pipeline structure exactly, adapting only import paths from qb_env to qb_data/qb_env. The PPOBuzzer wrapper pattern is essential for S_q metric computation. Smoke test mode must complete full pipeline (MC build → baseline → train → evaluate) in under 2 minutes with 50 questions, 1000 timesteps, and t5-small likelihood model.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AGT-01 | MLP policy trained with SB3 PPO on belief feature observations | SB3 PPO with MlpPolicy is standard for vectorized envs. qb-rl PPOBuzzer wrapper demonstrates exact integration. |
| AGT-07 | Smoke test mode (--smoke) for fast pipeline validation with small dataset | qb-rl smoke mode: 50 questions, 1000 timesteps, 32 n_steps, 8 batch_size. Completes in <2 minutes. |
| CFG-03 | Four-stage pipeline scripts (build_mc, run_baselines, train_ppo, evaluate_all) execute without errors | qb-rl provides complete reference. Scripts share config via _common.py, write to consistent artifact directories. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| stable-baselines3 | 2.3.0+ | PPO implementation | Battle-tested RL library, vectorized envs, automatic advantage normalization, torch-based |
| gymnasium | 1.1.0+ | Environment interface | Already used in Phase 2, SB3 native support |
| torch | 2.3.0+ | Neural networks | SB3 backend, already in project |
| pyyaml | 6.0+ | Config parsing | Already used in Phase 1-3 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | <2.0.0 | Array operations | Episode trace computation, already constrained |
| dataclasses | stdlib | Structured data | EpisodeTrace, already used project-wide |
| pathlib | stdlib | Path management | Artifact directories, already project standard |
| json | stdlib | Artifact serialization | Episode traces, summaries |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| stable-baselines3 | custom PPO | SB3 is battle-tested with 10K+ GitHub stars, custom implementation risks bugs and takes weeks |
| MlpPolicy | CnnPolicy or MultiInputPolicy | Observation is 1D belief vector (K+6 features), MLP is correct choice |
| YAML config | Python config classes | YAML already used in Phase 1-3, changing breaks consistency |

**Installation:**
```bash
pip install stable-baselines3>=2.3.0
# torch, gymnasium, pyyaml already installed from Phase 1-3
```

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── build_mc_dataset.py      # Phase 1 (already exists)
├── run_baselines.py          # NEW: baseline agent orchestration
├── train_ppo.py              # NEW: PPO training with checkpointing
├── evaluate_all.py           # NEW: comprehensive evaluation + controls
└── _common.py                # NEW: shared utilities (config, JSON, paths)

agents/
├── ppo_buzzer.py             # NEW: PPOBuzzer wrapper around SB3
├── threshold_buzzer.py       # Phase 3 (already exists)
├── bayesian_buzzer.py        # Phase 3 (already exists)
└── __init__.py

artifacts/
├── main/                     # Full dataset artifacts
│   ├── mc_dataset.json
│   ├── baseline_*.json
│   ├── ppo_model.zip
│   ├── ppo_runs.json
│   └── evaluation_report.json
└── smoke/                    # Smoke test artifacts (same structure)
```

### Pattern 1: PPOBuzzer Wrapper Class
**What:** Wraps SB3's PPO model to add quiz bowl-specific episode execution and S_q trace generation
**When to use:** Required for computing S_q metric (Σ(c_t × g_t)) which needs per-step buzz probabilities
**Example:**
```python
# Source: qb-rl/agents/ppo_buzzer.py
class PPOBuzzer:
    def __init__(self, env: TossupMCEnv, learning_rate=3e-4, n_steps=128,
                 batch_size=32, n_epochs=10, gamma=0.99,
                 policy_kwargs=None, verbose=0):
        if policy_kwargs is None:
            policy_kwargs = {"net_arch": [64, 64]}

        self.env = env
        self.model = PPO(
            "MlpPolicy", env, verbose=verbose,
            learning_rate=learning_rate, n_steps=n_steps,
            batch_size=batch_size, n_epochs=n_epochs,
            gamma=gamma, policy_kwargs=policy_kwargs
        )

    def train(self, total_timesteps: int = 100_000):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: Path):
        self.model.save(str(path))

    @classmethod
    def load(cls, path: Path, env: TossupMCEnv):
        agent = cls(env=env)
        agent.model = PPO.load(str(path), env=env)
        return agent

    def run_episode(self, deterministic=False, seed=None):
        """Execute episode and return trace with c_trace, g_trace for S_q."""
        obs, info = self.env.reset(seed=seed)
        c_trace, g_trace, entropy_trace = [], [], []
        buzz_step, buzz_index = -1, -1
        total_reward = 0.0

        while not terminated and not truncated:
            probs = self.action_probabilities(obs)
            c_val = 1.0 - probs[0]  # action 0 = wait
            g_val = probs[gold_index+1] / c_val if c_val > 1e-12 else 0.0

            c_trace.append(c_val)
            g_trace.append(g_val)

            action = np.argmax(probs) if deterministic else np.random.choice(len(probs), p=probs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

        return EpisodeTrace(qid=..., buzz_step=..., correct=...,
                           c_trace=c_trace, g_trace=g_trace, ...)
```

### Pattern 2: Shared _common.py Module
**What:** Centralized utilities for config loading, JSON serialization, artifact paths
**When to use:** All four pipeline scripts import from _common to ensure consistency
**Example:**
```python
# Source: qb-rl/scripts/_common.py
from pathlib import Path
import yaml
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

def load_config(config_path: str | None = None) -> dict:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)

def save_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(to_serializable(data), f, indent=2)
    return path

def load_mc_questions(path: Path) -> list[MCQuestion]:
    raw = load_json(path)
    return [mc_question_from_dict(item) for item in raw]
```

### Pattern 3: Consistent Artifact Directory Structure
**What:** artifacts/smoke/ and artifacts/main/ with identical file structure
**When to use:** All scripts write to `ARTIFACT_DIR / split` where split = "smoke" or "main"
**Example:**
```python
# Source: qb-rl/scripts/train_ppo.py
split = "smoke" if args.smoke else "main"
out_dir = ARTIFACT_DIR / split
mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"

# Save outputs
agent.save(out_dir / "ppo_model")  # Creates ppo_model.zip
save_json(out_dir / "ppo_runs.json", traces)
save_json(out_dir / "ppo_summary.json", summary)
```

### Pattern 4: Build Likelihood from Config Helper
**What:** Factory function that constructs likelihood model based on config["likelihood"]["model"]
**When to use:** Both run_baselines.py and train_ppo.py need likelihood model with same logic
**Example:**
```python
# Source: qb-rl/scripts/train_ppo.py
def build_likelihood(config: dict, mc_questions):
    model_name = config["likelihood"]["model"]
    if model_name == "tfidf":
        corpus = [q.question for q in mc_questions] + \
                 [p for q in mc_questions for p in q.option_profiles]
        return TfIdfLikelihood(corpus_texts=corpus)
    if model_name == "openai":
        return OpenAILikelihood(model=config["likelihood"].get("openai_model"))
    return SBERTLikelihood(model_name=config["likelihood"].get("sbert_name"))
```

### Anti-Patterns to Avoid

- **Training without episode traces:** SB3 PPO's `.learn()` doesn't generate c_trace/g_trace needed for S_q. Must wrap with custom episode execution.
- **Inconsistent artifact paths:** Hard-coding paths like "ppo_model.zip" breaks smoke vs main split. Always use `out_dir / filename`.
- **Duplicate likelihood construction:** Copy-pasting likelihood build logic across scripts. Use shared helper function.
- **Forgetting to seed environment:** Episode execution without `env.reset(seed=seed)` makes evaluation non-deterministic.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PPO implementation | Custom advantage calculation, clipping, normalization | stable-baselines3 PPO | SB3 has 5+ years of bug fixes, vectorized envs, tensorboard logging, tested on 100+ environments |
| Episode trace aggregation | Manual loop tracking c_t, g_t per step | PPOBuzzer.run_episode() pattern from qb-rl | Already handles action→probability conversion, gold index tracking, truncation edge cases |
| Config override parsing | Custom argparse for nested YAML keys | Existing merge_overrides from Phase 1 | Already handles dot notation (data.K=5), type coercion, nested dicts |
| JSON serialization of dataclasses | Manual __dict__ or json.dumps | to_serializable helper with asdict | Handles nested dataclasses, lists, numpy arrays, prevents recursion issues |

**Key insight:** qb-rl has already solved all integration challenges (SB3 + Gymnasium + episode traces + S_q computation). Porting this code is faster and more reliable than reimplementing.

## Common Pitfalls

### Pitfall 1: Missing c_trace/g_trace in Episode Execution
**What goes wrong:** Using SB3's built-in evaluation without capturing per-step buzz probabilities means S_q metric cannot be computed
**Why it happens:** SB3's `.predict()` returns actions, not probability distributions. Need to manually extract from policy distribution.
**How to avoid:** Always use PPOBuzzer.run_episode() which calls `self.model.policy.get_distribution(obs_tensor)` to extract probs
**Warning signs:** Evaluation script crashes with "KeyError: 'c_trace'" or S_q computation returns None

### Pitfall 2: Smoke Test Still Too Slow (>2 minutes)
**What goes wrong:** Smoke test with 50 questions and 1000 timesteps takes 5+ minutes, making iteration painful
**Why it happens:** Using t5-large likelihood (3GB model load), n_steps=128 (too many rollout steps), or forgot to reduce batch_size
**How to avoid:** Smoke config MUST use t5-small (60M params, loads in 5s), n_steps=32, batch_size=8, total_timesteps=1000
**Warning signs:** Smoke test timing exceeds 2 minutes, memory usage >8GB, or likelihood model download starts

### Pitfall 3: Artifact Path Collisions Between Smoke and Main
**What goes wrong:** Running main pipeline after smoke test overwrites smoke artifacts, losing baseline comparisons
**Why it happens:** Hard-coded paths like "artifacts/ppo_model.zip" instead of split-aware paths
**How to avoid:** All scripts use `split = "smoke" if args.smoke else "main"` and write to `ARTIFACT_DIR / split`
**Warning signs:** Smoke test results disappear after main run, or evaluation can't find baseline_summary.json

### Pitfall 4: Forgetting to Load MC Dataset Before Training
**What goes wrong:** train_ppo.py crashes with "FileNotFoundError: mc_dataset.json" if build_mc_dataset.py wasn't run first
**Why it happens:** Pipeline stages have dependencies: build → baselines → train → evaluate
**How to avoid:** Document dependency chain clearly. Add existence check at script start: `if not mc_path.exists(): raise FileNotFoundError with helpful message`
**Warning signs:** Script crashes immediately on startup with path not found error

### Pitfall 5: Environment Not Compatible with SB3 Vectorization
**What goes wrong:** SB3 PPO wraps env in DummyVecEnv, but TossupMCEnv has incorrect reset() signature
**Why it happens:** SB3 expects Gymnasium API (obs, info = reset()), not old Gym API (obs = reset())
**How to avoid:** Phase 2 TossupMCEnv already uses correct Gymnasium API. Verify reset() returns tuple, step() returns 5-tuple.
**Warning signs:** TypeError about reset() or step() return values during PPO initialization

### Pitfall 6: Checkpoint Loading Fails with Version Mismatch
**What goes wrong:** SB3 model saved with version 2.3.0 can't load with version 2.2.0 or vice versa
**Why it happens:** SB3 saves torch state dict with version metadata, incompatible across minor versions
**How to avoid:** Pin stable-baselines3>=2.3.0 in requirements.txt. Save version info alongside checkpoint.
**Warning signs:** RuntimeError or KeyError during PPO.load(), or model behavior changes after loading

## Code Examples

Verified patterns from qb-rl source:

### train_ppo.py Main Entry Point
```python
# Source: qb-rl/scripts/train_ppo.py
def main():
    args = parse_args()
    config = load_config(args.config)
    split = "smoke" if args.smoke else "main"
    out_dir = ARTIFACT_DIR / split
    mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"
    mc_questions = load_mc_questions(mc_path)

    likelihood_model = build_likelihood(config, mc_questions)
    env = make_env_from_config(mc_questions=mc_questions,
                                likelihood_model=likelihood_model,
                                config=config)

    ppo_cfg = config["ppo"]
    agent = PPOBuzzer(
        env=env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        policy_kwargs=ppo_cfg.get("policy_kwargs", {"net_arch": [64, 64]}),
        verbose=1
    )

    total_timesteps = int(args.timesteps if args.timesteps else ppo_cfg["total_timesteps"])
    agent.train(total_timesteps=total_timesteps)
    agent.save(out_dir / "ppo_model")

    # Generate episode traces for S_q computation
    traces = [asdict(agent.run_episode(deterministic=args.deterministic_eval))
              for _ in range(len(mc_questions))]
    summary = {**summarize_buzz_metrics(traces), **calibration_at_buzz(traces)}

    save_json(out_dir / "ppo_runs.json", traces)
    save_json(out_dir / "ppo_summary.json", summary)
    print(f"Saved PPO model to: {out_dir / 'ppo_model'}.zip")
```

### run_baselines.py Baseline Orchestration
```python
# Source: qb-rl/scripts/run_baselines.py
def main():
    args = parse_args()
    config = load_config(args.config)
    split = "smoke" if args.smoke else "main"
    out_dir = ARTIFACT_DIR / split
    mc_questions = load_mc_questions(out_dir / "mc_dataset.json")

    likelihood_model = build_likelihood(config, mc_questions)
    beta = float(config["likelihood"].get("beta", 5.0))
    alpha = float(config["bayesian"].get("alpha", 10.0))
    thresholds = [float(x) for x in config["bayesian"]["threshold_sweep"]]

    # Threshold sweep
    threshold_runs = sweep_thresholds(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        thresholds=thresholds,
        beta=beta, alpha=alpha
    )

    # SoftmaxProfile and SequentialBayes for each threshold
    for threshold in thresholds:
        softmax_agent = SoftmaxProfileBuzzer(likelihood_model, threshold, beta, alpha)
        softmax_runs = [asdict(softmax_agent.run_episode(q)) for q in mc_questions]
        # ... store results

    # Floor agent (always buzz final)
    floor_agent = AlwaysBuzzFinalBuzzer(likelihood_model, beta)
    floor_runs = [asdict(floor_agent.run_episode(q)) for q in mc_questions]

    # Save all results
    save_json(out_dir / "baseline_threshold_runs.json", threshold_payload)
    save_json(out_dir / "baseline_summary.json", summary)
```

### evaluate_all.py Comprehensive Evaluation
```python
# Source: qb-rl/scripts/evaluate_all.py
def main():
    args = parse_args()
    config = load_config(args.config)
    split = "smoke" if args.smoke else "main"
    out_dir = ARTIFACT_DIR / split
    mc_questions = load_mc_questions(out_dir / "mc_dataset.json")

    likelihood_model = build_likelihood(config, mc_questions)
    threshold = pick_best_softmax_threshold(out_dir, default_threshold)

    # Main evaluation
    agent = SoftmaxProfileBuzzer(likelihood_model, threshold, beta, alpha)
    full_eval = evaluate_questions(mc_questions, agent)

    # Control experiments
    shuffle_eval = run_shuffle_control(mc_questions, evaluator)
    alias_eval = run_alias_substitution_control(mc_questions, alias_lookup, evaluator)
    choices_only = run_choices_only_control(mc_questions)

    # Load PPO and baseline results
    ppo_summary = load_json(out_dir / "ppo_summary.json") if exists else {}
    baseline_summary = load_json(out_dir / "baseline_summary.json") if exists else {}

    # Generate plots
    plot_entropy_vs_clue_index(entropy_traces, out_dir / "plots/entropy.png")
    plot_calibration_curve(confidences, outcomes, out_dir / "plots/calibration.png")
    save_comparison_table(table_rows, out_dir / "plots/comparison.csv")

    save_json(out_dir / "evaluation_report.json", report)
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.4.0+ (already configured from Phase 1-3) |
| Config file | tests/conftest.py with shared fixtures |
| Quick run command | `pytest tests/test_ppo_buzzer.py -x` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AGT-01 | PPOBuzzer trains successfully with SB3 PPO on belief observations | integration | `pytest tests/test_ppo_buzzer.py::test_ppo_training_runs -x` | ❌ Wave 0 |
| AGT-01 | PPOBuzzer.run_episode() generates c_trace, g_trace for S_q | unit | `pytest tests/test_ppo_buzzer.py::test_episode_trace_generation -x` | ❌ Wave 0 |
| AGT-01 | PPOBuzzer saves and loads checkpoints correctly | unit | `pytest tests/test_ppo_buzzer.py::test_checkpoint_save_load -x` | ❌ Wave 0 |
| AGT-07 | Smoke test mode completes full pipeline in <2 minutes | smoke | Manual: `time python scripts/train_ppo.py --smoke` | ❌ Wave 0 |
| CFG-03 | build_mc_dataset.py creates mc_dataset.json | integration | Manual: `python scripts/build_mc_dataset.py --smoke` | ✅ Phase 1 |
| CFG-03 | run_baselines.py produces baseline_summary.json | integration | Manual: `python scripts/run_baselines.py --smoke` | ❌ Wave 0 |
| CFG-03 | train_ppo.py produces ppo_model.zip and ppo_summary.json | integration | Manual: `python scripts/train_ppo.py --smoke` | ❌ Wave 0 |
| CFG-03 | evaluate_all.py produces evaluation_report.json | integration | Manual: `python scripts/evaluate_all.py --smoke` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_ppo_buzzer.py -x` (unit tests only, <10 seconds)
- **Per wave merge:** `pytest tests/ -v` (full suite including Phase 1-3 tests, <30 seconds)
- **Phase gate:** Full suite green + manual smoke test pipeline (`scripts/train_ppo.py --smoke` completes in <2 min)

### Wave 0 Gaps
- [ ] `tests/test_ppo_buzzer.py` — covers AGT-01 (PPO training, episode traces, checkpointing)
- [ ] `scripts/run_baselines.py` — covers CFG-03 (baseline orchestration)
- [ ] `scripts/train_ppo.py` — covers AGT-01, AGT-07, CFG-03 (PPO training, smoke mode)
- [ ] `scripts/evaluate_all.py` — covers CFG-03 (comprehensive evaluation)
- [ ] `scripts/_common.py` — shared utilities (config, JSON, paths)
- [ ] `agents/ppo_buzzer.py` — covers AGT-01 (PPOBuzzer wrapper with episode traces)

## Sources

### Primary (HIGH confidence)
- qb-rl/scripts/train_ppo.py — complete PPO training pipeline with smoke mode
- qb-rl/scripts/run_baselines.py — baseline agent orchestration pattern
- qb-rl/scripts/evaluate_all.py — comprehensive evaluation with controls
- qb-rl/scripts/_common.py — shared utilities for config, JSON, artifact paths
- qb-rl/agents/ppo_buzzer.py — PPOBuzzer wrapper with episode trace generation
- qb-rl/configs/default.yaml — PPO hyperparameters (n_steps=128, batch_size=32, etc.)
- qb-rl/configs/smoke.yaml — smoke mode settings (50 questions, 1000 timesteps)
- stable-baselines3 documentation — PPO API, MlpPolicy, model save/load

### Secondary (MEDIUM confidence)
- .planning/STATE.md — Phase 1-3 complete, interfaces established
- .planning/research/ARCHITECTURE.md — four-layer architecture, factory patterns
- .planning/research/PITFALLS.md — gradient accumulation, artifact path collisions
- scripts/build_mc_dataset.py — existing Phase 1 script structure (argparse, overrides)
- configs/default.yaml — existing YAML config structure

### Tertiary (LOW confidence, inferred)
- SB3 PPO typical hyperparameters — learning_rate=3e-4, n_epochs=10, gamma=0.99 are standard
- Smoke test timing expectations — <2 minutes is reasonable for 50 questions with small model

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — SB3 + Gymnasium already used in project, qb-rl proves integration works
- Architecture: HIGH — qb-rl provides complete reference implementation to port
- Pitfalls: HIGH — specific issues identified in qb-rl (c_trace requirement, artifact paths, smoke timing)

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (30 days — stack is stable, qb-rl code is fixed reference)
