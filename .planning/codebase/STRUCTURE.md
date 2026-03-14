# Structure

## Directory Layout

```
qanta-buzzer/
├── agents/                     # Buzzer agent implementations
│   ├── __init__.py             # Public API: ThresholdBuzzer, SoftmaxProfileBuzzer, PPOBuzzer
│   ├── _math.py                # Shared math utils (sigmoid)
│   ├── bayesian_buzzer.py      # SoftmaxProfileBuzzer, SequentialBayesBuzzer
│   ├── ppo_buzzer.py           # PPOBuzzer (SB3 PPO wrapper), PPOEpisodeTrace
│   ├── softmax_profile_buzzer.py  # Alternative profile buzzer (may be legacy)
│   └── threshold_buzzer.py     # ThresholdBuzzer, AlwaysBuzzFinalBuzzer, sweep_thresholds
│
├── evaluation/                 # Metrics and plotting
│   ├── __init__.py             # Public API: system_score, calibration_at_buzz, etc.
│   ├── controls.py             # Control experiments (shuffle, choices-only, alias substitution)
│   ├── metrics.py              # S_q, ECE, Brier score, buzz accuracy, per-category stats
│   └── plotting.py             # Calibration curves, entropy plots, comparison tables
│
├── models/                     # Likelihood models and feature extraction
│   ├── __init__.py             # Public API: LikelihoodModel subclasses, features, T5PolicyModel
│   ├── answer_profiles.py      # Re-export shim → qb_data.answer_profiles
│   ├── features.py             # extract_belief_features(), entropy_of_distribution()
│   ├── likelihoods.py          # LikelihoodModel ABC + TfIdf, SBERT, T5, OpenAI implementations
│   └── t5_policy.py            # T5PolicyModel, PolicyHead for end-to-end text policy
│
├── qb_data/                    # Canonical data layer
│   ├── __init__.py             # Public API: TossupQuestion, QANTADatasetLoader, normalize_answer
│   ├── answer_profiles.py      # AnswerProfileBuilder (TF-IDF profiles per answer)
│   ├── config.py               # YAML config loading, merge_overrides, smoke config support
│   ├── data_loader.py          # TossupQuestion dataclass, QANTADatasetLoader, CSV/HF parsing
│   ├── dataset_splits.py       # create_stratified_splits() with category balancing
│   ├── huggingface_loader.py   # load_from_huggingface() fallback for QANTA data
│   ├── mc_builder.py           # MCQuestion, MCBuilder with 4 anti-artifact guards
│   └── text_utils.py           # normalize_answer(), tokenize_text()
│
├── qb_env/                     # Gymnasium environment + qb-rl compatibility
│   ├── __init__.py             # Public API: TossupMCEnv, TextObservationWrapper, shims
│   ├── data_loader.py          # Re-export shim → qb_data.data_loader
│   ├── mc_builder.py           # Re-export shim → qb_data.mc_builder
│   ├── text_utils.py           # Re-export shim → qb_data.text_utils
│   ├── text_wrapper.py         # TextObservationWrapper for T5 policy pipeline
│   └── tossup_env.py           # TossupMCEnv (main Gymnasium environment)
│
├── training/                   # T5 policy training loops
│   ├── __init__.py
│   ├── train_ppo_t5.py         # PPO fine-tuning for T5 policy
│   └── train_supervised_t5.py  # Supervised warm-start for T5 policy
│
├── scripts/                    # Pipeline entrypoints
│   ├── __init__.py
│   ├── _common.py              # Shared helpers: config loading, JSON I/O, path constants
│   ├── build_mc_dataset.py     # Step 1: Load questions → build MC dataset → save
│   ├── run_baselines.py        # Step 2: Sweep threshold/Bayesian buzzers
│   ├── train_ppo.py            # Step 3: Train PPO on belief features
│   ├── evaluate_all.py         # Step 4: Full evaluation + controls + plots
│   ├── train_t5_policy.py      # T5 pipeline: supervised + PPO
│   ├── compare_policies.py     # T5 pipeline: policy comparison
│   ├── sweep_reward_shaping.py # Multi-seed reward parameter sweep
│   ├── run_smoke_pipeline.py   # End-to-end smoke test runner
│   └── test_mc_builder.py      # Standalone MC builder test script
│
├── tests/                         # pytest test suite (261 tests, 16 files)
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures: sample_mc_question, sample_config, sample_tfidf_env
│   ├── test_agents.py             # ThresholdBuzzer, SoftmaxProfileBuzzer, precomputed equivalence
│   ├── test_answer_profile_cache.py # Answer profile memoization cache
│   ├── test_build_mc_dataset.py   # MC dataset construction tests
│   ├── test_dataset_splits.py     # Split reproducibility (cross-process determinism)
│   ├── test_environment.py        # TossupMCEnv reset/step/reward, precomputed beliefs
│   ├── test_factories.py          # Factory function tests (make_env_from_config, etc.)
│   ├── test_features.py           # Belief feature extraction tests
│   ├── test_likelihoods.py        # TfIdf, SBERT, T5 scoring, cache persistence/memory
│   ├── test_mc_builder_topk.py    # Top-M argpartition distractor ranking
│   ├── test_metrics.py            # S_q, ECE, Brier, calibration_at_buzz
│   ├── test_ppo_buzzer.py         # PPOBuzzer training, run_episode, PPO calibration
│   ├── test_ppo_t5.py             # T5 PPO training tests
│   ├── test_qb_rl_bridge.py       # qb-rl compatibility import tests
│   ├── test_supervised_t5.py      # T5 supervised training tests
│   ├── test_t5_policy.py          # T5PolicyModel forward/backward tests
│   └── test_text_wrapper.py       # TextObservationWrapper tests
│
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Full production config
│   ├── smoke.yaml              # Minimal config for smoke tests
│   └── t5_policy.yaml          # T5 policy pipeline config
│
├── generated/                  # Generated outputs (poster, presentation)
├── checkpoints/                # Model checkpoints (gitignored runtime)
├── artifacts/                  # Pipeline output artifacts (runtime)
│
├── pyproject.toml              # Package definition, dependencies, pytest config
├── requirements.txt            # Flat dependency list (legacy)
├── setup.cfg                   # Setuptools config
├── AGENTS.md                   # Canonical repo contract for all coding agents
├── CLAUDE.md                   # Claude-specific shim (points to AGENTS.md)
├── README.md                   # Project documentation
│
├── _legacy/                    # Pre-modularization prototypes (not installed)
│   ├── config.py, dataset.py, environment.py, model.py
│   ├── main.py, train_supervised.py, train_ppo.py
│   ├── metrics.py, visualize.py, demo.py
│   └── verify_data_loader.py, test_csv_loader.py, test_imports.py
│
└── repomix/                    # AI-consumable repo snapshots
    ├── repomix-code.xml        # Core code + tests
    ├── repomix-docs.xml        # Documentation + planning
    └── repomix-smoke.xml       # Smoke artifact data
```

## Key File Locations

| What | Where |
|------|-------|
| Main Gymnasium environment | `qb_env/tossup_env.py` |
| Likelihood model hierarchy | `models/likelihoods.py` |
| Belief feature extraction | `models/features.py` |
| MC question construction | `qb_data/mc_builder.py` |
| Data loading + TossupQuestion | `qb_data/data_loader.py` |
| Pipeline shared helpers | `scripts/_common.py` |
| Default YAML config | `configs/default.yaml` |
| Test fixtures | `tests/conftest.py` |

## Naming Conventions

- **Packages:** snake_case (`qb_data`, `qb_env`)
- **Modules:** snake_case matching their primary class (`bayesian_buzzer.py` → `SoftmaxProfileBuzzer`)
- **Classes:** PascalCase (`TossupMCEnv`, `MCQuestion`, `LikelihoodModel`)
- **Functions:** snake_case (`extract_belief_features`, `normalize_answer`)
- **Private helpers:** leading underscore (`_text_key`, `_best_torch_device`, `_to_dict`)
- **Constants:** UPPER_SNAKE_CASE (`PROJECT_ROOT`, `ARTIFACT_DIR`, `DEFAULT_CONFIG`)
- **Config keys:** snake_case in YAML (`train_ratio`, `buzz_correct`, `max_length`)

## Where to Add New Code

| Adding... | Put it in... |
|-----------|-------------|
| New likelihood model | `models/likelihoods.py` (subclass `LikelihoodModel`), register in `build_likelihood_from_config()` |
| New buzzer agent | `agents/` (new file), export from `agents/__init__.py` |
| New evaluation metric | `evaluation/metrics.py` |
| New control experiment | `evaluation/controls.py` |
| New data source | `qb_data/` (new loader), integrate in `scripts/build_mc_dataset.py` |
| New pipeline script | `scripts/` (use `scripts/_common.py` helpers) |
| New test | `tests/test_*.py` (use fixtures from `tests/conftest.py`) |
