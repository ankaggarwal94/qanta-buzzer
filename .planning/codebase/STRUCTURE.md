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
│   ├── __init__.py             # Public API: system_score, expected_wins_score, calibration_at_buzz
│   ├── controls.py             # Control experiments (shuffle, choices-only, alias substitution)
│   ├── metrics.py              # S_q, Expected Wins, ECE, Brier, buzz accuracy, per-category
│   └── plotting.py             # Calibration curves, entropy plots, comparison tables
│
├── models/                     # Likelihood models and feature extraction
│   ├── __init__.py             # Public API: LikelihoodModel subclasses, features, T5PolicyModel
│   ├── answer_profiles.py      # Re-export shim → qb_data.answer_profiles
│   ├── dspy_likelihood.py      # DSPyLikelihood (LM-based scorer with score cache)
│   ├── features.py             # extract_belief_features(), extract_padded_belief_features()
│   ├── likelihoods.py          # LikelihoodModel ABC + TfIdf, SBERT, T5, OpenAI, factory
│   └── t5_policy.py            # T5PolicyModel, PolicyHead for end-to-end text policy
│
├── qb_data/                    # Canonical data layer
│   ├── __init__.py             # Public API: TossupQuestion, QANTADatasetLoader, normalize_answer
│   ├── answer_profiles.py      # AnswerProfileBuilder (TF-IDF profiles per answer)
│   ├── config.py               # YAML config loading, merge_overrides, smoke config support
│   ├── data_loader.py          # TossupQuestion dataclass, QANTADatasetLoader, CSV/HF parsing
│   ├── dataset_splits.py       # create_stratified_splits() with category balancing
│   ├── dspy_answer_profiles.py # Optional DSPy LM-augmented answer profiles
│   ├── huggingface_loader.py   # load_from_huggingface() fallback for QANTA data
│   ├── mc_builder.py           # MCQuestion, MCBuilder with guards and variable-K
│   └── text_utils.py           # normalize_answer(), tokenize_text()
│
├── qb_env/                     # Gymnasium environment + qb-rl compatibility
│   ├── __init__.py             # Public API: TossupMCEnv, TextObservationWrapper, shims
│   ├── data_loader.py          # Re-export shim → qb_data.data_loader
│   ├── mc_builder.py           # Re-export shim → qb_data.mc_builder
│   ├── opponent_models.py      # OpponentBuzzModel protocol + logistic/empirical impls
│   ├── text_utils.py           # Re-export shim → qb_data.text_utils
│   ├── stop_only_env.py         # StopOnlyEnv: Discrete(2) WAIT/BUZZ wrapper
│   ├── text_wrapper.py         # TextObservationWrapper for T5 policy pipeline
│   └── tossup_env.py           # TossupMCEnv (main env: EW, variable-K, action masks)
│
├── training/                   # T5 policy training loops
│   ├── __init__.py
│   ├── hazard_pretrain.py      # Hazard bridge loss utilities
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
│   ├── optimize_dspy.py        # Offline DSPy compile/optimize workflow
│   ├── test_mc_builder.py      # Standalone MC builder test script
│   ├── run_full_pipeline.sh    # Full 19-phase pipeline (4-wave DAG, forces tfidf, PYTHONUNBUFFERED)
│   ├── manual-smoke.sh         # Four-stage smoke wrapper (venv-aware, python3)
│   └── ci.sh                   # CI entry point (runs pytest)
│
├── tests/                            # pytest test suite (342 tests, 24 files)
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures
│   ├── test_action_space_alignment.py # Factored action semantics guards
│   ├── test_agents.py                # Baseline agents, precomputed equivalence, K-agnostic
│   ├── test_answer_profile_cache.py  # Answer profile memoization cache
│   ├── test_build_mc_dataset.py      # MC dataset construction, CLI overrides
│   ├── test_dataset_splits.py        # Split reproducibility (cross-process determinism)
│   ├── test_dspy_answer_profiles.py  # DSPy answer profile augmentation (importorskip)
│   ├── test_dspy_likelihood.py       # DSPyLikelihood cache, shape, inheritance
│   ├── test_dspy_optimize.py         # Offline DSPy compile trainset (importorskip)
│   ├── test_environment.py           # TossupMCEnv: reward modes, EW, variable-K, masks
│   ├── test_factories.py             # Factories including DSPy dispatch
│   ├── test_features.py              # Belief features, padded features
│   ├── test_hazard_pretrain.py       # Hazard bridge survival terms and NLL loss
│   ├── test_likelihoods.py           # TfIdf, SBERT, T5 scoring, cache, memory
│   ├── test_mc_builder_topk.py       # Top-M argpartition distractor ranking
│   ├── test_mc_builder_variable_k.py # Variable-K dataset build
│   ├── test_metrics.py               # S_q, Expected Wins, ECE, Brier, calibration
│   ├── test_opponent_models.py       # Logistic/empirical opponent models
│   ├── test_ppo_buzzer.py            # PPOBuzzer training, traces, MaskablePPO
│   ├── test_ppo_t5.py                # T5 PPO training
│   ├── test_qb_rl_bridge.py          # qb-rl compatibility imports
│   ├── test_supervised_t5.py         # T5 supervised training
│   ├── test_t5_policy.py             # T5PolicyModel forward/backward
│   ├── test_text_wrapper.py          # TextObservationWrapper, K=3 formatting
│   └── test_variable_k_integration.py # Mixed-K build→env→baseline integration
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
└── repomix/                    # AI-consumable repo snapshots (XML + Markdown, line-numbered)
    ├── repomix-code.{xml,md}   # Core code + tests
    ├── repomix-docs.{xml,md}   # Documentation + planning
    └── repomix-smoke.{xml,md}  # Smoke artifact data
```

## Key File Locations

| What | Where |
|------|-------|
| Main Gymnasium environment | `qb_env/tossup_env.py` |
| Likelihood model hierarchy | `models/likelihoods.py` |
| DSPy likelihood scorer | `models/dspy_likelihood.py` |
| Opponent buzz models | `qb_env/opponent_models.py` |
| Belief feature extraction | `models/features.py` |
| MC question construction | `qb_data/mc_builder.py` |
| Data loading + TossupQuestion | `qb_data/data_loader.py` |
| Offline DSPy compile | `scripts/optimize_dspy.py` |
| Pipeline shared helpers | `scripts/_common.py` |
| Full pipeline script | `scripts/run_full_pipeline.sh` |
| Smoke pipeline wrapper | `scripts/manual-smoke.sh` |
| Default YAML config | `configs/default.yaml` |
| Full pipeline runbook | `docs/full-pipeline-runbook.md` |
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
