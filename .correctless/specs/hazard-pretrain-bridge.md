# Spec: Hazard-Pretrain Warm-Start Bridge

## Metadata
- **Task**: hazard-pretrain warm-start bridge for the T5 buzz policy
- **Intensity**: standard
- **Recommended-intensity**: standard
- **Intensity reason**: no security/`hooks/`/TB-xxx signals; fresh Correctless install (antipattern/QA/calibration signals dormant)
- **Override**: none
- **Branch**: feature/hazard-pretrain-bridge

## What
Wire the currently-unimplemented `--hazard-pretrain` bridge in the T5 buzz-policy trainer. A new `run_hazard_pretrain(...)` pass slots **between** supervised warm-start (Phase 1) and PPO (Phase 2) in `scripts/train_t5_policy.py::main()`. It loads the supervised checkpoint, and for each training question forwards the T5 policy over that question's `cumulative_prefixes` to obtain per-prefix stop-probabilities (from the wait head) and per-prefix answer-NLL (from the answer head vs `gold_index`), then minimizes the **existing** `training/hazard_pretrain.py::hazard_expected_nll_loss` — teaching the buzz/stop head *when to buzz* before PPO. It saves a checkpoint that PPO then loads. This turns three currently-inert CLI flags (`--hazard-pretrain`, `--beta-terminal`, `--freeze-answer-head`) into real behavior. Problem it solves: the flag is advertised but raises `NotImplementedError`, and the survival-loss math exists with no training loop to use it.

## Rules
- **R-001** [unit]: With `--hazard-pretrain` set, `validate_args` no longer raises `NotImplementedError` and returns normally (isolated `validate_args` logic — unit-level). (Guards the primary regression: `tests/test_hazard_pretrain.py::test_hazard_pretrain_flag_raises_not_implemented` currently asserts the raise and MUST be flipped.)
- **R-002** [integration]: When `--hazard-pretrain` is set, `main()` calls `run_hazard_pretrain` after the supervised phase and before PPO, and passes its returned checkpoint path to `run_ppo_training(pretrained_model_path=...)`. When the flag is absent, `run_hazard_pretrain` is NOT called and PPO consumes the supervised checkpoint unchanged. (Test via monkeypatching both trainers at the script entry.)
- **R-003** [integration]: `run_hazard_pretrain(config, train_questions, *, pretrained_model_path, beta_terminal=1.0, freeze_answer_head=False)` loads the model via `T5PolicyModel.load_pretrained`, runs the warm-start loop, saves to `checkpoints/hazard/best_model` (`config["checkpoint_dir"]/"hazard"/"best_model"`), and returns that path; the saved dir is re-loadable via `T5PolicyModel.load_pretrained` (contains `policy_head.pt`). (CPU smoke on t5-small.)
- **R-004** [unit]: For a question with T prefixes, the pass computes `stop_probs = softmax(wait_logits, dim=-1)[:, 1]` shape `[1, T]` (P(BUZZ)) and `nll_per_prefix = cross_entropy(answer_logits, gold_index broadcast, reduction="none")` shape `[1, T]`, and `hazard_expected_nll_loss(stop_probs, nll_per_prefix, beta_terminal=beta_terminal)` returns a **finite scalar** (`torch.isfinite`).
- **R-005** [unit]: With `freeze_answer_head=True`, every param in `model.policy_head.answer_head` has `requires_grad is False` and its weights are unchanged (`torch.allclose`) after the pass; with `freeze_answer_head=False`, wait-head params change. Documented caveat (not a bug): answer-NLL gradient still reaches the **shared T5 encoder**, so the flag freezes only the answer head, matching its literal name.
- **R-006** [unit]: `--beta-terminal` (argparse default `1.0`) is threaded into `hazard_expected_nll_loss(..., beta_terminal=<value>)`; it weights the never-buzz survival mass `survival[:, -1]`. A larger value yields a strictly larger loss for the same stop_probs/nll with residual survival mass > 0.
- **R-007** [unit]: `run_hazard_pretrain` reads only these `MCQuestion` fields — `cumulative_prefixes: list[str]`, `options: list[str]`, `gold_index: int` — whose names/types are pinned to `qb_data/mc_builder.py::MCQuestion` as the authoritative producer (AP-031 format-pinning). A question with a single prefix (T=1) still produces a finite loss (no crash).
- **R-008** [unit] (added in review — F1/F2): `run_hazard_pretrain` fails loud — raises a clear error, never silently returns a checkpoint — when `pretrained_model_path` does not exist or is not a loadable policy directory. Empty `train_questions` is a no-op: it returns a saved copy of the loaded model unchanged (no crash, no PPO-breaking output). Questions with zero prefixes (T=0) are skipped rather than crashing.

## Won't Do
- **Multi-question batching / the loss `mask=` path.** MVP is B=1 per question (the padded/masked path in `hazard_expected_nll_loss` stays untested/unused here).
- **A new `configs/*.yaml` `hazard:` section.** Hyperparameters come from CLI flags + reuse of the existing config (lr falls back to `supervised_lr`; epochs default small, e.g. 1, overridable via `config.get("hazard_epochs", 1)`).
- **Forcing `stop_prob = 1` at the final prefix.** Keep the real never-buzz mass so `beta_terminal` genuinely bites (per the loss's design).
- **Driving the real `TossupMCEnv` step-by-step.** Use `cumulative_prefixes` directly (matches the PPO observation distribution).
- **Validating that the warm-start *improves* PPO** (convergence/S_q/calibration). Requires full-scale CUDA runs (Device 2 / RTX 5090); out of scope here — this feature ships **smoke-validated (plumbing only)**.

## Risks
- **`--freeze-answer-head` still trains the shared encoder via answer-NLL grad** — Mitigate-by-documentation (accepted): matches the flag's literal name; a strict encoder-freeze variant is deferred. Documented in the docstring + AGENTS.md.
- **T=1 (single-prefix) questions make the survival model degenerate** — Accepted: rare; loss stays finite; the smoke test includes a ≥2-prefix question and a T=1 case for the no-crash guard (R-007).
- **Smoke-only validation** — Accepted + documented: `/cverify` confirms plumbing (flag runs, finite loss, freeze works, loadable checkpoint, PPO consumes it), NOT training efficacy. Stated in spec, docstring, and AGENTS.md so no one mistakes it for a validated ML win.
- **Fixture/producer format drift (AP-031)** — Mitigated by R-007 (pin `MCQuestion` field names to `qb_data/mc_builder.py`).
- **Untrusted checkpoint deserialization (`torch.load` pickle)** — N/A / accepted (review F4): `T5PolicyModel.load_pretrained` uses `torch.load` (pickle) on the checkpoint, but `pretrained_model_path` is produced in-pipeline by the trusted supervised phase, never user-supplied. Not an attack surface in this training workflow.

## Open Questions
- **Warm-start length**: default `hazard_epochs = 1` over `train_questions` for the MVP (config-overridable). Not blocking — a hyperparameter, not a contract.
- **Loop home**: `run_hazard_pretrain` co-located in `training/hazard_pretrain.py` (owns the math) vs. a sibling `training/train_hazard_t5.py` (mirrors `train_supervised_t5.py`/`train_ppo_t5.py`). Chosen: `hazard_pretrain.py` (cohesive, avoids a near-empty module). Reversible.
