# Spec — hazard-pretrain efficacy eval harness (ready for `/cspec`)

Hardened via a two-critic pass (experimental-validity + RL-metrics/feasibility) grounded in the real code. v0's naive assumptions that were **factually wrong** and are corrected below: (a) there is **no training seed**; (b) **Expected Wins is not free** (needs an opponent model); (c) smoke writes **no `best_model/`**; (d) `evaluate_t5_policy` **discards per-question** metrics.

Goal: quantify whether `--hazard-pretrain` improves the T5 buzz policy, via a paired WITH/WITHOUT (+ optional control) comparison, reusing existing eval code. Both arms are the SAME T5-policy architecture → strictly apples-to-apples (unlike `compare_policies.py`, whose by-architecture S_q/reward caveat does NOT apply here — confirmed).

## Prerequisite code enablers (build these first, via `/ctdd`)
The harness is NOT pure orchestration — two small enablers must land first, or the comparison is invalid:
- **E-1 training seed.** `scripts/train_t5_policy.py` has no `--seed` and nothing seeds torch/numpy/random; PPO samples unseeded (`train_ppo_t5.py:371`, `deterministic=False`). Add a `--seed` that seeds torch+numpy+random before supervised/hazard/PPO, **separate from `data.seed`** (which drives only the split, `train_t5_policy.py:530`). Hold the data split FIXED across all arms/seeds; vary only the training seed. (TDD: two seeds → different PPO trajectories; split unchanged.)
- **E-2 per-question eval surfacing.** `evaluate_t5_policy` returns aggregate `mean_sq`/`avg_buzz_pos` only; per-question `sq_scores`/`buzz_positions` are discarded locals (`compare_policies.py:567,572-584`). Add an option to return per-question runs keyed by `qid` (mirror the `summary["runs"]=runs` pattern + `controls.py:390 bootstrap_ci`). Needed for the paired bootstrap (R-5) and the buzz-position histogram (R-3).

## Experiment protocol
Run supervised warm-start ONCE; branch into arms that reuse it via `--skip-supervised --model-path <sup_ckpt>` (verified at `train_t5_policy.py:507,588`):
- **Arm A (control):** WITHOUT `--hazard-pretrain`.
- **Arm B (treatment):** WITH `--hazard-pretrain` (± `--beta-terminal`, `--freeze-answer-head`).
- **Arm C (compute-confound control, R-2):** a step-matched, signal-free hazard ablation (same optimizer-step count as B's hazard phase, but a null/scrambled objective) — separates "warm-start signal" from "more compute." (`supervised-longer` is a weaker fallback; B=1-per-question steps aren't commensurable with B=N — state the matching unit.)
Each arm × seed writes to a **DISTINCT `checkpoint_dir`** (R-6) — the default collides (`checkpoints/ppo_t5`, one dir from `train_t5_policy.py:200`), so arm 2 would overwrite arm 1's `config_used.json`/`split_manifest.json`.

## Testable rules (invariants + Enforcement)
- **R-1 controlled arms.** Identical `model_name`, data split, training seed, PPO budget across arms; only Phase 1.5 differs; all arms branch from one supervised ckpt. *Enforce:* harness asserts equality by diffing each arm's `config_used.json` (except hazard keys). Requires E-1.
- **R-2 compute-confound control.** Include Arm C (above); report B's hazard step + wall-clock cost. *Enforce:* report includes the C-vs-A and B-vs-A deltas side by side.
- **R-3 metrics, identical path.** For each arm×seed on the SAME test split, call `evaluate_t5_policy` (one checkpoint per arm — NOT the MLP-vs-T5 diff): accuracy, S_q (`metrics.py:55 system_score`), `expected_calibration_error` (`:150`), `brier_score` (`:189`), buzz-position mean + histogram. *Enforce:* same function, same test set (resolved from the split manifest, `compare_policies.py:523`). **Expected Wins is EXCLUDED from the default** (`expected_wins_score`, `:87`, needs a mandatory `opponent_survival_trace` + `reward_mode="expected_wins"` via `evaluate_all.py:630-653`'s `build_opponent_model_from_config`) — offer it only as an explicit opt-in arm that wires that opponent path.
- **R-4 primary endpoint (falsifiable).** Buzz earlier at matched accuracy. *Enforce:* condition on correct-only (or report a position×accuracy frontier — raw mean buzz-position co-varies with accuracy). Commit a concrete threshold, e.g. **"treatment's mean correct-answer buzz position is ≥1 prefix earlier than control AND answer accuracy is within −1pt, replicated in ≥2 of 3 seeds."** S_q is the secondary composite.
- **R-5 significance (scale-gated).** Paired bootstrap CI on per-question S_q (paired by `qid`, via E-2 + `controls.py:390`) that excludes 0. *Enforce:* this is a **full-scale / Device-2 endpoint only** — smoke's `max_questions=50` is a GLOBAL cap → test set ≈8 Qs → CIs meaningless. Smoke = plumbing + training-dynamics only (loss decreases, stop-prob distribution shifts). Do NOT use n=3 mean±std as significance.
- **R-6 provenance fail-loud.** Distinct per-arm dirs; assert identical split/config-except-hazard; record model_name, seed, device, git SHA. *Enforce:* also persist a manifest at supervised-save (only PPO writes one today, `:631`) so you can prove supervised never saw test qs.
- **R-7 scale caveat in output.** Report states the model_name/scale + that full-scale t5-large is a Device-2 (RTX 5090) run; never an unqualified "hazard helps."
- **R-8 reuse.** Reuse `evaluate_t5_policy` + `evaluation/metrics.py` + `controls.py` bootstrap; harness = orchestration + E-1/E-2 enablers + report/plot (`evaluation/plotting.py`).

## Smoke config overrides (required — else eval can't run)
At `ppo.iterations=5`, `best_model/` is never written (`eval_interval=10`/`save_interval=20`; `train_ppo_t5.py:748,771,779`). For smoke, override `eval_interval=save_interval=1` OR evaluate the returned in-memory model instead of a checkpoint path.

## Feasibility (M3 Max, MPS)
Hazard phase is trivial (B=1, 1 epoch, ~seconds). PPO dominates. Smoke t5-small, all arms × 3 seeds ≈ tens of minutes (~2–5 min/arm). t5-base ≈ single-digit× slower (preliminary signal, feasible). Full t5-large × 100 iters on MPS = many hours / likely OOM → **Device-2 (RTX 5090) run**.

## Acceptance (`/cverify`)
Smoke: E-1/E-2 land with tests; all arms run to a report; R-6 provenance asserted; R-3 metrics computed identically; training dynamics (R-5 smoke portion) shown. Full verdict (R-4 threshold + R-5 CIs) is produced at t5-base (preliminary, this machine) and flagged for t5-large confirmation on Device 2.

## Output
`hazard_efficacy_report.json` (per arm×seed metrics + A/B/C deltas + verdict + scale caveat) + a WITH/WITHOUT/control buzz-position-vs-accuracy plot.
