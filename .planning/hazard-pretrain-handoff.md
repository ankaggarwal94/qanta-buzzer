# Session handoff — conduct the hazard pre-training (Correctless workflow)

> **Operator note (not part of the prompt):** start the new session as **Fable 5 at Max effort**, in the canonical clone `/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer`. Ensure the **Correctless plugin is installed locally first** (see "Correctless setup" below) — the repo ships `.correctless/` content but the `.claude/` hook/command wiring is gitignored, so `/c*` commands won't exist in a fresh clone until Correctless is installed. Copy everything below the line into the new session.

---

You are conducting the **efficacy validation of the hazard-pretraining warm-start bridge** for the T5 quiz-bowl buzz policy, structured under the **Correctless workflow**. You have no memory of the session that built the bridge; everything you need is below.

## Background — what already exists (your input)

The hazard-pretrain **bridge is implemented and open as PR #32** (`feat/hazard-pretrain-bridge` → `main`), built via Correctless (spec→review→TDD→verify→docs). It is **smoke-validated as plumbing only** — the flag runs, the loss is finite, the freeze works, the checkpoint is loadable, PPO consumes it — but its **training efficacy is NOT established**. That is your job.

- `training/hazard_pretrain.py::run_hazard_pretrain` — loads the supervised checkpoint, then for each training question forwards the T5 policy over the question's `cumulative_prefixes` to get per-prefix stop-probabilities (wait head) and answer-NLL (answer head vs `gold_index`), and minimizes `hazard_expected_nll_loss` to teach the buzz/stop head *when to buzz*. Saves `checkpoints/hazard/best_model` for PPO. MVP is **B=1 per question** (no padding/mask path).
- `scripts/train_t5_policy.py::main` runs three phases: **supervised warm-start → (Phase 1.5) hazard bridge → PPO**. `--hazard-pretrain` enables Phase 1.5; `--beta-terminal <float>` weights the never-buzz survival penalty; `--freeze-answer-head` freezes the answer head only (answer-NLL gradient still reaches the shared encoder — intended).
- Existing artifacts to read first: `.correctless/specs/hazard-pretrain-bridge.md` (8 rules R-001…R-008) and `.correctless/verification/hazard-pretrain-bridge-verification.md` (what plumbing was verified). Config: `configs/t5_policy.yaml` (`model.model_name` t5-large / device `auto` = cuda>mps>cpu; a `smoke:` block uses t5-small, 50 questions). `run_hazard_pretrain` reads `checkpoint_dir` (required), `device`, `hazard_lr` (falls back to `supervised_lr` then `5e-5`), `hazard_epochs` (default 1), `max_grad_norm`.

## Your goal (definition of done)

Move the bridge from "plumbing works" to "efficacy characterized." Concretely:

1. **Baseline vs treatment comparison, same scale + seeds (use 1, 2, 3 — but note: `--seed` does NOT exist yet; enabler E-1 in the spec must add it, holding the data split fixed):**
   - WITHOUT: `python scripts/train_t5_policy.py --config configs/t5_policy.yaml [--smoke] --seed <s>`
   - WITH: same command **+ `--hazard-pretrain`** (optionally `--beta-terminal`, `--freeze-answer-head`).
   - Record **S_q, Expected Wins, and calibration** (the project's metrics) + the **buzz-timing distribution** for each arm.
2. **Inspect the hazard phase itself:** confirm the Phase-1.5 loss decreases and the stop-probability distribution shifts (the head is learning *when* to buzz), not just that it runs.
3. **Small knob sweep:** `--beta-terminal` (e.g. 0.5 / 1.0 / 2.0) and `--freeze-answer-head` on/off — does the warm-start shift buzzing earlier *without* hurting answer accuracy?
4. **Verdict:** does the hazard warm-start improve buzz timing / S_q / EW at the scale you ran, or is it neutral/harmful? State it plainly with the numbers.

## Compute reality — READ THIS (scope your claims to it)

- **This session runs on Device 1** = MacBook Pro M3 Max (64 GB, macOS, **MPS, no CUDA**). It is genuinely feasible here to run **t5-small (`--smoke`) and t5-base** hazard pre-training + PPO on the reduced data for a **real preliminary efficacy signal** — set `model.model_name: t5-base` (or `--smoke` for t5-small) and `device: mps` (or `auto`). Start with `--smoke` to confirm dynamics fast, then scale to t5-base.
- **The definitive full-scale t5-large comparison needs Device 2** = RTX 5090 (24 GB), which hosts **Codex, not Claude Code**. Do **not** claim full-scale efficacy from an M3-Max t5-small/base run — deliver the preliminary read + explicitly flag "full-scale t5-large efficacy remains a Device-2 (RTX 5090) run" as the next handoff. (Canonical compute decision: monorepo `CLAUDE.md` §compute; `CS321M/.../D8-compute-target-rtx5090-cuda-fallback.md`.)

## Correctless setup + workflow (how to run this)

- **Setup:** confirm the Correctless plugin is installed (`/cstatus` should respond). If not, install it (marketplace `joshft/correctless`) — the repo's `.claude/` wiring is gitignored by design. Note (from the repo's AGENTS.md): Correctless gates are **advisory, best-effort tooling, not an integrity boundary**; hooks require **bash ≥ 4** (use Homebrew bash, not macOS `/bin/bash` 3.2); wrapper scripts use the repo-local **`.venv` directly** — `source .venv/bin/activate` (or invoke `.venv/bin/python …`; do NOT use homebrew python3.11 — it breaks the torch/transformers imports).
- **Workflow:** this is a new empirical-validation feature, so:
  - **Start from the ready spec** at `.planning/hazard-eval-harness-spec.md` — adopt it via `/cspec` (it already carries testable rules R-1…R-8 + acceptance, hardened against the real code by a two-critic pass). Mind its two **prerequisite code enablers** you must `/ctdd` FIRST, or the comparison is invalid: **E-1** add a training `--seed` to `scripts/train_t5_policy.py` (none exists today — training is unseeded, so a multi-seed comparison is meaningless without it), and **E-2** surface per-question S_q/buzz from `evaluate_t5_policy` (currently discarded) for the paired bootstrap + buzz histogram.
  - `/ctdd` E-1, E-2, and the thin harness — TDD each; don't hand-edit the vendored `.correctless/` framework files (regenerated on update). Note the spec **excludes Expected Wins from the default** (it needs an opponent model + `reward_mode="expected_wins"`) and requires **smoke config overrides** (`eval_interval=save_interval=1`) or no `best_model/` is written at 5 PPO iters.
  - `/cverify` the results against the spec's criteria; `/cdocs` to write up the efficacy report.
- The efficacy **verdict itself** is a judgment call — consider an actor-critic pass ("the bridge helps" vs "the delta is noise/overfit at this scale") before committing the conclusion.

## Guardrails

- Seeds 1, 2, 3 for the multi-seed runs (repo convention). MPS/CPU device on this machine.
- The bridge is MVP **B=1 per question** — expect it to be slow; `--smoke` first.
- Do not modify vendored `.correctless/hooks|scripts/*.sh` (manifest drift + clobber-on-update). Feature/eval code goes in `scripts/`, `training/`, `evaluation/`, `tests/`.
- PR #32 is the bridge under test. If it has merged to `main` by the time you start, work from `main`; otherwise check out `feat/hazard-pretrain-bridge`. If you improve the bridge itself, PR against `main` separately from the efficacy write-up.

## How your output will be used

Your efficacy report + verdict is the input to the decision on whether to (a) keep/tune the hazard bridge, (b) escalate the full-scale t5-large run to Device 2 (RTX 5090 / Codex), or (c) drop it. Make the verdict and the scale-caveat explicit enough that that decision can be made from your report alone.
