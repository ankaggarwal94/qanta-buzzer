# Hazard-Pretrain Bridge — Efficacy Report (Device-1 Preliminary)

**Date:** 2026-08-18 · **Branch:** `feature/hazard-efficacy-eval` (stacked on PR #32's `feat/hazard-pretrain-bridge`) · **Machine:** Device 1 (MacBook Pro M3 Max, 64 GB, MPS) · **Report artifacts:** `results/hazard_efficacy_smoke/hazard_efficacy_report.json`, `results/hazard_efficacy_base/hazard_efficacy_report.json` (+ plots, committed)

## TL;DR verdict

**The hazard warm-start bridge, as configured (default `--beta-terminal 1.0`), is actively harmful at every scale tested, and its effect carries zero temporal signal — it teaches the policy *whether* to buzz, not *when*.** The harm is statistically significant at the t5-base preliminary scale (paired bootstrap on per-question S_q deltas, n=110: **B−A = −0.241, 95% CI [−0.245, −0.237]**, excludes 0) and is mechanistic, not noise: the step-matched null-signal ablation (arm C, `shuffled_nll`) reproduces arm B **to four decimal places** at both scales.

**Recommendation: (c)-lean — do not escalate the current design to Device 2 as-is; do not merge the bridge into default training paths.** Keep the bridge code behind its opt-in flag (it is now a well-instrumented experimental surface), and gate any Device-2 investment on the cheap diagnostic below. See "Decision guidance".

## What was run

Protocol per `.correctless/specs/hazard-efficacy-eval.md` (R-001..R-015): one shared supervised checkpoint per scale; arms branch from it via `--skip-supervised`; identical config/split/PPO budget asserted across arms (only Phase 1.5 differs); training seeds 1/2/3 (`--seed`, E-1); identical `evaluate_t5_policy` path on the same held-out test split (E-2 per-question records); provenance, resume-identity, and significance gating enforced by `scripts/run_hazard_efficacy.py`.

| Scale | Model | Split (train/val/test) | PPO | Wall-clock (child total) | Tree |
|---|---|---|---|---|---|
| Smoke | t5-small | 28 / 3 / 13 | 5 iters | ~10 min (18 runs) | 5.1 GB |
| Preliminary | t5-base | 486 / 104 / 110 | 25 iters | ~109 min (12 runs) | 10.7 GB |

Arms: **A** control (no hazard) · **B** treatment (`--hazard-pretrain`, β=1.0) · **C** compute-confound control (`--hazard-ablation shuffled_nll` — same optimizer-step count, temporal signal destroyed) · knob variants (smoke: β∈{0.5, 2.0}, `--freeze-answer-head`; t5-base: β=2.0).

## Results

### t5-base preliminary (n_test = 110, seeds 1–3)

| Arm | Policy buzz rate | Policy-buzz accuracy | Mean S_q | Δ vs A (S_q) |
|---|---|---|---|---|
| A control | 0.00 (never buzzes) | — (0 buzzes) | 0.2409 | — |
| B hazard β=1.0 | 0.00 | — | **0.0000** | **−0.2409** (CI [−0.245, −0.237]) |
| C ablation β=1.0 | 0.00 | — | 0.0000 | −0.2409 (≡ B) |
| β=2.0 variant | 1.00 (always, position 0) | 0.191 / 0.236 / 0.309 (seeds 3/2/1) | 0.2343–0.2449 | −0.0008 (S_q); **+0.246 accuracy** |

- **Hazard-phase dynamics (B):** loss 1.033 → 1.000 — it converges to *exactly* the terminal survival penalty β=1.0, i.e. total never-buzz collapse (per-position P(BUZZ) → ~0.005; expected buzz time 1.92 → 5.09 of ~5). 25 PPO iterations never recover from this initialization (buzz rate stays 0.00 in all 3 seeds).
- **Primary endpoint (R-4: "buzz ≥1 prefix earlier at ≤1pt accuracy cost, ≥2/3 seeds"):** undefined in every seed — *neither* A nor B ever policy-buzzes at β=1.0. Correctly flagged `undefined_position`; report verdict gated to `endpoint_not_met_at_this_scale`.

### Smoke (n_test = 13, seeds 1–3) — plumbing + dynamics evidence only

Same qualitative picture, plus the knob sweep dose-response (monotone, replicated 3/3 seeds):

| β / knob | E[buzz t] after bridge (init ≈1.94) | Buzz rate after PPO | S_q |
|---|---|---|---|
| β=0.5 | 3.94 (latest) | 0.00 | 0.111 |
| β=1.0 (B) | 3.57–3.63 | 0.00 | 0.141 |
| β=1.0 + freeze | 3.76–3.83 | 0.00 | 0.122 |
| **β=2.0** | **1.30 (earliest)** | **1.00** | 0.198 |
| A control | — | 0.00 | 0.241 |

## Mechanism (why this happens)

The bridge minimizes `E[NLL at buzz] + β·P(never buzz)`. With an answer head whose per-prefix NLL sits *above* β at every position (chance level ln 4 ≈ 1.386 at smoke; ≈1.03 after the t5-base warm start — still > 1.0), the optimum is a corner: **never buzz** (loss = β exactly, which is precisely where B converges). Push β above the NLL scale and the optimum flips to the other corner: **buzz immediately**. Neither corner contains timing information, and with the answer head's NLL approximately *flat across positions* at these scales, there is **no temporal signal in the objective to learn** — which is exactly what B ≡ C demonstrates empirically.

The β=2.0 result is real but must be read carefully: +0.246 policy-buzz accuracy is measured against a control that never buzzes (so its policy-buzz accuracy is undefined/0 by construction), and the variant buzzes at position 0 always — S_q is unchanged (−0.0008). It converts a never-buzz policy into an immediate-buzz policy; it does not produce discriminative timing.

## Verdict detail (actor-critic considered)

- **"The bridge helps":** not supportable at any tested configuration. No arm×seed beat control on S_q; the endpoint was never met; the only positive delta (β=2.0 accuracy) is a corner artifact against a degenerate baseline.
- **"The delta is noise / overfit at this scale":** rejected for the harm claim — the paired CI excludes 0 by a wide margin at n=110, the collapse is analytically derivable from the objective (loss → β exactly), and B ≡ C at 4 decimals across 2 scales × 3 seeds is not a noise signature.
- **Strongest surviving caveat (shapes the recommendation):** both tested scales have weak answer heads with approximately flat per-position NLL, so the objective had no timing signal to encode *by construction*. A fully-trained t5-large supervised head might have position-declining NLL (later clues → easier answers), which is the regime the bridge was designed for. **The tested-scale result therefore cannot rule out efficacy in that regime — but it strongly constrains how to test it.**

## Decision guidance (a / b / c from the handoff)

1. **Do NOT (b)-escalate by simply re-running this experiment at t5-large on Device 2.** If the t5-large head's NLL profile is also flat, the same corner collapse will recur at ~50× the compute.
2. **Run the cheap Device-2 diagnostic first:** train (or reuse) the full t5-large supervised checkpoint, then measure **per-position answer-NLL on held-out questions** (a probe forward pass, minutes, no PPO). 
   - If NLL **declines materially with position**: the bridge design has signal to encode — re-run the harness at t5-large with **β calibrated between the head's late-position and early-position NLL** (or normalize NLL per question before the survival loss), and expect the endpoint to be meaningfully testable. The harness is ready for this run as-is (`--config` + `--variant` β sweep).
   - If NLL is **flat**: **(c) drop** the current objective and redesign before spending any full-scale compute (candidate directions: margin/correctness-probability targets instead of raw NLL; per-question NLL normalization; β annealing; or supervising the stop head directly on "first position where the answer becomes correct").
3. **(a) keep/tune on Device 1 is not productive** beyond what is already measured: the β sweep bracketed the behavior (0.5/1.0/2.0 → late/never/immediate corners) and seeds replicate tightly; more Device-1 runs will not change the picture.

## Scale caveats (mandatory)

- Full-scale t5-large efficacy remains a Device-2 (RTX 5090) run. *(Verbatim caveat carried in both report JSONs.)*
- The t5-base "preliminary" scale used a capped split (`data.max_questions=700` → 486/104/110) and 25 PPO iterations; absolute metric levels are not representative of full training, and all conclusions are scoped to the tested configurations.
- Smoke-scale numbers (n_test=13) are plumbing/training-dynamics evidence only; the report correctly refuses significance labels there.
- The shared-supervised design fixes the supervised-phase RNG per scale (paired design; isolates Phase-1.5 + PPO effects). Seeds vary hazard/PPO only.

## Reproduction

```bash
# Smoke grid + knob sweep (resumable; ~10 min child wall-clock, ~5 GB)
python scripts/run_hazard_efficacy.py --smoke --config configs/t5_policy.yaml \
  --out-dir results/hazard_efficacy_smoke --seeds 1 2 3 --arms A B C --prune-checkpoints \
  --variant "beta_lo:--beta-terminal=0.5" --variant "beta_hi:--beta-terminal=2.0" --variant "freeze:--freeze-answer-head"

# t5-base preliminary grid (~109 min child wall-clock, ~10.7 GB; needs artifacts/main populated)
python scripts/run_hazard_efficacy.py --config configs/t5_policy_base_prelim.yaml \
  --out-dir results/hazard_efficacy_base --seeds 1 2 3 --arms A B C --prune-checkpoints \
  --variant "beta_hi:--beta-terminal=2.0"
```

Full protocol and invariants: `.correctless/specs/hazard-efficacy-eval.md`; verification: `.correctless/verification/hazard-efficacy-eval-verification.md`; harness operations: `AGENTS.md` § "Hazard efficacy harness".
