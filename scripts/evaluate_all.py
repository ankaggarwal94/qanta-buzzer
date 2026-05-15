#!/usr/bin/env python3
"""
Comprehensive evaluation with control experiments and visualization.

Runs the SoftmaxProfileBuzzer at the best threshold (from baseline sweep),
then executes control experiments (choices-only, shuffle, alias substitution)
and generates comparison plots and tables for the CS234 writeup.

Consumes outputs from:
- build_mc_dataset.py (prefer test_dataset.json, fallback mc_dataset.json)
- run_baselines.py (baseline_summary.json on validation)
- train_ppo.py (ppo_summary.json on validation, ppo_model.zip for test replay)

Produces:
- evaluation_report.json (full eval + controls + baseline + PPO summaries)
- plots/entropy_vs_clue.png
- plots/calibration.png
- plots/comparison.csv

Usage:
    python scripts/evaluate_all.py --smoke
    python scripts/evaluate_all.py --config configs/custom.yaml
    python scripts/evaluate_all.py --mc-path artifacts/main/mc_dataset.json

Ported from qb-rl reference implementation (scripts/evaluate_all.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bayesian_buzzer import SoftmaxProfileBuzzer
from agents.ppo_buzzer import PPOBuzzer
from agents.threshold_buzzer import (
    _softmax_episode_from_precomputed,
    precompute_beliefs,
)
from evaluation.controls import (
    run_alias_substitution_control,
    run_choices_only_control,
    run_shuffle_control_precomputed,
)
from evaluation.metrics import (
    calibration_at_buzz,
    per_category_accuracy,
    summarize_buzz_metrics,
)
from evaluation.plotting import (
    plot_calibration_curve,
    plot_entropy_vs_clue_index,
    save_comparison_table,
)
from qb_data.config import merge_overrides
from qb_env.stop_only_env import StopOnlyEnv
from qb_env.tossup_env import make_env_from_config
from qb_env.tossup_env import precompute_beliefs as env_precompute_beliefs
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    collect_env_texts,
    load_config,
    load_checkpoint_sidecar,
    load_embedding_cache,
    load_json,
    load_mc_questions,
    parse_overrides,
    redirect_combined_to_split,
    resolve_default_dataset_path,
    save_json,
    split_name_from_path,
)


def _safe_load(path: Path, context: str) -> list:
    """Inline safe-loader so test monkey-patches of ``load_mc_questions``
    in ``scripts.evaluate_all`` keep applying. Returns ``[]`` on
    JSONDecodeError / OSError / KeyError / TypeError / ValueError with
    a loud warning.

    ``TypeError`` and ``ValueError`` cover JSON that parses but has the
    wrong shape (e.g. a top-level list of strings or numbers, where
    ``mc_question_from_dict`` then subscripts a non-dict and raises
    ``TypeError``, or per-field ``int(...)`` / ``list(...)`` coercions
    raise ``ValueError``). Without these the partial-pipeline-fallback
    contract is silently broken and the entire run aborts."""
    try:
        return load_mc_questions(path)
    except (
        json.JSONDecodeError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"WARNING: failed to load MC questions from {path} ({context}): "
            f"{type(exc).__name__}: {exc}; treating as empty and continuing."
        )
        return []


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with config, smoke, and mc_path fields.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate all agents and controls."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
    )
    parser.add_argument(
        "--mc-path", type=str, default=None,
        help="Optional MC dataset JSON path (overrides config-derived path).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: artifacts/<split>).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: key=value (e.g. likelihood.model=tfidf)",
    )
    return parser.parse_args()


def pick_best_softmax_threshold(
    out_dir: Path, default_threshold: float
) -> float:
    """Select the best softmax threshold from baseline sweep results.

    Loads baseline_summary.json and extracts the threshold with the
    highest mean S_q score from the softmax_profile results.

    Parameters
    ----------
    out_dir : Path
        Directory containing baseline_summary.json.
    default_threshold : float
        Fallback threshold if baseline summary is unavailable.

    Returns
    -------
    float
        Best threshold by S_q score, or default_threshold if unavailable.
    """
    summary_path = out_dir / "baseline_summary.json"
    if not summary_path.exists():
        return default_threshold
    summary = load_json(summary_path)
    softmax = summary.get("softmax_profile", {})
    if not softmax:
        return default_threshold
    best_t = default_threshold
    best_sq = float("-inf")
    for t_str, metrics in softmax.items():
        sq = float(metrics.get("mean_sq", float("-inf")))
        if sq > best_sq:
            best_sq = sq
            best_t = float(t_str)
    return best_t


def main() -> None:
    """Run comprehensive evaluation with controls and visualizations."""
    args = parse_args()

    config = load_config(args.config, smoke=args.smoke)
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    split = "smoke" if args.smoke else "main"
    out_dir = Path(args.output_dir) if args.output_dir else ARTIFACT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    if args.mc_path:
        requested_mc_path = Path(args.mc_path)
        mc_path, eval_split, mc_warning = redirect_combined_to_split(
            requested_mc_path, preferred_split="test",
        )
        if mc_warning:
            print(mc_warning)
    else:
        requested_mc_path = None
        mc_path, eval_split, warning = resolve_default_dataset_path(
            out_dir,
            preferred_split="test",
        )
        if warning:
            print(warning)

    print(f"Loading MC questions from: {mc_path}")
    mc_questions = load_mc_questions(mc_path)
    if not mc_questions and not args.mc_path:
        fallback = mc_path.parent / "mc_dataset.json"
        if fallback.exists() and fallback != mc_path:
            print(
                f"Warning: {mc_path} contained 0 questions; "
                f"falling back to {fallback}"
            )
            mc_path = fallback
            eval_split = "combined"
            mc_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(mc_questions)} MC questions")

    # Load optional alias lookup sidecar.
    alias_path = out_dir / "alias_lookup.json"
    if alias_path.exists():
        alias_lookup = load_json(alias_path)
    else:
        print(f"Warning: alias_lookup.json not found at {alias_path}, using empty lookup")
        alias_lookup = {}

    # Build TF-IDF from train split even when evaluating on held-out data.
    # Initialize likelihood_questions explicitly so the fallback branch
    # never depends on Python's short-circuit ``or`` evaluation order.
    train_path = mc_path.parent / "train_dataset.json"
    likelihood_questions: list = []
    if train_path.exists():
        likelihood_questions = _safe_load(
            train_path, context="evaluate_all softmax likelihood"
        )
    if not likelihood_questions:
        likelihood_questions = mc_questions
        print(
            "Building likelihood model: "
            f"{config['likelihood']['model']} "
            "(train split missing or empty; using eval split)"
        )
    else:
        print(
            "Building likelihood model: "
            f"{config['likelihood']['model']} "
            f"(fit on train split with {len(likelihood_questions)} questions)"
        )
    likelihood_model = build_likelihood_model(config, likelihood_questions)
    load_embedding_cache(likelihood_model, config)
    beta = float(config["likelihood"].get("beta", 5.0))
    alpha = float(config["bayesian"].get("alpha", 10.0))
    default_threshold = float(config["bayesian"]["threshold_sweep"][0])
    threshold = pick_best_softmax_threshold(out_dir, default_threshold=default_threshold)
    print(f"Using best softmax threshold: {threshold}")

    # Honor the documented ``environment.reward`` alias for ``reward_mode``.
    # ``make_env_from_config`` reads both keys; offline reward replay must
    # match or baselines/eval will diverge from the env on CLI overrides.
    env_cfg = config.get("environment", {})
    reward_mode = str(env_cfg.get("reward", env_cfg.get("reward_mode", "time_penalty")))
    wait_penalty = float(env_cfg.get("wait_penalty", 0.0))
    buzz_correct = float(env_cfg.get("buzz_correct", 1.0))
    buzz_incorrect = float(env_cfg.get("buzz_incorrect", -0.5))
    early_buzz_penalty = float(env_cfg.get("early_buzz_penalty", 0.0))

    # Batch-encode every text the env will ever score before computing
    # beliefs (mirrors run_baselines.py / train_ppo.py). For SBERT / T5
    # likelihoods this turns ~N×~10 single-shot encoder calls into
    # batches of 64.
    _all_eval_texts = collect_env_texts(mc_questions)
    print(
        f"Pre-computing embeddings for {len(set(_all_eval_texts)):,} unique texts..."
    )
    likelihood_model.precompute_embeddings(_all_eval_texts, batch_size=64)

    # Precompute beliefs once (single pass of likelihood_model.score())
    print("Precomputing beliefs...")
    precomputed = precompute_beliefs(mc_questions, likelihood_model, beta)

    # Precomputed evaluation (zero extra score() calls)
    def evaluate_questions_precomputed(pqs):
        runs = [asdict(_softmax_episode_from_precomputed(
            pq, threshold, alpha,
            reward_mode=reward_mode,
            wait_penalty=wait_penalty,
            buzz_correct=buzz_correct,
            buzz_incorrect=buzz_incorrect,
            early_buzz_penalty=early_buzz_penalty,
        )) for pq in pqs]
        summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
        summary["runs"] = runs
        return summary

    # Live evaluator for controls that genuinely change option text (alias)
    def evaluate_questions_live(qset):
        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood_model,
            threshold=threshold,
            beta=beta,
            alpha=alpha,
            reward_mode=reward_mode,
            wait_penalty=wait_penalty,
            buzz_correct=buzz_correct,
            buzz_incorrect=buzz_incorrect,
            early_buzz_penalty=early_buzz_penalty,
        )
        runs = [asdict(agent.run_episode(q)) for q in qset]
        summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
        summary["runs"] = runs
        return summary

    # --- Run evaluations ---
    print("Running full evaluation...")
    full_eval = evaluate_questions_precomputed(precomputed)

    # Compute per-category breakdown
    print("\nComputing per-category breakdown...")
    per_category_results = per_category_accuracy(full_eval["runs"], mc_questions)

    # Sort by category name for readability
    per_category_sorted = dict(sorted(per_category_results.items()))

    print("\nPer-category accuracy:")
    for category, metrics in per_category_sorted.items():
        print(
            f"  {category:20s} (n={metrics['n']:3.0f}): "
            f"acc={metrics['buzz_accuracy']:.3f}, "
            f"S_q={metrics['mean_sq']:.3f}"
        )
    print()

    print("Running shuffle control...")
    shuffle_eval = run_shuffle_control_precomputed(
        precomputed, threshold, alpha,
        reward_mode=reward_mode,
        wait_penalty=wait_penalty,
        buzz_correct=buzz_correct,
        buzz_incorrect=buzz_incorrect,
        early_buzz_penalty=early_buzz_penalty,
    )

    if alias_lookup:
        print("Running alias substitution control...")
        alias_eval = run_alias_substitution_control(
            mc_questions,
            alias_lookup=alias_lookup,
            evaluator=evaluate_questions_live,
        )
        alias_control_report = {k: v for k, v in alias_eval.items() if k != "runs"}
    else:
        print(
            "Skipping alias substitution control: alias_lookup.json missing or empty"
        )
        alias_control_report = {
            "skipped": True,
            "reason": "alias_lookup.json missing or empty",
        }

    print("Running choices-only control...")
    choices_only = run_choices_only_control(mc_questions)

    # --- Load existing artifacts ---
    ppo_summary_path = out_dir / "ppo_summary.json"
    ppo_validation_summary = (
        load_json(ppo_summary_path) if ppo_summary_path.exists() else {}
    )
    baseline_summary_path = out_dir / "baseline_summary.json"
    baseline_summary = (
        load_json(baseline_summary_path) if baseline_summary_path.exists() else {}
    )
    ppo_checkpoint_path = out_dir / "ppo_model.zip"
    ppo_test_summary: dict[str, object] = {}
    # Distinguish two missing-checkpoint scenarios:
    #
    #   (a) Training failed at agent.save(). The trainer writes metadata
    #       BEFORE the model and re-stamps ``training_completed=True``
    #       only after a successful save. So a missing model paired with
    #       ``training_completed=False`` (or with ``run_metadata.json``
    #       absent on legacy artifacts where we cannot tell) means the
    #       co-written ``ppo_summary.json`` is from the failed run and
    #       must not seep into the report.
    #
    #   (b) Successful training where the checkpoint was later pruned
    #       for storage cleanup. ``run_metadata.json`` reports
    #       ``training_completed=True`` and ``ppo_summary.json`` holds
    #       valid validation metrics from that successful run. Replay
    #       is impossible (no model on disk) but the validation summary
    #       must be preserved -- discarding it would lose real metrics.
    #
    # Branch (a) discards the validation summary; branch (b) keeps it.
    if not ppo_checkpoint_path.exists() and (
        (out_dir / "run_metadata.json").exists()
        or (out_dir / "config_used.json").exists()
    ):
        rm_path = out_dir / "run_metadata.json"
        prior_training_completed = False
        if rm_path.exists():
            try:
                rm_data = load_json(rm_path)
                prior_training_completed = (
                    isinstance(rm_data, dict)
                    and rm_data.get("training_completed") is True
                )
            except (json.JSONDecodeError, OSError):
                prior_training_completed = False
        if prior_training_completed:
            print(
                f"NOTE: ppo_model.zip is missing under {out_dir} but "
                "run_metadata.json reports training_completed=True; "
                "treating this as a post-training storage cleanup "
                "(checkpoint pruned). Skipping PPO replay (no model on "
                "disk) but preserving the existing ppo_summary.json "
                "validation metrics in the report."
            )
        else:
            print(
                f"WARNING: ppo_model.zip is missing under {out_dir} but "
                "training sidecars (run_metadata.json / config_used.json) "
                "exist with no training_completed=True marker; the "
                "previous training run likely failed at agent.save(). "
                "Discarding any leftover ppo_summary.json so stale "
                "validation metrics from the prior run are not "
                "republished as current results. Re-run "
                "scripts/train_ppo.py before evaluating."
            )
            ppo_validation_summary = {}
    if ppo_checkpoint_path.exists():
        ppo_eval_config, _, config_error = load_checkpoint_sidecar(
            ppo_checkpoint_path,
            "config_used.json",
        )
        if config_error:
            print(
                "Warning: config_used.json next to PPO checkpoint could not be "
                f"read ({config_error}); using current evaluation config."
            )
            ppo_eval_config = None
        if not isinstance(ppo_eval_config, dict):
            ppo_eval_config = config

        run_metadata, run_metadata_path, run_metadata_error = load_checkpoint_sidecar(
            ppo_checkpoint_path,
            "run_metadata.json",
        )
        policy_mode = "flat_kplus1"
        training_completed = True  # Default for legacy artifacts predating the flag.
        if run_metadata_error:
            print(
                "Warning: run_metadata.json next to PPO checkpoint could not be "
                f"read ({run_metadata_error}); assuming flat_kplus1 policy."
            )
        elif run_metadata_path is None:
            print(
                "Warning: run_metadata.json not found next to PPO checkpoint; "
                "legacy checkpoints are assumed to use flat_kplus1."
            )
        elif isinstance(run_metadata, dict):
            policy_mode = str(run_metadata.get("policy_mode", "flat_kplus1"))
            # When the field is absent we trust the checkpoint (legacy);
            # when present the trainer guarantees True only after a
            # successful agent.save(). False = stale model.
            training_completed = bool(run_metadata.get("training_completed", True))
        if policy_mode not in {"flat_kplus1", "stop_only"}:
            raise ValueError(
                f"Unsupported PPO policy_mode '{policy_mode}' in run_metadata.json"
            )
        if not training_completed:
            print(
                "WARNING: run_metadata.json reports training_completed=False; "
                "the on-disk ppo_model.zip is from a prior run whose current "
                "training attempt failed at agent.save(). Refusing to replay "
                "stale checkpoint under fresh metadata, and discarding the "
                "stale ppo_summary.json that the same prior run produced so "
                "ppo_summary_source falls through to 'missing' rather than "
                "publishing stale validation metrics as current results. "
                "Re-run train_ppo.py."
            )
            # ppo_summary.json is co-written by train_ppo.py, so the same
            # prior-run-residue argument that motivates the model-replay
            # refusal applies to the validation summary. Clear it here so
            # the report's ppo_summary fallback chain (test -> validation
            # -> missing) skips straight to "missing" instead of silently
            # publishing the prior run's metrics under the current run's
            # provenance.
            ppo_validation_summary = {}

    if ppo_checkpoint_path.exists() and locals().get("training_completed", True):
        print("Replaying PPO checkpoint on evaluation split...")
        # Honor the likelihood corpus provenance recorded during training.
        ppo_ref_questions = None
        recorded_ref_path = None
        if isinstance(run_metadata, dict):
            recorded_ref_path = run_metadata.get("likelihood_reference_path")
        if recorded_ref_path:
            ref_path = Path(recorded_ref_path)
            if not ref_path.is_absolute():
                ref_path = ppo_checkpoint_path.parent / ref_path
            if ref_path.exists():
                ppo_ref_questions = _safe_load(
                    ref_path, context="evaluate_all PPO replay (recorded ref)"
                )
                if ppo_ref_questions:
                    ref_split = run_metadata.get("likelihood_reference_split", "recorded")
                    print(
                        f"  PPO likelihood built from recorded {ref_split} "
                        f"corpus ({len(ppo_ref_questions)} questions)"
                    )
        if not ppo_ref_questions:
            train_path = mc_path.parent / "train_dataset.json"
            if train_path.exists():
                ppo_ref_questions = _safe_load(
                    train_path, context="evaluate_all PPO replay (sibling train)"
                )
            if ppo_ref_questions:
                print(f"  PPO likelihood built from sibling train split ({len(ppo_ref_questions)} questions)")
            else:
                ppo_ref_questions = mc_questions
                print("  Warning: train split missing or empty; building PPO likelihood from eval split")
        ppo_likelihood_model = build_likelihood_model(ppo_eval_config, ppo_ref_questions)
        load_embedding_cache(ppo_likelihood_model, ppo_eval_config)
        # Mirror the train_ppo.py / softmax-eval pattern: pre-warm the
        # ppo_likelihood_model embedding cache with one batched encoder
        # pass over every text the replay env will score, instead of
        # letting precompute_beliefs() issue thousands of single-text
        # / per-K-options encoder calls. Test-split texts are not in the
        # train-side embedding cache loaded above, so SBERT/T5 hot loops
        # benefit ~10-60×; TF-IDF cost is unchanged.
        _ppo_replay_texts = collect_env_texts(mc_questions)
        print(
            f"Pre-computing PPO replay embeddings for "
            f"{len(set(_ppo_replay_texts)):,} unique texts..."
        )
        ppo_likelihood_model.precompute_embeddings(_ppo_replay_texts, batch_size=64)
        ppo_belief_cache = env_precompute_beliefs(
            mc_questions,
            ppo_likelihood_model,
            belief_mode=str(ppo_eval_config["environment"].get("belief_mode", "from_scratch")),
            beta=float(ppo_eval_config["likelihood"].get("beta", 5.0)),
        )
        ppo_env = make_env_from_config(
            mc_questions=mc_questions,
            likelihood_model=ppo_likelihood_model,
            config=ppo_eval_config,
            precomputed_beliefs=ppo_belief_cache,
        )
        if policy_mode == "stop_only":
            ppo_env = StopOnlyEnv(ppo_env)
        use_maskable = bool(
            ppo_eval_config.get("ppo", {}).get("use_maskable_ppo", False)
        )
        ppo_agent = PPOBuzzer.load(
            ppo_checkpoint_path,
            env=ppo_env,
            use_maskable_ppo=use_maskable,
        )
        ppo_runs = [
            asdict(ppo_agent.run_episode(deterministic=True, question_idx=i))
            for i in range(len(mc_questions))
        ]
        ppo_test_summary = {
            **summarize_buzz_metrics(ppo_runs),
            **calibration_at_buzz(ppo_runs),
            "eval_split": eval_split,
        }

    # --- Build evaluation report ---
    # Disambiguate which summary the legacy ``ppo_summary`` key carries so
    # downstream consumers (poster/presentation generators) can branch
    # safely instead of inferring split semantics from ``eval_split``.
    if ppo_test_summary:
        ppo_summary_for_report = ppo_test_summary
        ppo_summary_source = "test"
    elif ppo_validation_summary:
        ppo_summary_for_report = ppo_validation_summary
        ppo_summary_source = "validation"
    else:
        ppo_summary_for_report = {}
        ppo_summary_source = "missing"
    report = {
        # ``schema_version`` 2 indicates policy-buzz-only buzz_accuracy and
        # mean_buzz_step semantics in ``summarize_buzz_metrics``; consumers
        # that need pre-2026-05 semantics can read ``overall_outcome_accuracy``
        # or the raw ``runs`` traces.
        "schema_version": 2,
        "softmax_profile_best_threshold": threshold,
        "full_eval": {k: v for k, v in full_eval.items() if k != "runs"},
        "controls": {
            "choices_only": choices_only,
            "shuffle": {k: v for k, v in shuffle_eval.items() if k != "runs"},
            "alias_substitution": alias_control_report,
        },
        "per_category": per_category_sorted,
        "split_contract": {
            "baseline_selection_split": baseline_summary.get(
                "dataset_split", "legacy/unknown"
            ),
            "softmax_eval_split": eval_split,
            "ppo_train_split": ppo_validation_summary.get(
                "train_split", "legacy/unknown"
            ),
            "ppo_validation_split": ppo_validation_summary.get(
                "eval_split", "legacy/unknown"
            ),
            "ppo_test_split": ppo_test_summary.get("eval_split"),
            "requested_mc_path": (
                str(requested_mc_path) if requested_mc_path is not None else None
            ),
            "resolved_mc_path": str(mc_path),
        },
        "baseline_summary": baseline_summary,
        "ppo_validation_summary": ppo_validation_summary,
        "ppo_test_summary": ppo_test_summary,
        "ppo_summary": ppo_summary_for_report,
        "ppo_summary_source": ppo_summary_source,
    }

    # Add Expected Wins summary only when that reward mode is active
    if reward_mode == "expected_wins":
        from evaluation.metrics import expected_wins_score
        from qb_env.opponent_models import build_opponent_model_from_config

        opp_model = build_opponent_model_from_config(mc_questions, config)
        qid_to_q = {q.qid: q for q in mc_questions}
        if opp_model is not None:
            ew_scores = []
            for run in full_eval["runs"]:
                q = qid_to_q.get(run.get("qid", ""), mc_questions[0])
                opp_surv = [
                    opp_model.prob_survive_to_step(q, t)
                    for t in range(len(run.get("c_trace", [])))
                ]
                ew = expected_wins_score(
                    run.get("c_trace", []),
                    run.get("g_trace", []),
                    opp_surv,
                )
                ew_scores.append(ew)
            report["expected_wins"] = {
                "mean_ew": float(np.mean(ew_scores)) if ew_scores else 0.0,
                "n": len(ew_scores),
            }

    save_json(out_dir / "evaluation_report.json", report)

    # --- Generate visualizations ---
    print("Generating plots...")

    # Entropy vs clue index
    entropy_traces = [
        list(r["entropy_trace"])
        for r in full_eval["runs"]
        if r.get("entropy_trace")
    ]
    max_len = max((len(t) for t in entropy_traces), default=0)
    padded = np.full((len(entropy_traces), max_len), np.nan, dtype=np.float32)
    for i, trace in enumerate(entropy_traces):
        padded[i, : len(trace)] = np.array(trace, dtype=np.float32)
    entropy_trace = (
        np.nanmean(padded, axis=0).tolist() if max_len > 0 else []
    )
    plot_entropy_vs_clue_index(
        {"softmax_profile": entropy_trace},
        out_dir / "plots" / "entropy_vs_clue.png",
    )

    # Calibration curve — use canonical helper for consistency
    from evaluation.metrics import calibration_pairs_at_buzz
    confidences, outcomes = calibration_pairs_at_buzz(full_eval["runs"])
    plot_calibration_curve(
        confidences, outcomes, out_dir / "plots" / "calibration.png"
    )

    # Comparison table: include baseline sweep, controls, and PPO
    table_rows = []

    # Add baseline sweep results (threshold at multiple values)
    if "threshold" in baseline_summary:
        for threshold_str, metrics in baseline_summary["threshold"].items():
            table_rows.append({
                "agent": f"threshold_{threshold_str}",
                **{k: v for k, v in metrics.items() if k != "runs"},
            })

    # Add softmax_profile sweep results
    if "softmax_profile" in baseline_summary:
        for threshold_str, metrics in baseline_summary["softmax_profile"].items():
            table_rows.append({
                "agent": f"softmax_{threshold_str}",
                **{k: v for k, v in metrics.items() if k != "runs"},
            })

    # Add full softmax eval (best threshold) and control experiments
    table_rows.append({
        "agent": "full_softmax",
        **{k: v for k, v in full_eval.items() if k != "runs"},
    })
    table_rows.append({
        "agent": "shuffle_control",
        **{k: v for k, v in shuffle_eval.items() if k != "runs"},
    })
    if not alias_control_report.get("skipped"):
        table_rows.append({
            "agent": "alias_control",
            **{k: v for k, v in alias_control_report.items() if k != "runs"},
        })

    # Add PPO if available
    if ppo_summary_for_report:
        table_rows.append({"agent": "ppo", **ppo_summary_for_report})

    save_comparison_table(table_rows, out_dir / "plots" / "comparison.csv")

    print(f"Wrote evaluation report to: {out_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
