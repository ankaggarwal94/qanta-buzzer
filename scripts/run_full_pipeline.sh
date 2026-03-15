#!/usr/bin/env bash
# Full pipeline with parallelism — runs the core pipeline plus key extensions.
# Phases 9/10/12/18/19 require manual execution (see docs/full-pipeline-runbook.md).
#
# Dependencies form a DAG:
#
#   Phase 1 (build MC dataset)
#     ├── Wave A: Phases 2, 3, 5, 9, 10, 13  (independent, parallel)
#     │     ├── Phase 4  (needs Phase 2+3 outputs)
#     │     ├── Phase 6  (needs Phase 3+5 outputs)
#     │     └── Phase 7  (needs Phase 1 only, but writes to same PPO dir)
#     └── Wave B: Phases 11, 14, 15, 16, 17  (need Phase 1 MC dataset)
#
# Each parallel phase writes to its own artifacts directory to avoid clobber.
#
# Usage:
#   bash scripts/run_full_pipeline.sh                    # t5-base (balanced)
#   bash scripts/run_full_pipeline.sh --t5-model t5-small # fastest
#   bash scripts/run_full_pipeline.sh --t5-model t5-large # full quality
#   bash scripts/run_full_pipeline.sh --sequential        # no parallelism
#
# Requirements:
#   - Python venv activated with `pip install -e .`
#   - questions.csv at repo root
#   - ~10 GB free disk space
#
# Estimated wall time (Apple M3 Max, 64 GB):
#   t5-small, parallel: ~2–3 hours
#   t5-base, parallel:  ~3–5 hours
#   t5-large, parallel: ~6–10 hours

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
T5_MODEL="t5-base"
SEQUENTIAL=false
while [ $# -gt 0 ]; do
    case "$1" in
        --t5-model) T5_MODEL="$2"; shift 2 ;;
        --t5-model=*) T5_MODEL="${1#*=}"; shift ;;
        --sequential) SEQUENTIAL=true; shift ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "FULL PIPELINE — T5 model: $T5_MODEL, parallel: $([ "$SEQUENTIAL" = true ] && echo no || echo yes)"
echo "============================================================"
echo ""

RESULTS="$REPO_ROOT/results"
mkdir -p "$RESULTS"

# Activate venv if available
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Helper: run a command, log to file, print status on completion
run_phase() {
    local PHASE="$1"
    local LOG="$RESULTS/phase_${PHASE}.log"
    shift
    echo "[Phase $PHASE] STARTED at $(date +%H:%M:%S)"
    if "$@" > "$LOG" 2>&1; then
        echo "[Phase $PHASE] DONE at $(date +%H:%M:%S) — see $LOG"
    else
        echo "[Phase $PHASE] FAILED at $(date +%H:%M:%S) — see $LOG"
        return 1
    fi
}

# Helper: wait for background jobs, exit on first failure
wait_all() {
    local PIDS=("$@")
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "ERROR: Background job $pid failed"
            kill "${PIDS[@]}" 2>/dev/null || true
            exit 1
        fi
    done
}

########################################################################
# PHASE 1: Build MC dataset (sequential — everything depends on this)
########################################################################
echo "=== PHASE 1: Build MC dataset ==="
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/main
echo "[Phase 1] DONE — $(python -c "import json; print(f'{len(json.load(open(\"artifacts/main/mc_dataset.json\")))} MC questions')")"
echo ""

MC="artifacts/main/mc_dataset.json"

if [ "$SEQUENTIAL" = true ]; then
    ####################################################################
    # SEQUENTIAL MODE
    ####################################################################
    echo "=== Running all phases sequentially ==="

    echo "=== PHASE 2: Baselines (TF-IDF) ==="
    python scripts/run_baselines.py --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "$RESULTS/baselines_tfidf.json"

    echo "=== PHASE 3: PPO (100k steps) ==="
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" --seed 13 --deterministic-eval
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_default.json"
    cp artifacts/main/ppo_model.zip "$RESULTS/ppo_model_default.zip"

    echo "=== PHASE 4: Evaluate all ==="
    python scripts/evaluate_all.py --config configs/default.yaml --mc-path "$MC"
    cp artifacts/main/evaluation_report.json "$RESULTS/eval_default.json"

    echo "=== PHASE 5: T5 policy ==="
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml model.model_name="$T5_MODEL"

    echo "=== PHASE 6: Compare policies ==="
    python scripts/compare_policies.py \
        --mlp-checkpoint artifacts/main/ppo_model \
        --t5-checkpoint checkpoints/ppo_t5/best_model \
        --mc-path "$MC" \
        --output "$RESULTS/t5_comparison.json"

    echo "=== PHASE 9: Distractor comparison ==="
    for STRAT in tfidf_profile category_random; do
        python scripts/build_mc_dataset.py --config configs/default.yaml \
            --output-dir "artifacts/distractor_$STRAT" data.distractor_strategy="$STRAT"
        python scripts/run_baselines.py --config configs/default.yaml \
            --mc-path "artifacts/distractor_$STRAT/mc_dataset.json" likelihood.model=tfidf
        cp artifacts/main/baseline_summary.json "$RESULTS/baselines_$STRAT.json"
    done

    echo "=== PHASE 13: K-sensitivity ==="
    for K in 2 3 5 6; do
        python scripts/build_mc_dataset.py --config configs/default.yaml \
            --output-dir "artifacts/k$K" data.K="$K" data.distractor_strategy=category_random
        python scripts/run_baselines.py --config configs/default.yaml \
            --mc-path "artifacts/k$K/mc_dataset.json" likelihood.model=tfidf
        cp artifacts/main/baseline_summary.json "$RESULTS/baselines_k$K.json"
    done

    echo "=== PHASE 11: Expected Wins ==="
    python scripts/evaluate_all.py --config configs/default.yaml --mc-path "$MC" \
        environment.reward_mode=expected_wins environment.opponent_buzz_model.type=logistic
    cp artifacts/main/evaluation_report.json "$RESULTS/eval_ew_logistic.json"

    echo "=== PHASE 14: Reward modes ==="
    for MODE in simple human_grounded; do
        python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
            --seed 13 --deterministic-eval environment.reward_mode="$MODE"
        cp artifacts/main/ppo_summary.json "$RESULTS/ppo_$MODE.json"
    done

    echo "=== PHASE 15: Belief mode (sequential_bayes) ==="
    python scripts/run_baselines.py --config configs/default.yaml --mc-path "$MC" \
        environment.belief_mode=sequential_bayes likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "$RESULTS/baselines_seqbayes.json"

    echo "=== PHASE 16: Stop-only PPO ==="
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval --policy-mode stop_only
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_stop_only.json"

    echo "=== PHASE 17: No-buzz horizon ==="
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval environment.end_mode=no_buzz environment.no_buzz_reward=-0.25
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_no_buzz.json"

else
    ####################################################################
    # PARALLEL MODE
    ####################################################################
    echo "=== WAVE 1: Independent phases (parallel) ==="
    echo "Launching 4 parallel tracks..."
    echo ""

    PIDS=()

    # Track A: Baselines + eval (writes to artifacts/main/)
    (
        run_phase "2" python scripts/run_baselines.py \
            --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
        cp artifacts/main/baseline_summary.json "$RESULTS/baselines_tfidf.json"
    ) &
    PIDS+=($!)

    # Track B: PPO training (writes to artifacts/ppo_default/)
    (
        mkdir -p artifacts/ppo_default
        run_phase "3" python scripts/train_ppo.py \
            --config configs/default.yaml --mc-path "$MC" --seed 13 --deterministic-eval
        cp artifacts/main/ppo_summary.json "$RESULTS/ppo_default.json"
        cp artifacts/main/ppo_model.zip "$RESULTS/ppo_model_default.zip"
    ) &
    PIDS+=($!)

    # Track C: T5 policy (writes to checkpoints/, independent)
    (
        run_phase "5" python scripts/train_t5_policy.py \
            --config configs/t5_policy.yaml model.model_name="$T5_MODEL"
    ) &
    PIDS+=($!)

    # Track D: K-sensitivity builds (writes to artifacts/k*/)
    (
        for K in 2 3 5 6; do
            run_phase "13-k$K" python scripts/build_mc_dataset.py \
                --config configs/default.yaml \
                --output-dir "artifacts/k$K" data.K="$K" data.distractor_strategy=category_random
            run_phase "13-k${K}-baselines" python scripts/run_baselines.py \
                --config configs/default.yaml \
                --mc-path "artifacts/k$K/mc_dataset.json" likelihood.model=tfidf
            cp artifacts/main/baseline_summary.json "$RESULTS/baselines_k$K.json"
        done
    ) &
    PIDS+=($!)

    echo "Waiting for Wave 1 (${#PIDS[@]} tracks)..."
    wait_all "${PIDS[@]}"
    echo ""

    echo "=== WAVE 2: Phases that depend on Wave 1 ==="
    PIDS=()

    # Phase 4: Evaluate all (needs baselines from Phase 2)
    (
        run_phase "4" python scripts/evaluate_all.py \
            --config configs/default.yaml --mc-path "$MC"
        cp artifacts/main/evaluation_report.json "$RESULTS/eval_default.json"
    ) &
    PIDS+=($!)

    # Phase 6: Compare policies (needs Phase 3 PPO + Phase 5 T5)
    (
        run_phase "6" python scripts/compare_policies.py \
            --mlp-checkpoint artifacts/main/ppo_model \
            --t5-checkpoint checkpoints/ppo_t5/best_model \
            --mc-path "$MC" \
            --output "$RESULTS/t5_comparison.json"
    ) &
    PIDS+=($!)

    # Phase 11: Expected Wins eval
    (
        run_phase "11" python scripts/evaluate_all.py \
            --config configs/default.yaml --mc-path "$MC" \
            environment.reward_mode=expected_wins environment.opponent_buzz_model.type=logistic
        cp artifacts/main/evaluation_report.json "$RESULTS/eval_ew_logistic.json"
    ) &
    PIDS+=($!)

    # Phase 15: Belief mode comparison
    (
        run_phase "15" python scripts/run_baselines.py \
            --config configs/default.yaml --mc-path "$MC" \
            environment.belief_mode=sequential_bayes likelihood.model=tfidf
        cp artifacts/main/baseline_summary.json "$RESULTS/baselines_seqbayes.json"
    ) &
    PIDS+=($!)

    echo "Waiting for Wave 2 (${#PIDS[@]} tracks)..."
    wait_all "${PIDS[@]}"
    echo ""

    echo "=== WAVE 3: PPO ablations (sequential — share artifacts/main/) ==="

    echo "[Phase 14a] reward_mode=simple"
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval environment.reward_mode=simple
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_simple.json"

    echo "[Phase 14b] reward_mode=human_grounded"
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval environment.reward_mode=human_grounded
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_human_grounded.json"

    echo "[Phase 16] policy_mode=stop_only"
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval --policy-mode stop_only
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_stop_only.json"

    echo "[Phase 17] end_mode=no_buzz"
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval environment.end_mode=no_buzz environment.no_buzz_reward=-0.25
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_no_buzz.json"

fi

########################################################################
# FINAL SUMMARY
########################################################################
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results directory:"
ls -1 "$RESULTS"/*.json 2>/dev/null | while read f; do echo "  $(basename $f)"; done
echo ""
echo "Artifacts:"
for d in artifacts/main artifacts/k* artifacts/distractor_*; do
    [ -d "$d" ] && echo "  $d/ — $(ls $d/*.json 2>/dev/null | wc -l) JSON files"
done
echo ""
echo "Checkpoints:"
ls -d checkpoints/*/best_model 2>/dev/null | while read d; do echo "  $d/"; done
echo ""
echo "To generate the final comparison table:"
echo "  python -c \""
echo "import json, glob"
echo "for f in sorted(glob.glob('results/*.json')):"
echo "    s = json.load(open(f))"
echo "    name = f.split('/')[-1].replace('.json', '')"
echo "    acc = s.get('buzz_accuracy', s.get('accuracy', 'N/A'))"
echo "    sq = s.get('mean_sq', 'N/A')"
echo "    print(f'{name}: acc={acc}, S_q={sq}')"
echo "\""
