#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/device2_stopdff_run.sh [options]

Options:
  --experiment dp_sweep
  --artifact-dir NAME
  --out-dir DIR
  --max-wall-hours HOURS
  --num-bootstrap N
  --n-jobs N
  --resume
  --overwrite
  --smoke
  --calibrators CSV
  --reward-schedules CSV
  --continuations CSV
  --data-dir DIR
  --calibration PATH
  --fit-split NAME
  --eval-split NAME
  --device-index N
  --min-free-gib GiB
EOF
}

die() {
  printf 'device2_stopdff_run.sh: %s\n' "$*" >&2
  exit 2
}

require_value() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    die "$flag requires a value"
  fi
}

json_escape() {
  local value="$1"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '"%s"' "$value"
}

json_array() {
  local first=1
  printf '['
  for item in "$@"; do
    if [[ "$first" -eq 0 ]]; then
      printf ', '
    fi
    first=0
    json_escape "$item"
  done
  printf ']'
}

to_shell_path() {
  local path="$1"
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$path"
  else
    printf '%s\n' "$path"
  fi
}

join_display_path() {
  local base="$1"
  local child="$2"
  base="${base%/}"
  if [[ "$base" == *\\* ]]; then
    printf '%s\\%s\n' "$base" "$child"
  else
    printf '%s/%s\n' "$base" "$child"
  fi
}

write_manifest() {
  local manifest_path="$1"
  {
    printf '{\n'
    printf '  "timestamp": '; json_escape "$MANIFEST_TIMESTAMP"; printf ',\n'
    printf '  "repo_root": '; json_escape "$REPO_ROOT"; printf ',\n'
    printf '  "run_dir": '; json_escape "$RUN_DIR_DISPLAY"; printf ',\n'
    printf '  "artifact_path": '; json_escape "$ARTIFACT_DISPLAY"; printf ',\n'
    printf '  "git_commit": '; json_escape "$GIT_COMMIT"; printf ',\n'
    printf '  "dirty_status": '; json_escape "$DIRTY_STATUS"; printf ',\n'
    printf '  "original_args": '; json_array "${ORIGINAL_ARGS[@]}"; printf ',\n'
    printf '  "parsed_axes": {\n'
    printf '    "calibrators": '; json_escape "$CALIBRATORS"; printf ',\n'
    printf '    "reward_schedules": '; json_escape "$REWARD_SCHEDULES"; printf ',\n'
    printf '    "continuations": '; json_escape "$CONTINUATIONS"; printf '\n'
    printf '  },\n'
    printf '  "CUDA_VISIBLE_DEVICES": "0",\n'
    printf '  "preflight_path": '; json_escape "$PREFLIGHT_DISPLAY"; printf ',\n'
    printf '  "preflight_command": '; json_array "CUDA_VISIBLE_DEVICES=0" "${PREFLIGHT_CMD[@]}"; printf ',\n'
    printf '  "sweep_command": '; json_array "CUDA_VISIBLE_DEVICES=0" "${SWEEP_CMD[@]}"; printf '\n'
    printf '}\n'
  } > "$manifest_path"
}

ORIGINAL_ARGS=("$@")

EXPERIMENT="dp_sweep"
ARTIFACT_DIR="paper_exports"
OUT_DIR=""
MAX_WALL_HOURS="6"
NUM_BOOTSTRAP="500"
N_JOBS="1"
RESUME=0
OVERWRITE=0
SMOKE=0
CALIBRATORS="uncalibrated,platt-logistic,temperature,isotonic"
REWARD_SCHEDULES="acf_flat,power_mark,wait_cost_small,strict_wrong,low_wrong_cost"
CONTINUATIONS="empirical_bucket,pooled_empirical,oracle_trajectory"
DATA_DIR="data/processed"
CALIBRATION="paper_exports/calibration.json"
FIT_SPLIT="val"
EVAL_SPLIT="test"
DEVICE_INDEX="0"
MIN_FREE_GIB="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      require_value "$1" "${2-}"
      EXPERIMENT="$2"
      shift 2
      ;;
    --experiment=*)
      EXPERIMENT="${1#*=}"
      shift
      ;;
    --artifact-dir)
      require_value "$1" "${2-}"
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --artifact-dir=*)
      ARTIFACT_DIR="${1#*=}"
      shift
      ;;
    --out-dir)
      require_value "$1" "${2-}"
      OUT_DIR="$2"
      shift 2
      ;;
    --out-dir=*)
      OUT_DIR="${1#*=}"
      shift
      ;;
    --max-wall-hours)
      require_value "$1" "${2-}"
      MAX_WALL_HOURS="$2"
      shift 2
      ;;
    --max-wall-hours=*)
      MAX_WALL_HOURS="${1#*=}"
      shift
      ;;
    --num-bootstrap)
      require_value "$1" "${2-}"
      NUM_BOOTSTRAP="$2"
      shift 2
      ;;
    --num-bootstrap=*)
      NUM_BOOTSTRAP="${1#*=}"
      shift
      ;;
    --n-jobs)
      require_value "$1" "${2-}"
      N_JOBS="$2"
      shift 2
      ;;
    --n-jobs=*)
      N_JOBS="${1#*=}"
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --calibrators)
      require_value "$1" "${2-}"
      CALIBRATORS="$2"
      shift 2
      ;;
    --calibrators=*)
      CALIBRATORS="${1#*=}"
      shift
      ;;
    --reward-schedules)
      require_value "$1" "${2-}"
      REWARD_SCHEDULES="$2"
      shift 2
      ;;
    --reward-schedules=*)
      REWARD_SCHEDULES="${1#*=}"
      shift
      ;;
    --continuations)
      require_value "$1" "${2-}"
      CONTINUATIONS="$2"
      shift 2
      ;;
    --continuations=*)
      CONTINUATIONS="${1#*=}"
      shift
      ;;
    --data-dir)
      require_value "$1" "${2-}"
      DATA_DIR="$2"
      shift 2
      ;;
    --data-dir=*)
      DATA_DIR="${1#*=}"
      shift
      ;;
    --calibration)
      require_value "$1" "${2-}"
      CALIBRATION="$2"
      shift 2
      ;;
    --calibration=*)
      CALIBRATION="${1#*=}"
      shift
      ;;
    --fit-split)
      require_value "$1" "${2-}"
      FIT_SPLIT="$2"
      shift 2
      ;;
    --fit-split=*)
      FIT_SPLIT="${1#*=}"
      shift
      ;;
    --eval-split)
      require_value "$1" "${2-}"
      EVAL_SPLIT="$2"
      shift 2
      ;;
    --eval-split=*)
      EVAL_SPLIT="${1#*=}"
      shift
      ;;
    --device-index)
      require_value "$1" "${2-}"
      DEVICE_INDEX="$2"
      shift 2
      ;;
    --device-index=*)
      DEVICE_INDEX="${1#*=}"
      shift
      ;;
    --min-free-gib)
      require_value "$1" "${2-}"
      MIN_FREE_GIB="$2"
      shift 2
      ;;
    --min-free-gib=*)
      MIN_FREE_GIB="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "$EXPERIMENT" != "dp_sweep" ]]; then
  die "unsupported experiment '$EXPERIMENT'; only 'dp_sweep' is supported"
fi

if [[ -z "$ARTIFACT_DIR" || "$ARTIFACT_DIR" == "." || "$ARTIFACT_DIR" == ".." || ! "$ARTIFACT_DIR" =~ ^[A-Za-z0-9._-]+$ ]]; then
  die "--artifact-dir must be a single safe directory name matching [A-Za-z0-9._-]+"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
else
  die "repo-local Python not found; expected .venv/bin/python or .venv/Scripts/python.exe"
fi

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
GIT_SHORT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || printf 'nogit')"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null || true)"
if [[ "$GIT_COMMIT" == "unknown" ]]; then
  DIRTY_STATUS="unknown"
elif [[ -n "$GIT_STATUS" ]]; then
  DIRTY_STATUS="dirty"
else
  DIRTY_STATUS="clean"
fi

if [[ -z "$OUT_DIR" ]]; then
  RUN_STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
  RUN_DIR_DISPLAY="artifacts/device2_stopdff/${RUN_STAMP}_${GIT_SHORT}"
  RUN_DIR_PATH="$RUN_DIR_DISPLAY"
else
  RUN_DIR_DISPLAY="$OUT_DIR"
  RUN_DIR_PATH="$(to_shell_path "$OUT_DIR")"
fi

ARTIFACT_PATH="$RUN_DIR_PATH/$ARTIFACT_DIR"
ARTIFACT_DISPLAY="$(join_display_path "$RUN_DIR_DISPLAY" "$ARTIFACT_DIR")"
PREFLIGHT_PATH="$RUN_DIR_PATH/preflight.json"
PREFLIGHT_DISPLAY="$(join_display_path "$RUN_DIR_DISPLAY" "preflight.json")"
STDOUT_LOG="$RUN_DIR_PATH/stdout.log"
STDERR_LOG="$RUN_DIR_PATH/stderr.log"
MANIFEST_PATH="$RUN_DIR_PATH/command_manifest.json"
SWEEP_OUT="$ARTIFACT_PATH/stopdff_dp_sweep.json"

PREFLIGHT_CMD=(
  "$PYTHON"
  "scripts/device2_cuda_preflight.py"
  "--out-dir" "$RUN_DIR_PATH"
  "--data-dir" "$DATA_DIR"
  "--calibration" "$CALIBRATION"
  "--fit-split" "$FIT_SPLIT"
  "--eval-split" "$EVAL_SPLIT"
  "--output-json" "$PREFLIGHT_PATH"
  "--device-index" "$DEVICE_INDEX"
  "--min-free-gib" "$MIN_FREE_GIB"
)
if [[ "$RESUME" -eq 1 ]]; then
  PREFLIGHT_CMD+=("--resume")
fi
if [[ "$OVERWRITE" -eq 1 ]]; then
  PREFLIGHT_CMD+=("--overwrite")
fi

SWEEP_CMD=(
  "$PYTHON"
  "scripts/sweep_stopdff_dp.py"
  "--artifact-dir" "$ARTIFACT_PATH"
  "--out" "$SWEEP_OUT"
  "--max-wall-hours" "$MAX_WALL_HOURS"
  "--num-bootstrap" "$NUM_BOOTSTRAP"
  "--n-jobs" "$N_JOBS"
  "--calibrators" "$CALIBRATORS"
  "--reward-schedules" "$REWARD_SCHEDULES"
  "--continuations" "$CONTINUATIONS"
  "--data-dir" "$DATA_DIR"
  "--calibration" "$CALIBRATION"
  "--fit-split" "$FIT_SPLIT"
  "--eval-split" "$EVAL_SPLIT"
)
if [[ "$RESUME" -eq 1 ]]; then
  SWEEP_CMD+=("--resume")
fi
if [[ "$SMOKE" -eq 1 ]]; then
  SWEEP_CMD+=("--smoke" "--max-cells" "${SMOKE_MAX_CELLS:-2}")
fi

PREFLIGHT_STDOUT="$(mktemp)"
PREFLIGHT_STDERR="$(mktemp)"
trap 'rm -f "$PREFLIGHT_STDOUT" "$PREFLIGHT_STDERR"' EXIT

preflight_status=0
CUDA_VISIBLE_DEVICES=0 "${PREFLIGHT_CMD[@]}" >"$PREFLIGHT_STDOUT" 2>"$PREFLIGHT_STDERR" || preflight_status=$?
cat "$PREFLIGHT_STDOUT"
cat "$PREFLIGHT_STDERR" >&2

if [[ "$preflight_status" -ne 0 ]]; then
  if [[ -d "$RUN_DIR_PATH" ]]; then
    cat "$PREFLIGHT_STDOUT" >> "$STDOUT_LOG"
    cat "$PREFLIGHT_STDERR" >> "$STDERR_LOG"
  fi
  exit "$preflight_status"
fi

cat "$PREFLIGHT_STDOUT" >> "$STDOUT_LOG"
cat "$PREFLIGHT_STDERR" >> "$STDERR_LOG"

exec > >(tee -a "$STDOUT_LOG") 2> >(tee -a "$STDERR_LOG" >&2)

mkdir -p "$ARTIFACT_PATH"

MANIFEST_TIMESTAMP="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
write_manifest "$MANIFEST_PATH"

printf '[device2_stopdff_run] run_dir=%s\n' "$RUN_DIR_DISPLAY"
printf '[device2_stopdff_run] artifact_path=%s\n' "$ARTIFACT_DISPLAY"
printf '[device2_stopdff_run] manifest=%s\n' "$MANIFEST_PATH"

CUDA_VISIBLE_DEVICES=0 "${SWEEP_CMD[@]}"
