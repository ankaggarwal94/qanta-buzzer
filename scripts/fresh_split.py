#!/usr/bin/env python3
"""
Fresh train/val/test split for CS321M v10 section 0.3 protocol.

Preserves old artifacts, creates a fresh split with a new random seed
(NOT 42, NOT 13), and writes provenance documentation to
PROJECT_WIKI/SPLIT_PROVENANCE.md.

Usage:
    python scripts/fresh_split.py            # Execute fresh split
    python scripts/fresh_split.py --dry-run  # Preview actions without modifying filesystem
    python scripts/fresh_split.py --seed 9973  # Override seed (must not be 42 or 13)
    python scripts/fresh_split.py --help

Inputs:
    - questions.csv (or HuggingFace fallback)
    - artifacts/ directory (if exists, will be preserved)
    - data/processed/ directory (if exists, will be preserved)

Outputs:
    - artifacts.pre_v10_freshsplit_{UTC_TIMESTAMP}/ (preserved old artifacts)
    - data/processed.pre_v10_freshsplit_{UTC_TIMESTAMP}/ (preserved old processed data)
    - data/processed/train_dataset.json (fresh train split)
    - data/processed/val_dataset.json (fresh val split)
    - data/processed/test_dataset.json (fresh test split)
    - PROJECT_WIKI/SPLIT_PROVENANCE.md (provenance documentation)

Exit codes:
    0 - Success
    1 - Error (missing data source, invalid seed, etc.)
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project path setup per convention
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from qb_data.data_loader import QANTADatasetLoader
from qb_data.dataset_splits import create_stratified_splits, save_splits


# Forbidden seeds per v10 section 0.3
FORBIDDEN_SEEDS = {42, 13}

# Split ratios matching configs/default.yaml
SPLIT_RATIOS = [0.7, 0.15, 0.15]


def generate_seed() -> int:
    """Generate a one-time freeze seed that is guaranteed not in FORBIDDEN_SEEDS.

    Uses ``int(time.time()) % 1000000 + 7919`` (prime offset to avoid
    collisions with clock-based seeds).

    NOTE: This seed varies per invocation because ``time.time()``
    advances. The DERIVATION RULE is reproducible (you can compute the
    formula's output), but the RESULT is not deterministic across runs.
    Once you have used this script for the canonical freeze, record the
    seed in ``PROJECT_WIKI/SPLIT_PROVENANCE.md`` and pass
    ``--seed <recorded_value>`` for all subsequent re-runs to reproduce
    the same partition. The re-freeze guard in ``main()`` enforces this
    by refusing to call ``generate_seed()`` once a recorded
    ``FRESH_SPLIT_SEED`` line exists in ``SPLIT_PROVENANCE.md``.

    Returns
    -------
    int
        A seed value guaranteed not in FORBIDDEN_SEEDS.
    """
    seed = int(time.time()) % 1000000 + 7919
    # Safety check: if by astronomical coincidence we hit a forbidden seed
    while seed in FORBIDDEN_SEEDS:
        seed += 1
    return seed


_PROVENANCE_SEED_RE = re.compile(r"^FRESH_SPLIT_SEED=(\d+)", re.MULTILINE)


def recorded_fresh_split_seed(project_root: Path) -> int | None:
    """Return the recorded FRESH_SPLIT_SEED from SPLIT_PROVENANCE.md, if any.

    Reads ``PROJECT_WIKI/SPLIT_PROVENANCE.md`` and returns the integer
    seed declared on a ``FRESH_SPLIT_SEED=<int>`` line. Returns None
    when the file is missing or the line is absent. This is the
    one-shot lookup used by the re-freeze guard (WR-08).

    Parameters
    ----------
    project_root : Path
        Path to the repository root.

    Returns
    -------
    int or None
        Recorded seed, or None if not present.
    """
    provenance = project_root / "PROJECT_WIKI" / "SPLIT_PROVENANCE.md"
    if not provenance.exists():
        return None
    match = _PROVENANCE_SEED_RE.search(provenance.read_text(encoding="utf-8"))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (ValueError, IndexError):
        return None


def get_git_commit_sha(project_root: Path | None = None) -> str:
    """Get current git HEAD commit SHA.

    The subprocess is invoked with ``cwd=`` pinned to ``project_root``
    (the qanta-buzzer repo root by default), so the recorded SHA always
    reflects the repo the script lives in, not whatever directory the
    operator happened to launch the script from. Without ``cwd=``, a
    caller running ``python /abs/path/to/scripts/fresh_split.py`` from
    an unrelated git checkout would silently record THAT checkout's HEAD
    (or ``'unknown'`` if the launch CWD is not a git repo), corrupting
    SPLIT_PROVENANCE.md (Codex PR #14 thread #3309098605).

    Parameters
    ----------
    project_root : Path or None
        Repository root to query. Defaults to two levels above this
        script (``scripts/fresh_split.py`` -> ``qanta-buzzer/``).

    Returns
    -------
    str
        Commit SHA, or 'unknown' if git is unavailable or the directory
        is not a git working tree.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Seed value to set for random, numpy, and torch (if importable).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not required for splitting


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list. None uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fresh train/val/test split per v10 section 0.3 protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override seed value (must NOT be 42 or 13). When omitted "
            "AND PROJECT_WIKI/SPLIT_PROVENANCE.md does not record a prior "
            "FRESH_SPLIT_SEED, a one-time seed is generated via "
            "`int(time.time()) %% 1000000 + 7919`. When the provenance "
            "file already records a FRESH_SPLIT_SEED, --seed must be "
            "passed (use the recorded value to reproduce the freeze, "
            "or a new value with --allow-reseed to intentionally "
            "re-freeze)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all actions without modifying the filesystem.",
    )
    parser.add_argument(
        "--allow-reseed",
        action="store_true",
        help=(
            "Override the re-freeze guard. By default, when "
            "PROJECT_WIKI/SPLIT_PROVENANCE.md already records a "
            "FRESH_SPLIT_SEED, this script refuses to invent a new seed "
            "via generate_seed() because doing so would silently "
            "produce a different partition than the recorded freeze. "
            "Pass --seed <recorded_value> to reproduce the existing "
            "split, or pass --allow-reseed to intentionally start over "
            "with a new freeze (also use --seed to make that new freeze "
            "reproducible)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the fresh-split protocol.

    Parameters
    ----------
    argv : list[str] or None
        CLI arguments. None uses sys.argv.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    args = parse_args(argv)

    project_root = Path(__file__).parent.parent
    utc_now = datetime.now(timezone.utc)
    # PR #14 follow-up review (Copilot #3308936234): include microsecond
    # resolution in archive directory names so two runs within the same
    # second do not collide on ``artifacts.pre_v10_freshsplit_<ts>`` /
    # ``data/processed.pre_v10_freshsplit_<ts>`` with FileExistsError after
    # partial side effects. The %f format produces 6 microsecond digits;
    # collision requires two runs within ~1µs which is below clock
    # resolution on any realistic host.
    utc_timestamp = utc_now.strftime("%Y%m%dT%H%M%S_%fZ")

    # WR-08: re-freeze guard. If SPLIT_PROVENANCE.md already records a
    # FRESH_SPLIT_SEED, refuse to invent a new one via generate_seed().
    # The recorded freeze is load-bearing on the audit card
    # (THRESHOLDS_FROZEN_AFTER_FRESH_SPLIT, audit-table citations); a
    # silent re-freeze with a different seed produces a different
    # partition while every downstream artifact keeps pointing at the
    # old seed. The operator must either pass --seed <recorded_value>
    # (reproduce) or --allow-reseed (intentional re-freeze).
    recorded_seed = recorded_fresh_split_seed(project_root)
    if args.seed is None and recorded_seed is not None and not args.allow_reseed:
        print(
            f"ERROR: PROJECT_WIKI/SPLIT_PROVENANCE.md already records "
            f"FRESH_SPLIT_SEED={recorded_seed}. Re-running without "
            f"--seed would invent a new seed and silently produce a "
            f"different partition than the recorded freeze.\n"
            f"  Reproduce the freeze: --seed {recorded_seed}\n"
            f"  Intentionally re-freeze: --allow-reseed --seed <new_value>",
            file=sys.stderr,
        )
        return 1

    # Determine seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = generate_seed()

    # Validate seed
    if seed in FORBIDDEN_SEEDS:
        print(f"ERROR: Seed {seed} is forbidden (must not be 42 or 13).", file=sys.stderr)
        return 1

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Fresh Split Protocol v10 section 0.3")
    print(f"  FRESH_SPLIT_SEED={seed}")
    print(f"  UTC_TIMESTAMP={utc_timestamp}")
    print(f"  Split ratios: {SPLIT_RATIOS}")
    print()

    # PR #14 follow-up review (Codex #3309002349): in the real (non-dry-run)
    # path, load + validate the question source BEFORE the Step 1 destructive
    # moves below. Previously the moves ran first and the question-load came
    # after; if questions.csv was missing/corrupt or the HuggingFace fallback
    # failed, the script would exit having already renamed ``artifacts/`` out
    # of place -- leaving the checkout in a half-moved state that broke every
    # downstream command. By front-loading the validation, the Step 1 moves
    # only execute once we know we have a valid question set in memory.
    # Dry-run keeps its own separate validation path further down.
    questions = None
    csv_path = project_root / "questions.csv"
    if not args.dry_run:
        print("Step 0: Loading + validating questions (before destructive moves)...")
        if csv_path.exists():
            print(f"  Loading from CSV: {csv_path}")
            loader = QANTADatasetLoader()
            questions = loader.load_from_csv(str(csv_path))
            print(f"  Loaded {len(questions)} questions from CSV")
        else:
            print(f"  CSV not found at {csv_path}, trying HuggingFace fallback...")
            try:
                from qb_data.huggingface_loader import load_from_huggingface
                questions = load_from_huggingface(
                    "qanta-challenge/acf-co24-tossups", split="eval"
                )
                print(f"  Loaded {len(questions)} questions from HuggingFace")
            except Exception as e:
                print(
                    f"ERROR: Could not load questions. CSV missing and "
                    f"HuggingFace failed: {e}",
                    file=sys.stderr,
                )
                return 1
        if not questions:
            print("ERROR: No questions loaded.", file=sys.stderr)
            return 1
        print(f"  Validated: {len(questions)} questions ready for splitting")
        print()

    # --- Step 1: Preserve old artifacts ---
    artifacts_dir = project_root / "artifacts"
    artifacts_archive = project_root / f"artifacts.pre_v10_freshsplit_{utc_timestamp}"

    if artifacts_dir.exists():
        print(f"Step 1a: Preserving artifacts/ -> {artifacts_archive.name}")
        if not args.dry_run:
            shutil.move(str(artifacts_dir), str(artifacts_archive))
            print(f"  Moved: {artifacts_dir} -> {artifacts_archive}")
        else:
            print(f"  [DRY-RUN] Would move: {artifacts_dir} -> {artifacts_archive}")
    else:
        print("Step 1a: SKIP - artifacts/ directory does not exist")
        artifacts_archive = None

    # --- Step 2: Preserve old data/processed ---
    processed_dir = project_root / "data" / "processed"
    processed_archive = project_root / "data" / f"processed.pre_v10_freshsplit_{utc_timestamp}"

    if processed_dir.exists():
        print(f"Step 1b: Preserving data/processed/ -> data/{processed_archive.name}")
        if not args.dry_run:
            # Iter1 IN-05: COPYTREE (not MOVE) is intentional, in
            # contrast to Step 1a above which uses shutil.move(...) on
            # the artifacts/ tree. The asymmetry is load-bearing:
            #   - data/processed/ must remain in place after this
            #     step because Step 5 (below) writes the new
            #     train/val/test splits INTO it. Moving the directory
            #     out from under Step 5 would either re-create it
            #     empty (losing every sibling file the operator left
            #     in data/processed/) or fail on the missing parent.
            #   - data/processed/mc_dataset.json was built by an
            #     earlier scripts/build_mc_dataset.py run. It is
            #     consumed downstream by scripts/compute_csli.py
            #     after fresh_split. A shutil.move here would orphan
            #     mc_dataset.json, breaking Phase 4 CSLI (and
            #     contradicting the WR-07 documented limitation that
            #     compute_csli.py reads mc_dataset.json from this
            #     directory).
            # A future maintainer "fixing" the inconsistency by
            # switching this call to shutil.move would silently break
            # the pipeline. Keep COPYTREE here; the artifacts/ tree
            # gets MOVED above because nothing downstream re-reads it
            # in place.
            shutil.copytree(str(processed_dir), str(processed_archive))
            print(f"  Copied: {processed_dir} -> {processed_archive}")
        else:
            print(f"  [DRY-RUN] Would copy: {processed_dir} -> {processed_archive}")
    else:
        print("Step 1b: SKIP - data/processed/ directory does not exist")
        processed_archive = None

    # --- Step 3: Set seeds ---
    print(f"\nStep 2: Setting all random seeds to {seed}")
    if not args.dry_run:
        set_all_seeds(seed)
    else:
        print(f"  [DRY-RUN] Would set random/numpy/torch seeds to {seed}")

    # --- Step 4: Load questions ---
    print("\nStep 3: Loading questions...")
    csv_path = project_root / "questions.csv"

    if args.dry_run:
        print(f"  [DRY-RUN] Would load from: {csv_path}")
        if csv_path.exists():
            print(f"  [DRY-RUN] CSV file exists at {csv_path}")
        else:
            print(f"  [DRY-RUN] CSV not found; would try HuggingFace fallback")

        # Iter1 IN-04: data-quality validation during --dry-run.
        # The prior --dry-run printed what WOULD happen but never
        # touched the loader, so a corrupt CSV / wrong-schema HF
        # dataset surfaced only on the real run -- AFTER artifacts/
        # had been moved and data/processed/ had been copied (Step 1
        # has already completed by this point). Validating up-front
        # in --dry-run avoids that destructive failure mode.
        #
        # The validation is BEST-EFFORT: a failure here does NOT
        # abort the dry-run (the user may be inspecting the seed +
        # provenance template even when the data source is
        # intentionally absent). It reports the error and continues
        # so the rest of the --dry-run output still renders.
        if csv_path.exists():
            try:
                loader = QANTADatasetLoader()
                n = len(loader.load_from_csv(str(csv_path)))
                print(f"  [DRY-RUN] CSV validates; would load {n} questions")
            except Exception as exc:
                print(
                    f"  [DRY-RUN] CSV load would FAIL: {exc} "
                    "(fix before the real run -- artifacts have NOT "
                    "been touched in dry-run mode)",
                    file=sys.stderr,
                )
        else:
            try:
                from qb_data.huggingface_loader import load_from_huggingface

                hf_questions = load_from_huggingface(
                    "qanta-challenge/acf-co24-tossups", split="eval"
                )
                print(
                    f"  [DRY-RUN] HuggingFace fallback validates; "
                    f"would load {len(hf_questions)} questions"
                )
            except Exception as exc:
                print(
                    f"  [DRY-RUN] HuggingFace fallback would FAIL: "
                    f"{exc} (fix before the real run -- artifacts "
                    "have NOT been touched in dry-run mode)",
                    file=sys.stderr,
                )

        # For dry-run, report provenance template and exit
        print("\nStep 4: [DRY-RUN] Would create stratified splits with ratios", SPLIT_RATIOS)
        print("\nStep 5: [DRY-RUN] Would save splits to data/processed/")
        print("\nStep 6: [DRY-RUN] Would write PROJECT_WIKI/SPLIT_PROVENANCE.md:")
        print(f"  FRESH_SPLIT_SEED={seed}")
        print(f"  FRESH_SPLIT_CREATED_AT={utc_now.isoformat()}")
        print(f"  FRESH_SPLIT_COMMIT_SHA={get_git_commit_sha()}")
        old_path = str(artifacts_archive) if artifacts_archive else "N/A (no artifacts/ existed)"
        print(f"  OLD_SPLIT_PRESERVED_AT={old_path}")
        print(f"  THRESHOLDS_FROZEN_AFTER_FRESH_SPLIT=false")
        print(f"  TEST_SPLIT_INSPECTED_POST_FRESH_SPLIT=false")
        print("\n[DRY-RUN] Complete. No filesystem changes made.")
        return 0

    # Real-run question loading is now done in Step 0 (before destructive
    # moves) per Codex #3309002349; ``questions`` is already populated and
    # validated by the time we reach this point.

    # --- Step 5: Create fresh splits ---
    print(f"\nStep 4: Creating stratified splits (seed={seed}, ratios={SPLIT_RATIOS})")
    train, val, test = create_stratified_splits(
        questions,
        ratios=SPLIT_RATIOS,
        seed=seed,
    )

    # --- Step 6: Save splits ---
    output_dir = project_root / "data" / "processed"
    print(f"\nStep 5: Saving splits to {output_dir}/")
    save_splits(train, val, test, output_dir=str(output_dir))

    # --- Step 7: Write provenance documentation ---
    wiki_dir = project_root / "PROJECT_WIKI"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = wiki_dir / "SPLIT_PROVENANCE.md"

    commit_sha = get_git_commit_sha()
    old_artifacts_path = str(artifacts_archive.relative_to(project_root)) if artifacts_archive else "N/A (no artifacts/ existed)"
    old_processed_path = str(processed_archive.relative_to(project_root)) if processed_archive else "N/A (no data/processed/ existed)"

    # PR #14 follow-up review (Codex #3308590302): preserve frozen-threshold
    # provenance on split reruns. When the user reruns fresh_split with
    # --seed <recorded_value> to reproduce the canonical split AFTER thresholds
    # were frozen (THRESHOLDS_FROZEN_AFTER_FRESH_SPLIT=true +
    # THRESHOLD_MANIFEST_SHA256 + THRESHOLD_FREEZE_TIMESTAMP), this rewrite
    # would clobber those fields back to "false" + omit the hash, breaking
    # compute_csli._load_split_provenance which relies on the hash to verify
    # the split/threshold freeze. Preserve the freeze fields unless the
    # operator explicitly opted into a re-freeze via --allow-reseed.
    existing_freeze_fields: dict[str, str] = {}
    if provenance_path.exists() and not args.allow_reseed:
        existing_text = provenance_path.read_text(encoding="utf-8")
        for line in existing_text.splitlines():
            stripped = line.strip()
            for key in (
                "THRESHOLDS_FROZEN_AFTER_FRESH_SPLIT",
                "THRESHOLD_MANIFEST_SHA256",
                "THRESHOLD_FREEZE_TIMESTAMP",
                "TEST_SPLIT_INSPECTED_POST_FRESH_SPLIT",
            ):
                prefix = f"{key}="
                if stripped.startswith(prefix):
                    existing_freeze_fields[key] = stripped[len(prefix):]

    if existing_freeze_fields.get("THRESHOLDS_FROZEN_AFTER_FRESH_SPLIT") == "true":
        thresholds_frozen_line = "THRESHOLDS_FROZEN_AFTER_FRESH_SPLIT=true"
        # Preserve the manifest hash + freeze timestamp verbatim if recorded.
        manifest_sha_line = (
            f"\nTHRESHOLD_MANIFEST_SHA256={existing_freeze_fields['THRESHOLD_MANIFEST_SHA256']}"
            if "THRESHOLD_MANIFEST_SHA256" in existing_freeze_fields
            else ""
        )
        freeze_timestamp_line = (
            f"\nTHRESHOLD_FREEZE_TIMESTAMP={existing_freeze_fields['THRESHOLD_FREEZE_TIMESTAMP']}"
            if "THRESHOLD_FREEZE_TIMESTAMP" in existing_freeze_fields
            else ""
        )
    else:
        thresholds_frozen_line = "THRESHOLDS_FROZEN_AFTER_FRESH_SPLIT=false"
        manifest_sha_line = ""
        freeze_timestamp_line = ""

    test_inspected_line = (
        f"TEST_SPLIT_INSPECTED_POST_FRESH_SPLIT="
        f"{existing_freeze_fields.get('TEST_SPLIT_INSPECTED_POST_FRESH_SPLIT', 'false')}"
    )

    total = len(train) + len(val) + len(test)
    provenance_content = f"""# Split Provenance

## Fresh Split Gate Fields (v10 section 0.3)

```
FRESH_SPLIT_SEED={seed}
FRESH_SPLIT_CREATED_AT={utc_now.isoformat()}
FRESH_SPLIT_COMMIT_SHA={commit_sha}
OLD_SPLIT_PRESERVED_AT={old_artifacts_path}
{thresholds_frozen_line}{manifest_sha_line}{freeze_timestamp_line}
{test_inspected_line}
```

## Split Statistics

| Split | Count | Percentage |
|-------|-------|------------|
| Train | {len(train)} | {len(train)/total:.1%} |
| Val   | {len(val)} | {len(val)/total:.1%} |
| Test  | {len(test)} | {len(test)/total:.1%} |
| **Total** | **{total}** | **100.0%** |

## Preservation Log

- Old artifacts preserved at: `{old_artifacts_path}`
- Old processed data preserved at: `{old_processed_path}`
- Fresh split output: `data/processed/`

## Ratios

- Train ratio: {SPLIT_RATIOS[0]}
- Val ratio: {SPLIT_RATIOS[1]}
- Test ratio: {SPLIT_RATIOS[2]}

## Integrity Notes

- Seed {seed} is NOT 42 (configs/default.yaml data.shuffle_seed) and NOT 13 (configs/default.yaml environment.seed)
- All random generators (random, numpy, torch) seeded before split
- Stratified by category to preserve distribution across splits
- No test-set content inspected or printed during this operation
"""

    print(f"\nStep 6: Writing provenance to {provenance_path}")
    with open(provenance_path, "w", encoding="utf-8") as f:
        f.write(provenance_content)
    print(f"  Written: {provenance_path}")

    print("\nFresh split protocol complete.")
    print(f"  Seed: {seed}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
