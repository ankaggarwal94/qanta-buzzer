"""Mini-audit fix-round-3 regression suite (MA-RB-001, MA-HI-001, MA-HI-002,
MA-HAI-001, MA-PI-001, MA-HI-004, MA-CC-1, MA-CC-3, MA-CC-5).

Each BLOCKING mini-audit finding gets an instance + class regression here.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import pairing, schema, verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    FAKE_COMMIT,
    REPO_ROOT,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    VERDICT_SOURCE_PASS,
    build_package,
    cli_args_for,
    cli_subprocess_env,
    colm_no_network,
    expected_estimand_digest,
    make_ledger,
    make_ledger_row,
    make_record,
    repo_head_commit,
    run_cli,
    standard_records,
)


def _run_release(pkg):
    return verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )


def _run_source(pkg):
    return verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )


def _failing(report):
    return [leg for leg in report.legs if leg.get("outcome") == "FAIL"]


def _leg(report, leg_id):
    return [leg for leg in report.legs if leg.get("leg_id") == leg_id]


# ---------------------------------------------------------------------------
# MA-RB-001 (R-015): bootstrap draw-count ceiling
# ---------------------------------------------------------------------------


def test_max_bootstrap_draws_is_pinned_and_sane():
    assert 0 < schema.MAX_BOOTSTRAP_DRAWS <= 1_000_000


def test_oversized_draw_count_rejected_before_allocation():
    # MA-RB-001 [R-015]: draw_count=10**9 is refused with a typed error
    # BEFORE build_bootstrap_plan allocates a 24GB+ resample matrix.
    spec = {
        "procedure": "percentile_bootstrap",
        "draw_count": 10 ** 9,
        "resampling_seeds": [1],
        "statistic": "signed_index_mean",
    }
    with pytest.raises(schema.SchemaValidationError) as exc:
        pairing.recompute_interval(standard_records(), spec)
    assert "draw" in str(exc.value).lower()


def test_normal_draw_count_still_recomputes():
    # MA-RB-001: a normal draw_count is unaffected.
    spec = {
        "procedure": "percentile_bootstrap",
        "draw_count": 100,
        "resampling_seeds": [1],
        "statistic": "signed_index_mean",
    }
    result = pairing.recompute_interval(standard_records(), spec)
    assert isinstance(result["ci"], list) and len(result["ci"]) == 2


def test_oversized_draw_count_in_cell_fails_source_not_hangs(tmp_path: Path):
    # MA-RB-001 [R-015]: the ceiling is reachable in cheap SOURCE mode via a
    # cell interval — a huge draw_count is a leg FAIL (verdict reached), never
    # an OOM/hang. Digest recomputed so the interval gate is the only defect.
    def blow_up(profile):
        cell = profile["cells"][0]
        cell["interval"]["draw_count"] = 10 ** 9

    pkg = build_package(tmp_path, profile_mutator=blow_up)
    report = _run_source(pkg)
    assert report.verdict == VERDICT_FAIL
    assert report.receipt_path is not None  # a verdict was reached


def test_max_admissible_ints_table_present():
    # MA-RB-001 class fix: a MAX_* admissibility table for artifact-derived
    # ints that size allocations.
    assert set(schema.MAX_ADMISSIBLE_INTS) >= {
        "artifact_bytes",
        "bootstrap_draws",
        "bootstrap_cells",
    }
    assert schema.MAX_ADMISSIBLE_INTS["bootstrap_draws"] == (
        schema.MAX_BOOTSTRAP_DRAWS
    )


# ---------------------------------------------------------------------------
# MA-HI-001 (R-020): bounded, regular-file-only untrusted reads
# ---------------------------------------------------------------------------


def test_symlink_to_dev_null_rejected_fast(tmp_path: Path):
    # MA-HI-001 [R-020]: a symlink (here -> /dev/null) at an untrusted path is
    # rejected without following it — no is_file() trust, no blocking read.
    link = tmp_path / "expect.json"
    link.symlink_to("/dev/null")
    with pytest.raises(schema.TypedIngressError) as exc:
        schema.read_regular_file_bytes(link)
    assert "symlink" in str(exc.value).lower()
    assert str(tmp_path) not in str(exc.value)


def test_oversized_file_rejected_by_size_ceiling(tmp_path: Path):
    # MA-HI-001 [R-020]: a file above the byte ceiling is refused.
    big = tmp_path / "big.json"
    big.write_bytes(b"x" * 2048)
    with pytest.raises(schema.TypedIngressError):
        schema.read_regular_file_bytes(big, max_bytes=1024)


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="platform without os.mkfifo"
)
def test_fifo_at_expectations_rejected_without_hanging(tmp_path: Path):
    # MA-HI-001 [R-020]: a FIFO at --expectations is rejected FAST (S_ISREG
    # guard) — an unguarded read_bytes() would block forever. Hermetic: the
    # read runs in a worker thread with a hard timeout; if the guard failed,
    # the thread would still be alive after the timeout (no writer end).
    fifo = tmp_path / "expect.fifo"
    os.mkfifo(fifo)
    result: dict[str, object] = {}

    def worker():
        try:
            schema.read_regular_file_bytes(fifo)
            result["outcome"] = "read"
        except schema.TypedIngressError as exc:
            result["outcome"] = "rejected"
            result["msg"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            result["outcome"] = f"error:{exc.__class__.__name__}"

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "read hung on a FIFO (S_ISREG guard missing)"
    assert result.get("outcome") == "rejected", result
    assert "regular file" in str(result.get("msg", "")).lower()


def test_regular_file_reads_normally(tmp_path: Path):
    # MA-HI-001: the honest case is unchanged.
    p = tmp_path / "ok.json"
    p.write_bytes(b'{"schema_version": 1}\n')
    assert schema.read_regular_file_bytes(p) == b'{"schema_version": 1}\n'


def test_cli_fifo_expectations_does_not_hang(tmp_path: Path):
    # MA-HI-001 [R-020/R-037]: the FIFO reaches the CLI as a typed refusal,
    # not a hang — bounded via subprocess timeout.
    if not hasattr(os, "mkfifo"):
        pytest.skip("platform without os.mkfifo")
    pkg = build_package(tmp_path)
    fifo = tmp_path / "expect.fifo"
    os.mkfifo(fifo)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "reproducibility.colm_aims_2026.verify",
            "--mode",
            "release",
            "--tree",
            str(pkg.tree),
            "--expectations",
            str(fifo),
            "--receipts-dir",
            str(pkg.receipts_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=cli_subprocess_env(),
        timeout=30,
        check=False,
    )
    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# MA-HI-002 (R-036/R-013): symlinked tree members refused
# ---------------------------------------------------------------------------


def test_symlink_tree_member_pointing_outside_is_refused(tmp_path: Path):
    # MA-HI-002 [R-036/R-013]: a symlinked member inside --tree pointing to a
    # file OUTSIDE the tree is refused (typed containment error), never read
    # or hashed into input_tree_sha256.
    outside = tmp_path / "secret_outside.txt"
    outside.write_bytes(b"external bytes never hashed\n")
    pkg = build_package(tmp_path)
    (pkg.tree / "sneaky.json").symlink_to(outside)
    with pytest.raises(verifier.ContainmentError):
        _run_source(pkg)


def test_snapshot_refuses_symlink_member(tmp_path: Path):
    # MA-HI-002: the tree-walk helper refuses a symlink member directly.
    pkg = build_package(tmp_path)
    (pkg.tree / "alias.json").symlink_to(pkg.tree / "profile.json")
    with pytest.raises(verifier.ContainmentError):
        verifier._read_tree_snapshot(pkg.tree)


def test_honest_tree_snapshot_matches_disk(tmp_path: Path):
    # MA-HI-002/MA-HI-004: the honest snapshot covers exactly the real files.
    pkg = build_package(tmp_path)
    snap = verifier._read_tree_snapshot(pkg.tree)
    assert set(snap) == {
        "profile.json",
        "records.jsonl",
        "presentation_manifest.json",
        "sealed-notes.bin",
    }
    assert snap["profile.json"] == pkg.profile_path.read_bytes()


# ---------------------------------------------------------------------------
# MA-HI-004 (R-036/R-014): snapshot-then-verify (no gate/receipt desync)
# ---------------------------------------------------------------------------


def test_tree_read_once_gate_and_receipt_cannot_desync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # MA-HI-004 [R-036]: the tree snapshot is read ONCE at ingress; a
    # concurrent writer that mutates a tree file AFTER the snapshot cannot
    # desync the gate from the receipt. We wrap the snapshot reader to mutate
    # sealed-notes.bin on disk right after it returns; both the tree_files
    # gate and the receipt digest must reflect the SNAPSHOT (pre-mutation)
    # bytes, proving nothing re-read from disk.
    pkg = build_package(tmp_path)
    real = verifier._read_tree_snapshot
    calls = {"n": 0}

    def wrapper(tree):
        snap = real(tree)
        calls["n"] += 1
        # Mutate on disk AFTER the one-shot snapshot.
        (pkg.tree / "sealed-notes.bin").write_bytes(b"MUTATED-AFTER-SNAPSHOT\n")
        return snap

    monkeypatch.setattr(verifier, "_read_tree_snapshot", wrapper)
    report = _run_release(pkg)
    assert calls["n"] == 1, "tree snapshot must be read exactly once per run"
    # The passing release verdict + receipt attest the pre-mutation bytes.
    assert report.verdict == VERDICT_RELEASE_PASS
    receipt = json.loads(Path(report.receipt_path).read_text("utf-8"))
    # Independent digest over the SNAPSHOT bytes (pre-mutation).
    snap = real(pkg.tree)  # note: this re-reads current disk (post-mutation)
    # The receipt must NOT equal the post-mutation digest — it froze the
    # snapshot bytes.
    post_lines = [
        f"{rel}:{hashlib.sha256(data).hexdigest()}"
        for rel, data in sorted(snap.items())
    ]
    post_digest = hashlib.sha256(
        ("\n".join(post_lines) + "\n").encode("utf-8")
    ).hexdigest()
    assert receipt["input_tree_sha256"] != post_digest


def test_receipt_digest_equals_snapshot_digest(tmp_path: Path):
    # MA-HI-004: for an unmutated honest run the receipt digest equals the
    # independent snapshot digest (the single source of truth).
    pkg = build_package(tmp_path)
    report = _run_source(pkg)
    snap = verifier._read_tree_snapshot(pkg.tree)
    lines = [
        f"{rel}:{hashlib.sha256(data).hexdigest()}"
        for rel, data in sorted(snap.items())
    ]
    expected = hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()
    receipt = json.loads(Path(report.receipt_path).read_text("utf-8"))
    assert receipt["input_tree_sha256"] == expected


# ---------------------------------------------------------------------------
# MA-HAI-001 (R-011/R-025): estimand-field reconciliation
# ---------------------------------------------------------------------------


def _rehash_estimand(profile):
    cell = profile["cells"][0]
    cell["estimand_digest"] = expected_estimand_digest(cell["estimand"])


def _mut_calibration(profile):
    profile["cells"][0]["estimand"]["calibration_identity"] = "cal-FAKE"
    _rehash_estimand(profile)


def _mut_continuation(profile):
    profile["cells"][0]["estimand"]["continuation_identity"] = "cont-FAKE"
    _rehash_estimand(profile)


def _mut_denominator(profile):
    profile["cells"][0]["estimand"]["denominator_policy"] = "n_pairing_population"
    _rehash_estimand(profile)


def _mut_timeout(profile):
    profile["cells"][0]["estimand"]["timeout_parameters"]["trajectory_horizon"] = 99
    _rehash_estimand(profile)


def _mut_arm(profile):
    profile["cells"][0]["estimand"]["arm_mc"] = "arm-ghost"
    _rehash_estimand(profile)


def _mut_random_k(profile):
    profile["cells"][0]["estimand"]["random_k_draw_id"] = "draw-favorable-7"
    _rehash_estimand(profile)


ESTIMAND_PROBES = [
    ("calibration_identity", _mut_calibration),
    ("continuation_identity", _mut_continuation),
    ("denominator_policy", _mut_denominator),
    ("timeout_horizon", _mut_timeout),
    ("arm_id", _mut_arm),
    ("random_k_draw", _mut_random_k),
]


@pytest.mark.parametrize(
    "name,mutate", ESTIMAND_PROBES, ids=[n for n, _ in ESTIMAND_PROBES]
)
def test_fabricated_estimand_field_fails_release(tmp_path: Path, name, mutate):
    # MA-HAI-001 [R-011/R-025]: each fabricated estimand field (with the
    # self-recomputed digest kept consistent) fails the estimand
    # reconciliation leg — including R-025's substituted favorable draw.
    pkg = build_package(tmp_path, profile_mutator=mutate)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL, name
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "estimand_reconciliation"
    ]
    assert legs, f"{name}: estimand reconciliation leg did not fire"


def test_honest_estimand_reconciles(tmp_path: Path):
    # MA-HAI-001: the honest package passes the reconciliation leg.
    pkg = build_package(tmp_path)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS
    assert _leg(report, "estimand_reconciliation")[0]["outcome"] == "PASS"


def test_authorized_multidraw_random_k_reconciles(tmp_path: Path):
    # MA-HAI-001 [R-025]: a real draw declared in the estimand PASSES only
    # when the frozen ledger authorizes it via a sanctioned Random-K
    # disposition naming that draw.
    def mut_estimand(profile):
        profile["cells"][0]["estimand"]["random_k_draw_id"] = "draw-A"
        _rehash_estimand(profile)

    def authorize(ledger):
        ledger["rows"].append(
            make_ledger_row(
                claim_id="clm-rk-A",
                artifact_family="random_k",
                artifact_id="records.jsonl",
                provenance_class="historical_randomk_v5",
                status="UNVERIFIED",
                headline_eligible=True,
                author_decision="predeclared_multidraw_family",
                random_k_draw_id="draw-A",
            )
        )

    pkg = build_package(
        tmp_path, profile_mutator=mut_estimand, ledger_mutator=authorize
    )
    report = _run_release(pkg)
    assert _leg(report, "estimand_reconciliation")[0]["outcome"] == "PASS"


# ---------------------------------------------------------------------------
# MA-PI-001 (R-013): anchor.ledger_path & rights path containment
# ---------------------------------------------------------------------------


def _abs_ledger_path(pkg):
    return lambda exp: exp["anchor"].update(
        ledger_path=str(pkg.ledger_path.resolve())
    )


PI_LEDGER_MUTATIONS = {
    "absolute": lambda exp, pkg: exp["anchor"].update(
        ledger_path=str(pkg.ledger_path.resolve())
    ),
    "dotdot_escape": lambda exp, pkg: exp["anchor"].update(
        ledger_path="../ledger.json"
    ),
    "in_tree_collapse": lambda exp, pkg: exp["anchor"].update(
        ledger_path="tree/ledger.json"
    ),
}


@pytest.mark.parametrize("name", sorted(PI_LEDGER_MUTATIONS))
def test_non_contained_ledger_path_fails_release(tmp_path: Path, name):
    # MA-PI-001 [R-013]: an absolute path, a `..` escape, or an in-tree
    # collapse of anchor.ledger_path each fails the anchor_ledger leg.
    mutate = PI_LEDGER_MUTATIONS[name]
    pkg = build_package(tmp_path)
    # Copy the ledger to the in-tree location for the collapse case so the
    # only defect is the containment violation, not an absent file.
    if name == "in_tree_collapse":
        (pkg.tree / "ledger.json").write_bytes(pkg.ledger_path.read_bytes())
    from tests._colm_aims_helpers import rewrite_json

    rewrite_json(pkg.expectations_path, lambda exp: mutate(exp, pkg))
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL, name
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "anchor_ledger"
    ]
    assert legs, f"{name}: anchor_ledger containment leg did not fire"


def test_symlink_redirect_ledger_path_refused(tmp_path: Path):
    # MA-PI-001 [R-013]: a relative ledger_path that is a symlink redirecting
    # OUTSIDE the bundle is refused (resolve-and-contain-under-base).
    pkg = build_package(tmp_path)
    outside = tmp_path / "elsewhere_ledger.json"
    outside.write_bytes(pkg.ledger_path.read_bytes())
    link = pkg.root / "redir.json"
    link.symlink_to(outside)
    from tests._colm_aims_helpers import rewrite_json

    rewrite_json(
        pkg.expectations_path,
        lambda exp: exp["anchor"].update(ledger_path="redir.json"),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert [
        leg for leg in _failing(report) if leg.get("leg_id") == "anchor_ledger"
    ]


def test_non_contained_rights_path_fails_release(tmp_path: Path):
    # MA-PI-001 [R-013]: the rights inventory reference is contained the same
    # way — an absolute path fails the rights_inventory leg.
    pkg = build_package(tmp_path)
    from tests._colm_aims_helpers import rewrite_json

    rewrite_json(
        pkg.expectations_path,
        lambda exp: exp["rights_inventory"].update(
            path=str(pkg.rights_path.resolve())
        ),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "rights_inventory"
    ]


def test_pristine_relative_sibling_paths_pass(tmp_path: Path):
    # MA-PI-001 guard: the pristine relative-sibling ledger/rights paths still
    # PASS release.
    pkg = build_package(tmp_path)
    assert _run_release(pkg).verdict == VERDICT_RELEASE_PASS


# MA-PI-001 class fix: enumerate ALL anchor/expectations-referenced path
# fields and assert each passes through the containment guard.
ANCHOR_REFERENCED_PATH_FIELDS = [
    ("anchor.ledger_path", ("anchor", "ledger_path"), "anchor_ledger"),
    (
        "rights_inventory.path",
        ("rights_inventory", "path"),
        "rights_inventory",
    ),
]


@pytest.mark.parametrize(
    "name,field_path,leg_id",
    ANCHOR_REFERENCED_PATH_FIELDS,
    ids=[n for n, _, _ in ANCHOR_REFERENCED_PATH_FIELDS],
)
def test_every_referenced_path_field_is_containment_guarded(
    tmp_path: Path, name, field_path, leg_id
):
    # MA-PI-001 class fix: every referenced sidecar path field, set to an
    # absolute path, fails its leg (no field escapes the guard).
    pkg = build_package(tmp_path)
    from tests._colm_aims_helpers import rewrite_json

    def mutate(exp):
        node = exp
        for key in field_path[:-1]:
            node = node[key]
        node[field_path[-1]] = "/etc/passwd"

    rewrite_json(pkg.expectations_path, mutate)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL, name
    assert [
        leg for leg in _failing(report) if leg.get("leg_id") == leg_id
    ], f"{name}: containment guard did not fire on the {leg_id} leg"


# ---------------------------------------------------------------------------
# MA-CC-1 (R-037): sys.path dedupe + force-front bootstrap
# ---------------------------------------------------------------------------


def test_direct_path_forces_repo_root_ahead_of_stale_checkout(tmp_path: Path):
    # MA-CC-1 [R-037]: seed sys.path with a stale shadow AHEAD of the repo
    # root (repo root present but not first — the membership-guard hazard),
    # load verify.py by file path so its bootstrap runs, and assert the repo
    # root is force-fronted to sys.path[0] exactly once (no duplicate), so the
    # gate code the receipt SHA-stamps resolves from THIS repo.
    stale = tmp_path / "stale_checkout"
    (stale / "reproducibility" / "colm_aims_2026").mkdir(parents=True)
    (stale / "reproducibility" / "__init__.py").write_text("", encoding="utf-8")
    (stale / "reproducibility" / "colm_aims_2026" / "__init__.py").write_text(
        "STALE_SHADOW = True\n", encoding="utf-8"
    )
    repo = str(REPO_ROOT)
    verify_py = str(REPO_ROOT / "reproducibility" / "colm_aims_2026" / "verify.py")
    probe = (
        "import importlib.util, sys\n"
        "stale, repo, target = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "sys.path.insert(0, repo)\n"
        "sys.path.insert(0, stale)\n"
        "spec = importlib.util.spec_from_file_location('_verify_under_test', target)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "if sys.path[0] != repo:\n"
        "    raise SystemExit('REPO_NOT_FRONTED:%r' % sys.path[:3])\n"
        "if sys.path.count(repo) != 1:\n"
        "    raise SystemExit('REPO_DUPLICATED:%d' % sys.path.count(repo))\n"
        "import reproducibility.colm_aims_2026 as pkg\n"
        "if getattr(pkg, 'STALE_SHADOW', False):\n"
        "    raise SystemExit('STALE_PACKAGE_LOADED')\n"
        "print('REAL_FRONTED')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, str(stale), repo, verify_py],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=cli_subprocess_env(),
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr[-400:])
    assert "REAL_FRONTED" in proc.stdout


# ---------------------------------------------------------------------------
# MA-CC-3 (R-016/R-039): cross-device publish leaves no relic
# ---------------------------------------------------------------------------


def _stage(base: Path, name: str, content: str = '{"v": 1}\n') -> Path:
    staged = base / f"staged-{name}"
    staged.mkdir(parents=True)
    (staged / "profile.json").write_text(content, encoding="utf-8")
    return staged


def test_cross_device_exdev_leaves_no_relic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # MA-CC-3 [R-016/R-039]: a cross-device publish (os.rename EXDEV after the
    # mkdir claim) is reclaimed so runs_root stays empty and the caller gets a
    # typed error — never a permanent poisoned relic.
    runs_root = tmp_path / "runs"

    def exdev(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "rename", exdev)
    with pytest.raises(schema.ColmAimsError) as exc:
        schema.publish_evidence_package(
            _stage(tmp_path, "a"), runs_root, "run-0001"
        )
    assert "EXDEV" in str(exc.value) or "cross-device" in str(exc.value).lower()
    monkeypatch.undo()
    assert list(runs_root.iterdir()) == [], "no relic may remain after EXDEV"


def test_cross_device_by_stdev_fails_before_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # MA-CC-3 [R-016/R-039]: a genuine device mismatch is caught by the
    # same-st_dev check BEFORE any mkdir claim — dest is never created.
    runs_root = tmp_path / "runs"
    real_stat = os.stat

    def fake_stat(path, *a, **k):
        st = real_stat(path, *a, **k)
        # Report the staged dir on a different device.
        if "staged-" in str(path):
            return os.stat_result(
                (st.st_mode, st.st_ino, st.st_dev + 999)
                + tuple(st)[3:]
            )
        return st

    monkeypatch.setattr(os, "stat", fake_stat)
    with pytest.raises(schema.ColmAimsError) as exc:
        schema.publish_evidence_package(
            _stage(tmp_path, "a"), runs_root, "run-0001"
        )
    assert "same filesystem" in str(exc.value).lower()
    monkeypatch.undo()
    assert not (runs_root / "run-0001").exists(), (
        "dest must never be claimed on a cross-device refusal"
    )


def test_same_device_honest_publish_still_works(tmp_path: Path):
    # MA-CC-3 guard: the honest same-filesystem publish is unchanged.
    runs_root = tmp_path / "runs"
    published = schema.publish_evidence_package(
        _stage(tmp_path, "a"), runs_root, "run-0001"
    )
    assert (published / "profile.json").read_text("utf-8") == '{"v": 1}\n'


# ---------------------------------------------------------------------------
# MA-CC-5 (R-013/R-022): git object-existence bound + SKIPPED leg + no env door
# ---------------------------------------------------------------------------


def test_false_branch_fails_object_leg(tmp_path: Path):
    # MA-CC-5 [R-013]: a valid 40-hex commit that does NOT exist, with the
    # source repo available, FAILs the anchor_source_commit_object leg (the
    # False branch — previously untested).
    nonexistent = "b" * 40
    pkg = build_package(tmp_path, source_commit=nonexistent)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "anchor_source_commit_object"
    ]
    assert legs, "the nonexistent-commit False branch did not FAIL the object leg"


def test_skipped_leg_when_git_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # MA-CC-5: with the source repo bound to a non-git dir, the object leg is
    # SKIPPED-with-reason (rendered + receipted), never silently passed.
    monkeypatch.setattr(verifier, "_SOURCE_REPO", tmp_path)
    pkg = build_package(tmp_path, source_commit=FAKE_COMMIT)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS
    legs = [
        leg
        for leg in report.legs
        if leg.get("leg_id") == "anchor_source_commit_object"
    ]
    assert legs and legs[0]["outcome"] == "SKIPPED"
    assert legs[0].get("reason")
    from reproducibility.colm_aims_2026 import render

    summary = render.render_summary(report)
    assert "skipped legs:" in summary
    assert "anchor_source_commit_object" in summary


def test_git_env_door_cannot_flip_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # MA-CC-5 [R-022]: a GIT_DIR pointing at an unrelated empty repo in the
    # environment cannot make a nonexistent commit "exist". The check binds to
    # this repo and scrubs GIT_* — so the False branch still FAILs.
    bogus_git = tmp_path / "bogus.git"
    subprocess.run(
        ["git", "init", "--bare", str(bogus_git)],
        capture_output=True,
        check=False,
    )
    monkeypatch.setenv("GIT_DIR", str(bogus_git))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    pkg = build_package(tmp_path, source_commit="c" * 40)
    report = _run_release(pkg)
    # The env door does not turn the missing object into a PASS.
    assert report.verdict == VERDICT_FAIL
    assert [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "anchor_source_commit_object"
    ]


def test_git_env_denylist_scrubbed_from_child():
    # MA-CC-5 [R-022]: the denylist names the redirection vars the code reads.
    assert {"GIT_DIR", "GIT_WORK_TREE", "GIT_CEILING_DIRECTORIES"} <= (
        verifier._GIT_ENV_DENYLIST
    )


def test_honest_head_commit_passes_object_leg(tmp_path: Path):
    # MA-CC-5 guard: the real HEAD commit passes the object leg (True branch).
    commit = repo_head_commit()
    if commit == FAKE_COMMIT:
        pytest.skip("no real git HEAD available")
    pkg = build_package(tmp_path, source_commit=commit)
    report = _run_release(pkg)
    legs = [
        leg
        for leg in report.legs
        if leg.get("leg_id") == "anchor_source_commit_object"
    ]
    assert legs and legs[0]["outcome"] in ("PASS", "SKIPPED")
