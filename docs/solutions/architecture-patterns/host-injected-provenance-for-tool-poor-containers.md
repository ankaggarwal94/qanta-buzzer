---
title: Host-injected provenance for tool-poor container runtimes
date: 2026-05-27
category: architecture-patterns
module: modal_pipeline_provenance
problem_type: architecture_pattern
component: tooling
severity: high
applies_when:
  - A producer running inside a container records host-derived metadata (git SHA, build ID, hostname, secret reference) into an artifact provenance block
  - The container's base image lacks the binary the producer would otherwise invoke (e.g., Modal debian_slim has no `git`; distroless lacks shell tools; minimal Alpine images drop most VCS clients)
  - The host has authoritative access to the value AND the same code revision being audited
  - The artifact's provenance is load-bearing for audit reproducibility (cache-keying, paper-evidence integrity, manifest verification)
  - The same producer code is expected to run unchanged in local-dev (where the tool is available) and cloud (where it isn't)
related_components:
  - scripts/_common.py:build_generation_provenance
  - scripts/compute_csli.py:_build_generation_provenance
  - modal_cs321m.py:_main_impl
  - modal_cs321m.py:run_pipeline
  - paper_exports/*.json (metadata.generation)
tags:
  - modal
  - provenance
  - deterministic-builds
  - container-pattern
  - audit-trail
  - env-var-injection
  - host-vs-container
---

# Host-injected provenance for tool-poor container runtimes

## Context

In Modal cloud runs of the CS321M audit pipeline, the three artifact producers (`csli.json`, `calibration.json`, `stopdff.json`) recorded `git_commit=""` in their `metadata.generation` provenance blocks despite the host having a valid commit. Root cause: Modal's `debian_slim` base image lacks the `git` binary, so the shared `_git_output(["rev-parse", "HEAD"])` helper in `scripts/_common.py` raised `FileNotFoundError` and returned `None`.

Because the audit-card integrity chain anchors every artifact to the commit that produced it, an empty `git_commit` silently defeated the provenance check exactly where it mattered most — long-running cloud runs whose freshness is harder to eyeball than a local invocation. Modal Runs 2 v4 and v5 shipped with the gap (Modal app `ap-LN6SqqBOa0BF2iR8JJGtCh` predecessor); Run 3 (commit `0f8da5d`) closed it.

The pattern generalizes: any container runtime that strips tools the host has but the producer expects to call as a subprocess — build tooling (`git`, `hg`, `svn`), system identity (`hostname`, `uname`, `dmidecode`), or anything else the host can compute cheaply at submit time.

## Guidance

Three-step host-to-container value injection. The host pre-computes the value, transports it as a function argument across the container boundary, broadcasts via `os.environ` to subprocess descendants, and the producer-side code prefers the env var over the live tool query so local and cloud paths share the same code.

**Step 1 — Host pre-computes at submit time** (`modal_cs321m.py:_main_impl`):

```python
host_git_commit = ""
try:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        check=False, capture_output=True, text=True,
    )
    if result.returncode == 0:
        host_git_commit = result.stdout.strip()
except OSError:
    pass

remote_result = run_pipeline.remote(
    stages, args.config, args.output_dir, smoke,
    budget_limit, initial_spend,
    host_git_commit=host_git_commit,
)
```

**Step 2 — Container entry-point sets the env var** (`modal_cs321m.py:run_pipeline`):

```python
def run_pipeline(..., host_git_commit: str = "") -> dict:
    if host_git_commit:
        os.environ["MODAL_HOST_GIT_COMMIT"] = host_git_commit
    # ...spawn stage subprocesses; they inherit os.environ
```

Setting `os.environ` in the container entry-point means every subprocess spawned by `run_pipeline` (the seven stage scripts in this case) inherits the value automatically. No need to thread it through CLI args of each stage.

**Step 3 — Producer prefers env var over live query** (`scripts/_common.py:build_generation_provenance`):

```python
host_commit_env = os.environ.get("MODAL_HOST_GIT_COMMIT")
git_commit = host_commit_env if host_commit_env else _git_output(["rev-parse", "HEAD"])
```

The symmetry matters: local-dev runs with `MODAL_HOST_GIT_COMMIT` unset still work because the live `git rev-parse HEAD` fallback fires. Cloud runs where the live query fails are now backed by the host's pre-computed value.

The pattern generalizes to any container runtime that serializes function arguments across the host/container boundary — Modal, Ray remote, Dask futures, Celery tasks with a base image that strips host tooling. The container-arg path is the transport; the env-var inside the container is the broadcast mechanism to subprocess descendants.

## Why This Matters

- **Deterministic-build provenance.** The artifact records the SHA of the code Modal actually executed (the host-submitted version), not whatever incidental VCS state existed in the container. The audit chain points at one well-defined SHA end-to-end.
- **Audit-trail integrity.** An empty `git_commit` field silently breaks every downstream consumer that anchors on it (manifest verification, re-execution scripts, paper-export validation). The failure was silent because empty-string is a valid type, not an exception — the provenance block looked complete.
- **Alternative considered — `image.apt_install("git")` in the Modal image** — works, but bloats the image, slows cold-start, increases attack surface, and ties provenance to whatever the container-installed `git` reports rather than the host's authoritative view. Env-var injection is smaller, faster, and matches the deterministic-build intent.
- **Symmetric fallback for local-dev.** Env-var-first with live-query fallback means no branching code path between local and cloud — the same producer works in both modes.
- **Contract pinned by regression tests.** `tests/test_pr14_review_regressions.py` covers all three steps: `test_build_generation_provenance_prefers_host_git_commit_env_var`, `test_build_generation_provenance_falls_back_to_live_git_when_no_env_var`, `test_compute_csli_build_generation_provenance_prefers_host_git_commit_env_var`, and `test_modal_run_pipeline_propagates_host_git_commit_env_var` (source-text contract on the orchestrator wiring).

## When to Apply

Apply this pattern when **all** of the following hold:

1. The container/sandbox base image lacks a tool the host has (e.g., Modal `debian_slim` minus `git`).
2. The host has authoritative access to the tool and the value it produces.
3. The value is small (a SHA, a hostname, a build number) and trivially serializable across the function-arg boundary.
4. The value is needed for provenance, audit logs, cache keys, or any record-keeping that must point back to a well-defined host-side artifact.
5. You want the same producer code to work in local-dev (where the tool is available) and cloud (where it isn't) without runtime branching.

**Do not apply** when the container actually needs the *tool*, not the *value* (e.g., the container must run `git checkout` itself). In that case, install the tool. **Do not apply** when the value is large enough to bloat the function-arg serialization payload — use a mounted volume or remote-fetch instead.

## Examples

### Example 1 — Modal `git_commit` (this session, commit `0f8da5d`)

Pre-fix (Modal Runs 2 v4 / v5): `csli.json`, `calibration.json`, `stopdff.json` all recorded `git_commit=""` because `debian_slim` has no `git`.

Post-fix (Modal Run 3, app `ap-LN6SqqBOa0BF2iR8JJGtCh`): all three recorded `git_commit=3ee38d636879`, matching the host's local HEAD at submit time. The fix is the exact 3-step wiring above; no Modal image changes were required.

### Example 2 — Hostname-style identity values

Same pattern for `socket.gethostname()` when the container hostname is the runtime sandbox ID (not meaningful for audit) but the host's machine name is the meaningful provenance:

```python
# Host side (submit-time):
host_machine = socket.gethostname()
run_pipeline.remote(..., host_machine=host_machine)

# Container entry-point:
if host_machine:
    os.environ["AUDIT_HOST_MACHINE"] = host_machine

# Producer side:
machine = os.environ.get("AUDIT_HOST_MACHINE") or socket.gethostname()
```

### Example 3 — Build SHA from a non-git VCS

If the host uses `hg` or `svn` (uncommon, but possible in legacy monorepos) and the container image only ships `git`, compute the host SHA via the appropriate tool and inject identically. The producer reads `os.environ.get("AUDIT_BUILD_SHA")` first, falls back to whatever VCS-detection logic it has, and the audit block records the host-authoritative value.

The unifying principle: **compute on the host where the tool exists, transport the value (not the tool) across the boundary, broadcast via `os.environ` to all subprocess descendants, and have the producer prefer the injected value over the live query so local and cloud paths share the same code.**

## Related

- [`docs/solutions/architecture-patterns/cryptographic-artifact-provenance-with-runtime-verification.md`](cryptographic-artifact-provenance-with-runtime-verification.md) — **Parent contract-layer pattern.** That doc defines the `metadata.generation` contract (`script_sha256`, `git_commit`, `git_dirty`, `git_status_relevant_paths`) and the aggregator-side SHA-match runtime verification rendered in `audit_card.md`. This doc fixes the *transport layer* for `git_commit` when the runtime container lacks the `git` binary. Update candidate after this learning lands: add a "Runtime environment caveat" sub-section pointing here and update the helper-snippet to show the env-var-first preference.
- [`docs/solutions/logic-errors/producer-emitted-flags-without-consumer-propagation.md`](../logic-errors/producer-emitted-flags-without-consumer-propagation.md) — Sibling in the audit-pipeline trust-contract family. Different axis (producer-vs-consumer semantic drift, not host-vs-container tool asymmetry).
- [`docs/solutions/logic-errors/scientific-metric-edge-case-guards.md`](../logic-errors/scientific-metric-edge-case-guards.md) — Sibling in the audit-pipeline trust-contract family. Producer-side metric-correctness guards (CSLI unique-qid coverage, Platt fallbacks, StopDFF reachability).
- Commits: [`0f8da5d`](../../..) (the env-var-injection wiring + the two new Codex-comment fixes it bundled), [`bb41819`](../../..) (artifact regen after producer SHA drift — the cryptographic-provenance safety net firing one commit late). PR #14 follow-up review thread.
- External pattern reference: deterministic-build principles in software supply chain (SLSA Level 3 provenance — host-truth provenance over container-truth). The pattern here is a specialization of "the artifact records the upstream source of the value, not the downstream computation environment."
