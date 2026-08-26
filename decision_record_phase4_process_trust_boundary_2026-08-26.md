# Decision record — Phase-4 process/host trust boundary (2026-08-26)

**Disposition:** Author-approved amendment for the fastest defensible
camera-ready path. This records an explicit claim boundary; it is not a claim
that hostile-process isolation was implemented.

## Decision

The Phase-4 R-081 launcher is not a sandbox, privilege boundary, or
hostile-process containment mechanism. It is an integrity and reproducibility
workflow. The
certified producer and dependencies, OS and filesystem, and processes sharing
the launcher's OS identity are trusted to cooperate. A conforming run has no
surviving producer descendants with access to its launch workspace after the
direct producer exits.

The launcher continues to treat artifact bytes and paths adversarially within
that supported workflow: malformed data, symlink/reparse traversal, ordinary
replacement or mutation races, stale handles, destination races, partial
writes, durability failures, and post-commit cleanup defects all retain their
existing fail-closed or truthful-STOP behavior. "Private promotion" means a
path-detached launcher-owned tree reconstructed from comparator-approved
bytes; it does not mean ACL, principal, or process isolation.

The closed launch-receipt field is:

```text
process_trust_model = trusted_same_os_identity_no_surviving_descendants_v1
```

The D7(b) driver rejects a missing or different value. `PASS`, the launch
receipt, `PASS_RELEASE`, and `CAMERA_READY_CLOSURE` certify the declared bytes
and scientific bindings only under this boundary. They do not establish
provenance or tamper resistance against a deliberately hostile process with
the same OS access token or against privileged host compromise.

## Operational consequence

The certified ceremony may run only in an operator-controlled account/session
with no untrusted same-identity writer and no surviving producer descendants.
If that precondition cannot be established, the ceremony must stop until a
separately approved, pinned, and tested OS isolation backend is available.
Implementing Windows Job Objects, cgroups, a separate OS principal, or a VM is
future hardening and is not silently implied by this amendment.

## Scope

This decision amends R-081 only. It changes no estimator, model, input,
comparator, record, D7(b) inference procedure, manuscript result, or existing
byte-integrity gate.
