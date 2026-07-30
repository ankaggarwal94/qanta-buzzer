"""StopDFF v5 standalone evidentiary pipeline (profile stopdff_bucketed_dp_paired_v2).

Implements the normative v5 contracts:
  - SCIENTIFIC_CONTRACT.md      (Bellman policy, calibrators, continuation, FVI, bootstrap, verdicts)
  - IDENTITY_AND_ARTIFACT_CONTRACT.md (canonical JSON identities + manifests)
  - ACCEPTANCE_CONTRACT.md      (standalone checker + negative mutation suite)

This package is self-contained and deterministic. Real-data adapter scoring is the
only step that requires the sentence-transformers model; all DP / FVI / bootstrap /
verdict / checker logic is pure and unit-testable on fixtures.
"""

PROFILE_NAME = "stopdff_bucketed_dp_paired_v2"
METRIC_FAMILY = "empirical_bucketed_finite_horizon_stopdff"
SCHEMA_VERSION = 2
PROTOCOL_VERSION = "v5"
