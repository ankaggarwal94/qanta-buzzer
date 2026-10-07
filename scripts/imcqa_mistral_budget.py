"""Create-once cumulative reservation for the explicitly approved Mistral run.

The reservation is an engineering bound on configured resources and durations,
not a provider invoice cap: image builds, crashes and cancellation are external.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

APPROVAL_ID = "imcqa-mistral-20261007-eight-usd-v1"
CEILING = Decimal("8.00")
GPU_RATE = Decimal("0.00063924")
CPU_RATE = Decimal("0.00004396")
STAGES = {"development": {"questions": 140, "contexts": 5600, "timeout": 1860},
          "evaluation": {"questions": 850, "contexts": 34000, "timeout": 8250}}
CACHE_TIMEOUT = 1200
STARTUP_TIMEOUT = 90
SCALEDOWN = 2


def budget_plan(rate_verified_utc: str, *, now=None) -> dict:
    """Reserve every permitted attempt up front without refunding failed work."""
    now = now or datetime.now(timezone.utc)
    verified = datetime.fromisoformat(rate_verified_utc.replace("Z", "+00:00"))
    if verified.tzinfo is None or not 0 <= (now - verified).total_seconds() <= 86400:
        raise ValueError("official resource rates require an attestation within 24 hours")
    gpu = {name: str(GPU_RATE * (row["timeout"] + STARTUP_TIMEOUT + SCALEDOWN)
                     + Decimal("0.20")) for name, row in STAGES.items()}
    cpu = CPU_RATE * (CACHE_TIMEOUT + STARTUP_TIMEOUT + SCALEDOWN)
    total = sum(map(Decimal, gpu.values())) + cpu + Decimal("0.40") + Decimal("0.20")
    if total > CEILING:
        raise ValueError("cumulative reservation exceeds approved ceiling")
    return {
        "schema_version": "imcqa-mistral-budget-v1", "approval_id": APPROVAL_ID,
        "ceiling_usd": str(CEILING), "rate_verified_utc": rate_verified_utc,
        "pricing_source": "https://modal.com/pricing",
        "gpu_rate_usd_per_second": str(GPU_RATE), "cache_cpu_rate_usd_per_second": str(CPU_RATE),
        "gpu_resource_maximums": {"gpu": "L40S", "cpu": [2, 2], "memory_mib": [32768, 32768]},
        "cache_resource_maximums": {"cpu": [2, 2], "memory_mib": [8192, 8192]},
        "gpu_stage_reservations_usd": gpu, "cpu_cache_reservation_usd": str(cpu),
        "image_builds_allowed": 0, "cancellation_provider_contingency_usd": "0.40",
        "storage_egress_contingency_usd": "0.20",
        "reserved_estimate_usd": str(total), "unallocated_headroom_usd": str(CEILING - total),
        "stages": {name: dict(value) for name, value in STAGES.items()}, "cache_timeout_seconds": CACHE_TIMEOUT,
        "startup_timeout_seconds": STARTUP_TIMEOUT, "scaledown_seconds": SCALEDOWN,
        "automatic_function_retries": 0, "maximum_gpu_calls": 2, "maximum_cpu_cache_calls": 1,
        "refund_policy": "none: failed, interrupted and ambiguous attempts keep their full reservation",
        "invoice_verified": False, "provider_enforced_spend_cap": False,
        "external_limitations": [
            "Provider container-crash rescheduling can occur even with retries=0; worker claims prevent scoring replay after claiming.",
            "No image builds are allowed; cancellation latency and provider limit enforcement are external assumptions.",
            "Cache deletion is best-effort; an interrupted cleanup may leave storage charges requiring manual cleanup.",
        ],
    }


def validate_budget(plan: dict, *, now=None, require_fresh=True) -> None:
    """Reject changed ceilings, omitted reservations and stale launch attestations."""
    if not require_fresh:
        now = datetime.fromisoformat(plan["rate_verified_utc"].replace("Z", "+00:00"))
    if plan != budget_plan(plan["rate_verified_utc"], now=now):
        raise ValueError("budget differs from the immutable approved reservation")
