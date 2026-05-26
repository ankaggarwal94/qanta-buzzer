"""Device 2 (RTX 5090) CUDA fallback probe. Run on Windows 11 to verify GPU readiness.

No qanta-buzzer deps required -- only torch. Outputs JSON to stdout; exits 1 on failure.
The probe only reports ready after a real CUDA allocation, kernel, and synchronization succeed.

By default the probe exercises ``cuda:0``. On hosts with multiple visible GPUs, pass
``--device-index N`` (or set ``CUDA_VISIBLE_DEVICES`` upstream) so the readiness check
actually targets the intended device -- otherwise a failing Device 2 can be masked by a
healthy Device 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a specific CUDA device for readiness. Defaults to cuda:0; "
            "pass --device-index to target a different GPU on multi-GPU hosts."
        ),
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help=(
            "Zero-based CUDA device index to probe (default: 0). Must be less "
            "than torch.cuda.device_count() at runtime."
        ),
    )
    return parser.parse_args(argv)


args = _parse_args()
device_index = args.device_index

try:
    import torch
except ImportError:
    print(
        json.dumps({
            "status": "error",
            "error": "torch not installed",
            "device_index": device_index,
            "hint": "pip install torch --index-url https://download.pytorch.org/whl/cu124",
        }),
        file=sys.stderr,
    )
    sys.exit(1)

if not torch.cuda.is_available():
    print(
        json.dumps({
            "status": "no_cuda",
            "torch_version": torch.__version__,
            "cuda_built": torch.version.cuda or "none",
            "device_index": device_index,
            "hint": "CUDA not available. Ensure NVIDIA drivers and CUDA toolkit are installed.",
        }),
        file=sys.stderr,
    )
    sys.exit(1)

device_count = torch.cuda.device_count()
if device_index < 0 or device_index >= device_count:
    print(
        json.dumps({
            "status": "invalid_device_index",
            "torch_version": torch.__version__,
            "cuda_built": torch.version.cuda or "none",
            "device_index": device_index,
            "device_count": device_count,
            "hint": (
                f"--device-index {device_index} is out of range; "
                f"torch.cuda.device_count() reports {device_count}. "
                "Pass a value in [0, device_count) or adjust CUDA_VISIBLE_DEVICES."
            ),
        }),
        file=sys.stderr,
    )
    sys.exit(1)

try:
    device = torch.device(f"cuda:{device_index}")
    lhs = torch.ones((64, 64), device=device)
    rhs = torch.eye(64, device=device)
    result = lhs @ rhs
    checksum = float(result.sum().item())
    torch.cuda.synchronize(device)
except RuntimeError as exc:
    print(
        json.dumps({
            "status": "cuda_runtime_error",
            "torch_version": torch.__version__,
            "cuda_built": torch.version.cuda or "none",
            "device_index": device_index,
            "error": str(exc),
            "hint": "CUDA enumerates but a tensor operation failed; check driver/runtime/PyTorch compatibility.",
        }),
        file=sys.stderr,
    )
    sys.exit(1)

props = torch.cuda.get_device_properties(device_index)
report = {
    "device_index": device_index,
    "device_count": device_count,
    "device_name": torch.cuda.get_device_name(device_index),
    "cuda_version": torch.version.cuda,
    "torch_version": torch.__version__,
    "memory_total_gib": round(props.total_memory / (1024 ** 3), 1),
    "kernel_check": "passed",
    "kernel_checksum": checksum,
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "ready",
}

print(json.dumps(report, indent=2))
sys.exit(0)
