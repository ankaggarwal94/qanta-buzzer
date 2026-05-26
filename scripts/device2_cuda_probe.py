"""Device 2 (RTX 5090) CUDA fallback probe. Run on Windows 11 to verify GPU readiness.

No qanta-buzzer deps required -- only torch. Outputs JSON to stdout; exits 1 on failure.
The probe only reports ready after a real CUDA allocation, kernel, and synchronization succeed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

try:
    import torch
except ImportError:
    print(
        json.dumps({
            "status": "error",
            "error": "torch not installed",
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
            "hint": "CUDA not available. Ensure NVIDIA drivers and CUDA toolkit are installed.",
        }),
        file=sys.stderr,
    )
    sys.exit(1)

try:
    device = torch.device("cuda:0")
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
            "error": str(exc),
            "hint": "CUDA enumerates but a tensor operation failed; check driver/runtime/PyTorch compatibility.",
        }),
        file=sys.stderr,
    )
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
report = {
    "device_name": torch.cuda.get_device_name(0),
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
