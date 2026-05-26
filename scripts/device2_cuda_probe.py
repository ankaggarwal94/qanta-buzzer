"""Device 2 (RTX 5090) CUDA fallback probe. Run on Windows 11 to verify GPU readiness.

No qanta-buzzer deps required -- only torch. Outputs JSON to stdout; exits 1 on failure.
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

report = {
    "device_name": torch.cuda.get_device_name(0),
    "cuda_version": torch.version.cuda,
    "torch_version": torch.__version__,
    "memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "ready",
}

print(json.dumps(report, indent=2))
sys.exit(0)
