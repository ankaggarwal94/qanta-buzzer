from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid for scalar confidence proxies."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)
