"""Non-subsampled Gaussian RDP accounting used by the reference mechanism."""

import math

import numpy as np


def accountant(events: int, z: float, delta: float):
    """Return minimum converted epsilon and the evaluated RDP order grid."""
    if events < 1:
        raise ValueError("events must be positive")
    if z <= 0:
        raise ValueError("z must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    rows = []
    best = None
    orders = np.concatenate((np.arange(1.1, 10, 0.1), np.arange(10, 65, 0.25)))
    for alpha in orders:
        rdp = events * alpha / (2 * z * z)
        conversion = math.log(1 / delta) / (alpha - 1)
        epsilon = rdp + conversion
        row = {"order": float(alpha), "rdp": float(rdp),
               "conversion": float(conversion), "epsilon": float(epsilon)}
        rows.append(row)
        if best is None or epsilon < best["epsilon"]:
            best = row
    return best, rows


def calibrate(events: int, target: float, delta: float) -> float:
    """Calibrate normalized noise on the fixed RDP order grid."""
    if target <= 0:
        raise ValueError("target must be positive")
    low, high = 0.01, 1000.0
    for _ in range(100):
        midpoint = (low + high) / 2
        best, _ = accountant(events, midpoint, delta)
        if best["epsilon"] > target:
            low = midpoint
        else:
            high = midpoint
    return high
