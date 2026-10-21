"""Foot force from the leg Jacobian and the joint torque.

The solve matches ``physics_features.analytic_grf`` with ``sign=-1``, then
negates that result so a standing foot has negative world Z, the same sign
as ``estimate_foot_grf``. Returning ``solve(A, J @ tau)`` is that negation.
Do not negate again.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9
_FOOT_NAMES = ("FL", "FR", "RL", "RR")


def jacobian_foot_force(tau, J, tau_bias=None, damping=1e-2):
    """World force (..., 3) from torque (..., 3) and Jacobian (..., 3, 3)."""
    tau = np.asarray(tau, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    if tau_bias is not None:
        tau = tau - np.asarray(tau_bias, dtype=np.float64)
    jjt = J @ np.swapaxes(J, -1, -2)
    scale = np.maximum(np.trace(jjt, axis1=-2, axis2=-1), _EPS)
    regularized = jjt + damping * np.eye(3) * scale[..., None, None]
    return np.linalg.solve(regularized, J @ tau[..., None])[..., 0]


def format_jacobian_grf(solver, estimated) -> str:
    """``Jacobian GRF FL=(sx sy sz)/(jx jy jz) ...`` solver triple, then estimate."""
    solver = np.asarray(solver, dtype=np.float64).reshape(-1, 3)
    estimated = np.asarray(estimated, dtype=np.float64).reshape(-1, 3)
    parts = []
    n = min(solver.shape[0], estimated.shape[0])
    for i in range(n):
        name = _FOOT_NAMES[i] if i < len(_FOOT_NAMES) else str(i)
        s = solver[i]
        e = estimated[i]
        parts.append(
            f"{name}=({s[0]:.1f} {s[1]:.1f} {s[2]:.1f})/"
            f"({e[0]:.1f} {e[1]:.1f} {e[2]:.1f})"
        )
    return "Jacobian GRF " + " ".join(parts)
