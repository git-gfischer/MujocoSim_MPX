"""Per-leg Jacobian foot-force estimate, J^T f = tau.

Shown next to the MuJoCo solver force. Not a floating-base inverse-dynamics solve.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FootForceJacobianConfig:
    """Knobs for ``estimate_foot_force_jacobian``."""

    # False: return zeros, skip the Jacobian, print nothing, draw nothing.
    enabled: bool = True

    # Print in the quiet balance loops. Locomotion uses its own contact-print gate.
    echo: bool = False

    # Relative damping on J J^T, same default as analytic_grf.
    damping: float = 1e-2

    # Subtract qfrc_bias on the three leg joints before the solve.
    subtract_bias: bool = True


foot_force_jacobian_config = FootForceJacobianConfig()
