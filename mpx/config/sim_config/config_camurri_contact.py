"""
Camurri et al., RA-L 2017, Section IV — logistic stance probability.

The paper fits ``beta`` and ``beta0`` by maximum likelihood and does not
publish the numbers. These defaults put P = 0.5 at 15 N of vertical GRF
and P(0) = 0.05, so a Go2 foot crosses the paper's 0.5 decision at the
same load the Schmitt label uses for touchdown.

Typical use::

    from mpx.config.sim_config.config_camurri_contact import camurri_contact_config
    camurri_contact_config.beta = 0.2
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# P(f_z = 0) and the force where P = 0.5. beta, beta0 are the paper's weights.
_P_AT_ZERO = 0.05
_F_HALF_N = 15.0
_BETA0 = math.log(_P_AT_ZERO / (1.0 - _P_AT_ZERO))
_BETA = -_BETA0 / _F_HALF_N


@dataclass
class CamurriContactConfig:
    """Weights for ``P(S=1 | f) = 1 / (1 + exp(-beta * f - beta0))``."""

    # If False, the estimator returns zeros and the simulators skip the solve.
    enabled: bool = False

    # Print the stance on the balance loops, which otherwise stay quiet.
    # Sims that already print MuJoCo contact print Camurri beside it.
    echo: bool = False

    # Logistic-regression weights. Positive beta: more upward force, higher P.
    beta: float = _BETA
    beta0: float = _BETA0

    # Paper: a foot is in the contact set C when this probability is reached.
    threshold: float = 0.5

    # Same separation gate as ``estimate_foot_grf``.
    dist_threshold: float = 0.005

    # "world_z" is the paper's f_z. "contact_normal" uses the terrain normal
    # so a loaded foot on a slope still counts.
    force_axis: str = "world_z"

    def __post_init__(self) -> None:
        if self.force_axis not in ("world_z", "contact_normal"):
            raise ValueError(
                f"force_axis must be 'world_z' or 'contact_normal', got {self.force_axis!r}"
            )
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")


camurri_contact_config = CamurriContactConfig()
