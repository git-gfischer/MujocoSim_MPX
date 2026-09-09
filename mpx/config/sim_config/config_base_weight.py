"""
Constant extra-mass load on the floating base for quadruped simulations.

The load is applied as a force ``F = extra_mass_kg * g`` [N], not by changing
MuJoCo ``body_mass``. Direction is either world-down (gravity) or along the
base body ``-z`` (perpendicular to the base, gravity-like when upright).

Typical use::

    from mpx.config.sim_config.config_base_weight import base_weight_config
    from mpx.utils.simulation_utils.base_weight import BaseWeightForce

    base_weight = BaseWeightForce.from_config(cfg=base_weight_config)
    # After RandomBaseForcePerturbation.tick_and_apply(data):
    base_weight.apply(data)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WeightDirection = Literal["world_down", "base_normal"]


@dataclass
class BaseWeightConfig:
    """Configuration for ``BaseWeightForce``."""

    enabled: bool = False

    # Extra payload mass [kg]; force magnitude is ``extra_mass_kg * g``.
    extra_mass_kg: float = 10.0

    # Gravity magnitude used to convert mass to force [m/s^2].
    g: float = 9.81

    # ``world_down``: inertial -Z. ``base_normal``: body -z of the floating base.
    direction: WeightDirection = "base_normal"


# Default profile used by examples; reassign or construct ``BaseWeightConfig(...)`` to tune.
base_weight_config = BaseWeightConfig()
