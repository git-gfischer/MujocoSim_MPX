"""
Reset-time domain randomization ranges for quadruped simulations.

Typical use::

    from mpx.config.sim_config.config_reset_randomization import (
        loco_reset_randomization_config,
    )
    from mpx.utils.simulation_utils.reset_randomizer import ResetRandomizer

    randomizer = ResetRandomizer.from_config(loco_reset_randomization_config)
    # In _respawn(), after reset_mpc(...):
    # sample, mpc_data = randomizer.sample_and_apply(targets)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloatRangeSpec:
    """Inclusive sampling range for one reset knob."""

    enabled: bool = True
    low: float = 0.0
    high: float = 1.0
    log_uniform: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(
                f"FloatRangeSpec low ({self.low}) must be <= high ({self.high})"
            )
        if self.log_uniform and self.low <= 0.0:
            raise ValueError("log_uniform ranges require low > 0")


@dataclass
class ResetRandomizationConfig:
    """Master switch plus per-knob ranges for :class:`ResetRandomizer`."""

    enabled: bool = True
    rng_seed: int | None = None

    payload: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.0, high=3.0)
    max_speed: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.25, high=0.70)
    max_yaw_rate: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.40, high=1.20)
    step_freq: FloatRangeSpec = FloatRangeSpec(enabled=True, low=1.00, high=1.70)
    duty_factor: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.55, high=0.80)
    solref_timeconst: FloatRangeSpec = FloatRangeSpec(
        enabled=True, low=0.012, high=0.035, log_uniform=True
    )
    # Sliding friction μ on foot geoms (geom_friction[:, 0]). Torsional/rolling stay as in XML.
    # Nominal XML foot μ is 1.2; this range stays at or below that so episodes are
    # same-or-slipperier. MuJoCo contact μ is min(foot, floor); floor is 2.0 on flat.
    friction: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.20, high=1.00)


# Loco profile follows the dataclass master switch. Balance stays off until enabled here.
loco_reset_randomization_config = ResetRandomizationConfig()

balance_reset_randomization_config = ResetRandomizationConfig(
    enabled=False,
    max_speed=FloatRangeSpec(enabled=False, low=0.25, high=0.70),
    max_yaw_rate=FloatRangeSpec(enabled=False, low=0.40, high=1.20),
    step_freq=FloatRangeSpec(enabled=False, low=1.00, high=1.70),
    duty_factor=FloatRangeSpec(enabled=False, low=0.55, high=0.80),
)