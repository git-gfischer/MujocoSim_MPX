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

    # Realised median base speed in v3 was only 0.26 m/s against a [0.25, 0.70]
    # range. Widened, and the sign is handled by the navigator so reverse and
    # lateral motion are covered too.
    # Measured: at 1.0 m/s this MPC lost the robot in 6 of 10 episodes, and the
    # falls were what pushed joints past their limits. 0.80 keeps the coverage
    # Task 9d asks for while leaving the controller able to track the command.
    max_speed: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.20, high=0.80)
    max_yaw_rate: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.40, high=1.50)

    step_freq: FloatRangeSpec = FloatRangeSpec(enabled=True, low=1.00, high=1.70)
    duty_factor: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.55, high=0.80)

    # Contact time constant [s]. Clamped well below the v3 [0.012, 0.035]: the
    # upper end there was 6.6 sim steps at 200 Hz and made the foot bouncy, which
    # is what produced the one-frame contact dropouts. Must stay >= 2 * timestep
    # (0.004 s at the 500 Hz default).
    # Measured on the v4 pilot: of the eight knobs, solref_timeconst separated
    # falls from survivals most strongly (Cohen's d -0.92), with the stiff end
    # losing the robot — a stiffer contact means harder impacts for the MPC to
    # absorb. 0.010-0.014 keeps 5-7 substeps at 500 Hz, stays far tighter than
    # the v3 [0.012, 0.035], and the contact-quality margin (0.75% one-step runs
    # against a 2% limit) covers the small loss of stiffness.
    solref_timeconst: FloatRangeSpec = FloatRangeSpec(
        enabled=True, low=0.010, high=0.014, log_uniform=True
    )

    # Sliding friction μ on foot geoms (geom_friction[:, 0]). Torsional/rolling stay as in XML.
    # MuJoCo contact μ is min(foot, floor); the flat floor is 2.0.
    # Floor raised to the nominal XML value of 1.2 for the standard folders: at
    # μ < 0.3 the fall rate reached 38% of episodes, which wastes wall-clock and
    # skews episode lengths. Genuinely slippery conditions belong in a dedicated
    # `slippery` folder with its own low-friction range, not mixed into nominal
    # locomotion (DATASET_FIX_TASKS_R2, Task R2-8 item 3).
    friction: FloatRangeSpec = FloatRangeSpec(enabled=True, low=1.2, high=2.00)

    # Commanded base height [m]. The v3 crouch of ~0.21 m left about 5 cm of swing
    # clearance and contributed to the micro-bouncing. Kept close to the 0.27 m
    # nominal: commanding 0.32 m against a 0.27 m spawn is a 5 cm step the MPC
    # must absorb at episode start, and it cost episodes to falls.
    base_height: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.26, high=0.30)


# Loco profile follows the dataclass master switch. Balance stays off until enabled here.
loco_reset_randomization_config = ResetRandomizationConfig()

balance_reset_randomization_config = ResetRandomizationConfig(
    enabled=False,
    max_speed=FloatRangeSpec(enabled=False, low=0.25, high=0.70),
    max_yaw_rate=FloatRangeSpec(enabled=False, low=0.40, high=1.20),
    step_freq=FloatRangeSpec(enabled=False, low=1.00, high=1.70),
    duty_factor=FloatRangeSpec(enabled=False, low=0.55, high=0.80),
)