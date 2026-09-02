"""
Random pulsed external forces on a floating base via ``mj_data.qfrc_applied``.

Functionality
-------------
- Samples random force orientation uniformly on the unit sphere.
- Samples random force magnitude in ``force_magnitude_range`` [N].
- Applies each sampled force as a constant pulse for a random duration.
- Waits a random cooldown between pulses.
- Writes force in world frame to ``qfrc_applied[:3]`` and clears base moments
  ``qfrc_applied[3:6]``.
- Reads default parameters from
  ``mpx.config.sim_config.config_ext_base_forces.ext_base_force_config``.

Example (usage in another simulation loop)
------------------------------------------
```python
import numpy as np
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation

# Build scheduler from config defaults
base_force = RandomBaseForcePerturbation.from_config(sim_dt=1.0 / sim_frequency)

def sim_step_with_disturbance(action):
    # Apply/clear qfrc_applied before MuJoCo step
    base_force.tick_and_apply(env.mjData)
    return env.step(action=action)

# Optional: restart pulse schedule after an episode reset / respawn
base_force.reset()

while env.viewer.is_running():
    obs, rew, terminated, truncated, info = sim_step_with_disturbance(tau_cmd)
    if base_force.is_active:
        # Optional debug/logging
        applied_force_world = base_force.force.copy()
```
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from mpx.config.sim_config.config_ext_base_forces import (
    ExtBaseForceConfig,
    ext_base_force_config,
)


def _sample_unit_direction(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


@dataclass
class RandomBaseForcePerturbation:
    """
    Apply a constant world-frame force on the base for a random duration, then wait
    a random cooldown before the next pulse.

    Force direction is uniform on the unit sphere; magnitude is uniform in
    ``force_magnitude_range`` [N].
    """

    rng: np.random.Generator
    sim_dt: float
    force_magnitude_range: tuple[float, float] = ext_base_force_config.force_magnitude_range
    duration_range_s: tuple[float, float] = ext_base_force_config.duration_range_s
    cooldown_range_s: tuple[float, float] = ext_base_force_config.cooldown_range_s
    enabled: bool = ext_base_force_config.enabled
    start_after_cooldown: bool = ext_base_force_config.start_after_cooldown
    _active: bool = field(default=False, init=False, repr=False)
    _steps_left: int = field(default=0, init=False, repr=False)
    _cooldown_left: int = field(default=0, init=False, repr=False)
    force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    @classmethod
    def from_config(
        cls,
        sim_dt: float,
        cfg: ExtBaseForceConfig = ext_base_force_config,
    ) -> RandomBaseForcePerturbation:
        """
        Build a perturbation scheduler from :class:`ExtBaseForceConfig`.

        The returned instance is reset immediately using
        ``start_after_cooldown=cfg.start_after_cooldown``.
        """
        pert = cls(
            rng=np.random.default_rng(cfg.rng_seed),
            sim_dt=sim_dt,
            force_magnitude_range=cfg.force_magnitude_range,
            duration_range_s=cfg.duration_range_s,
            cooldown_range_s=cfg.cooldown_range_s,
            enabled=cfg.enabled,
            start_after_cooldown=cfg.start_after_cooldown,
        )
        pert.reset()
        return pert

    def reset(self, *, start_after_cooldown: bool | None = None) -> None:
        """Clear applied force; optionally schedule the first pulse after a random wait."""
        if start_after_cooldown is None:
            start_after_cooldown = self.start_after_cooldown
        self._active = False
        self.force = np.zeros(3, dtype=np.float64)
        self._steps_left = 0
        if not self.enabled:
            self._cooldown_left = 0
            return
        if start_after_cooldown:
            self._cooldown_left = self._sample_cooldown_steps()
        else:
            self._cooldown_left = 0

    def _seconds_to_steps(self, t_s: float) -> int:
        return max(1, int(round(t_s / self.sim_dt)))

    def _sample_cooldown_steps(self) -> int:
        lo, hi = self.cooldown_range_s
        return self._seconds_to_steps(float(self.rng.uniform(lo, hi)))

    def _begin_pulse(self) -> None:
        lo, hi = self.force_magnitude_range
        magnitude = float(self.rng.uniform(lo, hi))
        direction = _sample_unit_direction(self.rng)
        self.force = (magnitude * direction).astype(np.float64)
        d_lo, d_hi = self.duration_range_s
        self._steps_left = self._seconds_to_steps(float(self.rng.uniform(d_lo, d_hi)))
        self._active = True

    def tick_and_apply(self, mj_data) -> None:
        """Advance the pulse scheduler and write ``mj_data.qfrc_applied[:3]`` (world frame)."""
        mj_data.qfrc_applied[3:6] = 0.0
        if not self.enabled:
            mj_data.qfrc_applied[:3] = 0.0
            self.force = np.zeros(3, dtype=np.float64)
            return

        if self._active:
            mj_data.qfrc_applied[:3] = self.force
            self._steps_left -= 1
            if self._steps_left <= 0:
                self._active = False
                self.force = np.zeros(3, dtype=np.float64)
                mj_data.qfrc_applied[:3] = 0.0
                self._cooldown_left = self._sample_cooldown_steps()
            return

        mj_data.qfrc_applied[:3] = 0.0
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return
        self._begin_pulse()
        mj_data.qfrc_applied[:3] = self.force

    @property
    def is_active(self) -> bool:
        return self._active and self.enabled
