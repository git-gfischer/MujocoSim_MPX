"""
Constant extra-mass force on a floating base via ``mj_data.qfrc_applied``.

Functionality
-------------
- Converts ``extra_mass_kg`` to a force ``F = m * g`` [N].
- ``world_down``: applies ``[0, 0, -F]`` in the inertial frame.
- ``base_normal``: applies ``-F * body_z`` (world), so the load stays
  perpendicular to the base and matches gravity when the robot is upright.
- **Adds** to ``qfrc_applied[:3]`` so random base-force pulses are preserved.
  Call after ``RandomBaseForcePerturbation.tick_and_apply``.

Reads defaults from
``mpx.config.sim_config.config_base_weight.base_weight_config``.

Example
-------
```python
from mpx.config.sim_config.config_base_weight import base_weight_config
from mpx.utils.simulation_utils.base_weight import BaseWeightForce

base_weight = BaseWeightForce.from_config(cfg=base_weight_config)

def sim_step_with_load(action):
    base_force.tick_and_apply(env.mjData)
    base_weight.apply(env.mjData)
    return env.step(action=action)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from mpx.config.sim_config.config_base_weight import (
    BaseWeightConfig,
    WeightDirection,
    base_weight_config,
)

_WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_BODY_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _rotate_body_vec_to_world(quat_wxyz: np.ndarray, vec_body: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector from the floating-base body frame into the world frame."""
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    vec = np.asarray(vec_body, dtype=np.float64).reshape(3)
    out = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(out, vec, quat)
    return out


@dataclass
class BaseWeightForce:
    """Constant extra-mass force on ``qfrc_applied[:3]`` (world frame)."""

    enabled: bool = base_weight_config.enabled
    extra_mass_kg: float = base_weight_config.extra_mass_kg
    g: float = base_weight_config.g
    direction: WeightDirection = base_weight_config.direction
    force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    @classmethod
    def from_config(
        cls,
        cfg: BaseWeightConfig = base_weight_config,
    ) -> BaseWeightForce:
        """Build an applicator from :class:`BaseWeightConfig`."""
        return cls(
            enabled=cfg.enabled,
            extra_mass_kg=cfg.extra_mass_kg,
            g=cfg.g,
            direction=cfg.direction,
        )

    @property
    def magnitude(self) -> float:
        return float(self.extra_mass_kg) * float(self.g)

    def world_force(self, mj_data) -> np.ndarray:
        """World-frame force [N] for the current base orientation (zeros if disabled)."""
        mag = self.magnitude
        if not self.enabled or mag <= 0.0:
            return np.zeros(3, dtype=np.float64)
        if self.direction == "base_normal":
            direction = _rotate_body_vec_to_world(mj_data.qpos[3:7], _BODY_DOWN)
            n = float(np.linalg.norm(direction))
            if n < 1e-12:
                direction = _WORLD_DOWN
            else:
                direction = direction / n
            return (mag * direction).astype(np.float64)
        return (mag * _WORLD_DOWN).astype(np.float64)

    def apply(self, mj_data) -> None:
        """Add the extra-mass force to ``mj_data.qfrc_applied[:3]``."""
        self.force = self.world_force(mj_data)
        if not self.enabled:
            return
        mj_data.qfrc_applied[:3] += self.force
