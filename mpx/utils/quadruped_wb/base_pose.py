"""Base pose randomization and desired-pose visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from gym_quadruped.utils.mujoco.visual import render_vector

from mpx.utils.sim_utils import _quat_mul_wxyz, _quat_from_rpy_wxyz
from mpx.config.sim_config.base_pose_randomizer_config import BasePoseRandomizationConfig

class BasePoseRandomizer:
    """Apply randomized base height and orientation for full-contact balance setups."""

    def __init__(
        self,
        *,
        nominal_p0: np.ndarray,
        nominal_quat0: np.ndarray,
        cfg: BasePoseRandomizationConfig | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.nominal_p0 = np.asarray(nominal_p0, dtype=np.float64).reshape(3).copy()
        self.nominal_quat0 = np.asarray(nominal_quat0, dtype=np.float64).reshape(4).copy()
        self.cfg = cfg or BasePoseRandomizationConfig()
        self.rng = np.random.default_rng(rng_seed)

    @staticmethod
    def _is_four_leg_balance_mode(robot_cfg: Any) -> bool:
        if getattr(robot_cfg, "behaviour", None) != "balance":
            return False
        if not bool(getattr(robot_cfg, "use_balance_fixed_contact", False)):
            return False
        mask = np.asarray(getattr(robot_cfg, "balance_fixed_contact_mask", []), dtype=np.float64).reshape(-1)
        return mask.shape[0] == 4 and np.all(mask > 0.5)

    def sample(self, robot_cfg: Any) -> tuple[np.ndarray, np.ndarray]:
        p0 = self.nominal_p0.copy()
        quat0 = self.nominal_quat0.copy()

        if not self.cfg.enabled or not self._is_four_leg_balance_mode(robot_cfg):
            return p0, quat0

        p0[2] = max(
            float(self.cfg.min_base_height),
            p0[2] + self.rng.uniform(*self.cfg.z_offset_range),
        )
        roll = np.deg2rad(self.rng.uniform(*self.cfg.roll_range_deg))
        pitch = np.deg2rad(self.rng.uniform(*self.cfg.pitch_range_deg))
        yaw = np.deg2rad(self.rng.uniform(*self.cfg.yaw_range_deg))
        dq = _quat_from_rpy_wxyz(roll, pitch, yaw)
        quat0 = _quat_mul_wxyz(dq, quat0)
        quat0 /= np.linalg.norm(quat0)
        return p0, quat0

    def apply_to_config(self, robot_cfg: Any) -> None:
        p0, quat0 = self.sample(robot_cfg)
        robot_cfg.p0 = jnp.asarray(p0)
        robot_cfg.quat0 = jnp.asarray(quat0)


@dataclass(frozen=True)
class DesiredBasePoseVisualizationConfig:
    """Viewer styling for desired base orientation/height markers."""

    enabled: bool = True
    axis_scale: float = 0.20
    axis_alpha: float = 0.95
    height_offset_xy: tuple[float, float] = (0.28, 0.0)
    height_alpha: float = 0.85


class DesiredBasePoseVisualizer:
    """Draw desired base-frame axes and desired world-frame height in passive viewer."""

    def __init__(self, cfg: DesiredBasePoseVisualizationConfig | None = None) -> None:
        self.cfg = cfg or DesiredBasePoseVisualizationConfig()
        self._axis_ids = [-1, -1, -1]
        self._height_id = -1

    @staticmethod
    def _quat_to_rotmat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        q /= max(np.linalg.norm(q), 1e-12)
        w, x, y, z = q
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def update(self, viewer, desired_pos_xyz: np.ndarray, desired_quat_wxyz: np.ndarray) -> None:
        if viewer is None or not self.cfg.enabled:
            return

        p = np.asarray(desired_pos_xyz, dtype=np.float64).reshape(3)
        rot = self._quat_to_rotmat_wxyz(desired_quat_wxyz)
        axis_colors = (
            np.array([1.0, 0.15, 0.15, self.cfg.axis_alpha], dtype=np.float64),  # X
            np.array([0.15, 1.0, 0.15, self.cfg.axis_alpha], dtype=np.float64),  # Y
            np.array([0.2, 0.45, 1.0, self.cfg.axis_alpha], dtype=np.float64),   # Z
        )

        for i in range(3):
            self._axis_ids[i] = render_vector(
                viewer,
                rot[:, i],
                pos=p,
                scale=self.cfg.axis_scale,
                color=axis_colors[i],
                geom_id=self._axis_ids[i],
            )

        dx, dy = self.cfg.height_offset_xy
        height_origin = np.array([p[0] + dx, p[1] + dy, 0.0], dtype=np.float64)
        height_vec = np.array([0.0, 0.0, max(0.0, float(p[2]))], dtype=np.float64)
        self._height_id = render_vector(
            viewer,
            height_vec,
            pos=height_origin,
            scale=1.0,
            color=np.array([1.0, 0.95, 0.2, self.cfg.height_alpha], dtype=np.float64),
            geom_id=self._height_id,
        )
