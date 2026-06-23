"""Desired balance pose sampler for quadruped balance control.

Samples a random target height and body yaw on each reset. The returned
quaternion (wxyz) is a pure yaw rotation — no roll or pitch — and is passed
to ``reference_generator_balance`` as ``base_quat_ref``.

Typical use::

    from mpx.utils.quad_utils_balance.desired_pose_sampler import (
        DesiredPoseSampler, DesiredPoseConfig, desired_pose_config,
    )

    sampler = DesiredPoseSampler.from_config(config.robot_height, desired_pose_config)

    # In _respawn():
    desired_height, desired_quat = sampler.sample()
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DesiredPoseConfig:
    """Sampling ranges for the desired balance pose (height + yaw)."""

    enabled: bool = True

    # Target height range [m].
    height_range: tuple[float, float] = (0.20, 0.30)

    # Desired body yaw range [deg].
    roll_range_deg: tuple[float, float] = (-30.5, 30.5)
    pitch_range_deg: tuple[float, float] = (-30.5, 30.5)
    yaw_range_deg: tuple[float, float] = (-30.0, 30.0)

    # None → non-reproducible; set an int for repeatable sampling.
    rng_seed: int | None = None


class DesiredPoseSampler:
    """Samples a random target height and body yaw for balance MPC."""

    def __init__(
        self,
        nominal_height: float,
        height_range: tuple[float, float],
        yaw_range_deg: tuple[float, float],
        roll_range_deg: tuple[float, float],
        pitch_range_deg: tuple[float, float],
        enabled: bool = True,
        rng_seed: int | None = None,
    ):
        self.nominal_height = nominal_height
        self.height_range = height_range
        self.yaw_range_deg = yaw_range_deg
        self.roll_range_deg = roll_range_deg
        self.pitch_range_deg = pitch_range_deg
        self.enabled = enabled
        self._rng = np.random.default_rng(rng_seed)

    @classmethod
    def from_config(cls, nominal_height: float, cfg: DesiredPoseConfig) -> DesiredPoseSampler:
        """Build from :class:`DesiredPoseConfig`."""
        return cls(
            nominal_height=nominal_height,
            height_range=cfg.height_range,
            yaw_range_deg=cfg.yaw_range_deg,
            roll_range_deg=cfg.roll_range_deg,
            pitch_range_deg=cfg.pitch_range_deg,
            enabled=cfg.enabled,
            rng_seed=cfg.rng_seed,
        )

    def sample(self) -> tuple[float, np.ndarray]:
        """Return ``(height, quat_wxyz)`` — a random height and pure-yaw quaternion.

        When ``enabled=False`` returns the nominal height and identity quaternion.
        """
        if not self.enabled:
            return self.nominal_height, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        height = float(self._rng.uniform(*self.height_range))
        roll   = np.deg2rad(float(self._rng.uniform(*self.roll_range_deg)))
        pitch  = np.deg2rad(float(self._rng.uniform(*self.pitch_range_deg)))
        yaw    = np.deg2rad(float(self._rng.uniform(*self.yaw_range_deg)))

        cr, sr = np.cos(roll / 2),  np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2),   np.sin(yaw / 2)

        quat_wxyz = np.array([
            cr*cp*cy + sr*sp*sy,   # w
            sr*cp*cy - cr*sp*sy,   # x
            cr*sp*cy + sr*cp*sy,   # y
            cr*cp*sy - sr*sp*cy,   # z
        ], dtype=np.float64)
        return height, quat_wxyz


# Default profile used by balance examples.
desired_pose_config = DesiredPoseConfig()