"""Torque lag and torque-speed limit between the MPC command and ``data.ctrl``.

See :mod:`mpx.config.sim_config.config_motor_model` for the model and its parameters.
"""
from __future__ import annotations

import numpy as np

from mpx.config.sim_config.config_motor_model import MotorModelConfig, motor_model_config

_REQUIRED = (
    "torque_time_constant_s",
    "peak_torque_nm",
    "corner_speed_rad_s",
    "no_load_speed_rad_s",
)


class MotorModel:
    """Per-robot actuator state. Call once per physics step; ``reset()`` on respawn."""

    def __init__(self, cfg: MotorModelConfig, *, dt: float, n_joints: int) -> None:
        self.cfg = cfg
        self.dt = float(dt)
        self.n_joints = int(n_joints)
        self._tau = np.zeros(self.n_joints, dtype=np.float64)
        if not cfg.enabled:
            return
        missing = [name for name in _REQUIRED if getattr(cfg, name) is None]
        if missing:
            raise ValueError(f"motor model enabled but unset: {missing}")
        if self.n_joints % 3:
            raise ValueError(f"n_joints must be 3 per leg, got {self.n_joints}")
        legs = self.n_joints // 3
        self._peak = np.tile(np.asarray(cfg.peak_torque_nm, dtype=np.float64), legs)
        self._corner = np.tile(np.asarray(cfg.corner_speed_rad_s, dtype=np.float64), legs)
        self._no_load = np.tile(np.asarray(cfg.no_load_speed_rad_s, dtype=np.float64), legs)
        if np.any(self._peak <= 0) or np.any(self._corner < 0) or np.any(self._no_load <= self._corner):
            raise ValueError("need peak > 0, corner >= 0, and no_load > corner for every joint")
        tc = float(cfg.torque_time_constant_s)
        if tc < 0:
            raise ValueError(f"torque_time_constant_s must be >= 0, got {tc}")
        self._alpha = 1.0 if tc == 0 else 1.0 - float(np.exp(-self.dt / tc))

    @classmethod
    def from_config(
        cls, *, dt: float, n_joints: int, cfg: MotorModelConfig | None = None
    ) -> "MotorModel":
        return cls(cfg if cfg is not None else motor_model_config, dt=dt, n_joints=n_joints)

    def reset(self) -> None:
        self._tau[:] = 0.0

    def __call__(self, tau_cmd, joint_vel) -> np.ndarray:
        """Torque to write into ``data.ctrl``. ``joint_vel`` is the true joint velocity."""
        tau_cmd = np.asarray(tau_cmd, dtype=np.float64).reshape(-1)
        if not self.cfg.enabled:
            return tau_cmd
        tau = self._tau + self._alpha * (tau_cmd - self._tau)
        w = np.asarray(joint_vel, dtype=np.float64).reshape(-1)
        drop = np.clip((self._no_load - np.abs(w)) / (self._no_load - self._corner), 0.0, 1.0)
        limit = np.where(tau * w > 0.0, self._peak * drop, self._peak)
        self._tau = np.clip(tau, -limit, limit)
        return self._tau.copy()
