"""
Collection-time observation noise for joint encoders and IMU.

Functionality
-------------
- Additive i.i.d. Gaussian noise on joint positions and IMU accel/gyro.
- Optional IMU bias random walk (Wiener process), toggled independently.
- Does **not** mutate MuJoCo state; call on copied observation arrays only.
- ``reset()`` zeros IMU biases (call at episode start / respawn).

Reads defaults from
``mpx.config.sim_config.config_sensor_noise.sensor_noise_config``.

Example
-------
```python
from mpx.config.sim_config.config_sensor_noise import sensor_noise_config
from mpx.utils.simulation_utils.sensor_noise import SensorNoise

noise = SensorNoise.from_config(dt=1.0 / 50.0, cfg=sensor_noise_config)
q, acc, gyro = noise.apply(joint_pos, imu_acc, imu_gyro)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mpx.config.sim_config.config_sensor_noise import (
    SensorNoiseConfig,
    sensor_noise_config,
)


@dataclass
class SensorNoise:
    """Corrupt joint-position and IMU observations; leave derived signals alone."""

    rng: np.random.Generator
    dt: float
    enabled: bool = sensor_noise_config.enabled
    joint_pos_std: float = sensor_noise_config.joint_pos_std
    imu_acc_std: float = sensor_noise_config.imu_acc_std
    imu_gyro_std: float = sensor_noise_config.imu_gyro_std
    imu_random_walk: bool = sensor_noise_config.imu_random_walk
    imu_acc_bias_rw_std: float = sensor_noise_config.imu_acc_bias_rw_std
    imu_gyro_bias_rw_std: float = sensor_noise_config.imu_gyro_bias_rw_std
    _acc_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
        init=False,
        repr=False,
    )
    _gyro_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
        init=False,
        repr=False,
    )

    @classmethod
    def from_config(
        cls,
        dt: float,
        cfg: SensorNoiseConfig = sensor_noise_config,
    ) -> SensorNoise:
        """Build an observation-noise model from :class:`SensorNoiseConfig`."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        noise = cls(
            rng=np.random.default_rng(cfg.rng_seed),
            dt=float(dt),
            enabled=cfg.enabled,
            joint_pos_std=float(cfg.joint_pos_std),
            imu_acc_std=float(cfg.imu_acc_std),
            imu_gyro_std=float(cfg.imu_gyro_std),
            imu_random_walk=cfg.imu_random_walk,
            imu_acc_bias_rw_std=float(cfg.imu_acc_bias_rw_std),
            imu_gyro_bias_rw_std=float(cfg.imu_gyro_bias_rw_std),
        )
        noise.reset()
        return noise

    def reset(self) -> None:
        """Zero IMU biases. The RNG is left running so episodes stay independent."""
        self._acc_bias[:] = 0.0
        self._gyro_bias[:] = 0.0

    def to_metadata(self) -> dict:
        """JSON-friendly snapshot of the active noise settings (no RNG state)."""
        return {
            "enabled": bool(self.enabled),
            "dt": float(self.dt),
            "joint_pos_std": float(self.joint_pos_std),
            "imu_acc_std": float(self.imu_acc_std),
            "imu_gyro_std": float(self.imu_gyro_std),
            "imu_random_walk": bool(self.imu_random_walk),
            "imu_acc_bias_rw_std": float(self.imu_acc_bias_rw_std),
            "imu_gyro_bias_rw_std": float(self.imu_gyro_bias_rw_std),
        }

    def apply(
        self,
        joint_pos: np.ndarray,
        imu_acc: np.ndarray,
        imu_gyro: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return noisy copies of encoder and IMU readings.

        ``joint_pos`` is (n_joints,), ``imu_acc`` and ``imu_gyro`` are (3,).
        """
        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1).copy()
        imu_acc = np.asarray(imu_acc, dtype=np.float32).reshape(-1).copy()
        imu_gyro = np.asarray(imu_gyro, dtype=np.float32).reshape(-1).copy()
        if imu_acc.size != 3 or imu_gyro.size != 3:
            raise ValueError("imu_acc and imu_gyro must have length 3")
        if not self.enabled:
            return joint_pos, imu_acc, imu_gyro

        if self.joint_pos_std > 0.0:
            joint_pos = joint_pos + self.rng.normal(
                0.0, self.joint_pos_std, size=joint_pos.shape,
            ).astype(np.float32)

        if self.imu_random_walk:
            sqrt_dt = float(np.sqrt(self.dt))
            if self.imu_acc_bias_rw_std > 0.0:
                self._acc_bias += self.rng.normal(
                    0.0, self.imu_acc_bias_rw_std * sqrt_dt, size=3,
                )
            if self.imu_gyro_bias_rw_std > 0.0:
                self._gyro_bias += self.rng.normal(
                    0.0, self.imu_gyro_bias_rw_std * sqrt_dt, size=3,
                )

        if self.imu_acc_std > 0.0:
            imu_acc = imu_acc + self.rng.normal(
                0.0, self.imu_acc_std, size=3,
            ).astype(np.float32)
        if self.imu_gyro_std > 0.0:
            imu_gyro = imu_gyro + self.rng.normal(
                0.0, self.imu_gyro_std, size=3,
            ).astype(np.float32)

        imu_acc = (imu_acc + self._acc_bias).astype(np.float32, copy=False)
        imu_gyro = (imu_gyro + self._gyro_bias).astype(np.float32, copy=False)
        return joint_pos.astype(np.float32, copy=False), imu_acc, imu_gyro
