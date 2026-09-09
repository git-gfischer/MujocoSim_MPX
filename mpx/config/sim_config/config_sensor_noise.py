"""
Observation noise for proprioceptive dataset collection.

Applied only when packing recorded samples (``--collect``). The simulator
state and MPC stay noise-free. Joint positions and IMU get additive noise;
derived channels (joint velocity, torque, foot kinematics, GRF) stay clean.

Typical use::

    from mpx.config.sim_config.config_sensor_noise import sensor_noise_config
    from mpx.utils.simulation_utils.sensor_noise import SensorNoise

    noise = SensorNoise.from_config(dt=1.0 / control_hz, cfg=sensor_noise_config)
    joint_pos, imu_acc, imu_gyro = noise.apply(joint_pos, imu_acc, imu_gyro)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SensorNoiseConfig:
    """Configuration for ``SensorNoise`` (collection-time observation noise)."""

    # If False, recorded samples stay ground-truth.
    enabled: bool = True

    # Reproducibility for the observation RNG. ``None`` is nondeterministic.
    rng_seed: int | None = 42

    # Joint encoder white noise std [rad], i.i.d. per sample and joint.
    joint_pos_std: float = 0.01

    # IMU white noise std at the collection sample rate (not a density).
    imu_acc_std: float = 0.05   # [m/s^2]
    imu_gyro_std: float = 0.01  # [rad/s]

    # If True, also add a slowly drifting IMU bias (discrete Wiener process).
    # Joint encoders are not random-walked.
    imu_random_walk: bool = False

    # Bias random-walk intensity: increment std = value * sqrt(dt) each sample.
    imu_acc_bias_rw_std: float = 0.01    # [(m/s^2)/sqrt(s)]
    imu_gyro_bias_rw_std: float = 0.001  # [(rad/s)/sqrt(s)]


# Default profile used by collection; reassign or construct ``SensorNoiseConfig(...)`` to tune.
sensor_noise_config = SensorNoiseConfig()
