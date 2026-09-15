"""
Observation noise for proprioceptive dataset collection.

Applied only when packing recorded samples (``--collect``). The simulator state
and the MPC stay noise-free; the recorder writes the corrupted values and, for
audit, the clean ones under the ``*_true`` privileged columns.

Everything downstream of an encoder is derived from the **noisy** reading:
foot kinematics come from forward kinematics on the measured joint vector, not
from the simulator's geom poses. See
:mod:`mpx.utils.simulation_utils.measured_kinematics`.

Defaults model a Unitree Go2-class platform:

* 14-bit joint encoders → 2π / 2^14 ≈ 3.83e-4 rad quantisation,
* velocity differentiated from position and low-pass filtered, so it is far
  noisier than position,
* torque inferred from motor current, so it carries both additive noise and a
  per-episode scale error,
* a MEMS IMU with a bias random walk — the drift the PI representation's known
  failure mode on real hardware depends on, which is why
  ``imu_random_walk`` defaults to True.

Typical use::

    from mpx.config.sim_config.config_sensor_noise import sensor_noise_config
    from mpx.utils.simulation_utils.sensor_noise import SensorNoise

    noise = SensorNoise.from_config(dt=1.0 / control_hz, cfg=sensor_noise_config)
    reading = noise.apply(joint_pos=q, joint_vel=dq, joint_torque=tau,
                          imu_acc=acc, imu_gyro=gyro)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JointVelFilterConfig:
    """Low-pass applied to the differentiated joint velocity."""

    type: str = "butter_lowpass"
    cutoff_hz: float = 20.0
    order: int = 2


@dataclass
class LatencyConfig:
    """Whole-sample transport delay, in control steps, per sensor group."""

    joint: int = 1
    imu: int = 0


@dataclass
class SensorNoiseConfig:
    """Configuration for ``SensorNoise`` (collection-time observation noise)."""

    # If False, recorded samples stay ground-truth.
    enabled: bool = True

    # Reproducibility for the observation RNG. ``None`` is nondeterministic.
    rng_seed: int | None = 42

    # Control-step period [s]; overridden by ``SensorNoise.from_config(dt=...)``.
    dt: float = 0.02

    # ── joint encoders ───────────────────────────────────────────────────────
    # White noise std [rad], i.i.d. per sample and joint, applied before quantisation.
    joint_pos_std: float = 0.003

    # 14-bit absolute encoder over 2*pi. Quantise AFTER adding noise.
    joint_pos_quantization_rad: float = 0.000383

    # Velocity noise [rad/s] applied to the raw differentiated signal, BEFORE the
    # filter, so the filter shapes it the way a real driver would.
    joint_vel_std: float = 0.15
    joint_vel_filter: JointVelFilterConfig = field(default_factory=JointVelFilterConfig)

    # ── actuator current sensing ─────────────────────────────────────────────
    # Torque is inferred from motor current, never measured directly.
    joint_torque_std: float = 0.3            # [N*m] additive, per sample
    joint_torque_scale_err_std: float = 0.02 # multiplicative, drawn once per episode

    # ── IMU ──────────────────────────────────────────────────────────────────
    # White noise std at the collection sample rate (not a density).
    imu_acc_std: float = 0.30   # [m/s^2]
    imu_gyro_std: float = 0.05  # [rad/s]

    # Slowly drifting bias (discrete Wiener process). Keep this ON: IMU drift is
    # the documented failure mode of the PI representation on real data, and a
    # dataset without it cannot reproduce or study that failure.
    imu_random_walk: bool = True

    # Bias random-walk intensity: increment std = value * sqrt(dt) each sample.
    imu_acc_bias_rw_std: float = 0.01    # [(m/s^2)/sqrt(s)]
    imu_gyro_bias_rw_std: float = 0.001  # [(rad/s)/sqrt(s)]

    # Initial bias drawn once per episode.
    imu_acc_bias_init_std: float = 0.10   # [m/s^2]
    imu_gyro_bias_init_std: float = 0.01  # [rad/s]

    # ── transport ────────────────────────────────────────────────────────────
    latency_steps: LatencyConfig = field(default_factory=LatencyConfig)

    # Probability that a sample is lost and the previous one is held instead.
    dropout_prob: float = 0.001


# Default profile used by collection; reassign or construct ``SensorNoiseConfig(...)`` to tune.
sensor_noise_config = SensorNoiseConfig()
