"""Onboard proprioception for live PI — same chain as dataset collection.

One noisy sensor reading is the source of truth. Joints, IMU, and torque come
from :class:`~mpx.utils.simulation_utils.sensor_noise.SensorNoise`. Foot
positions and velocities are forward kinematics of that joint vector
(:class:`~mpx.utils.simulation_utils.measured_kinematics.MeasuredKinematics`),
then gravity-aligned with the complementary-filter estimate, never
``base_quat``. The simulator state is not mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import mujoco
import numpy as np

from mpx.config.sim_config.config_sensor_noise import (
    LatencyConfig,
    sensor_noise_config,
)
from mpx.utils.simulation_utils.attitude_estimator import ComplementaryAttitudeFilter
from mpx.utils.simulation_utils.measured_kinematics import (
    ImuReader,
    MeasuredKinematics,
    gravity_align,
    measured_joint_torque,
    quat_to_rotmat,
)
from mpx.utils.simulation_utils.sensor_noise import SensorNoise

# ``SensorNoise`` white-noise stds and the complementary-filter alpha were
# chosen at the 50 Hz collection rate. Preserve those time constants when PI
# samples faster: 1 joint-latency step stays 20 ms, and the filter's ~1 s
# accelerometer blend stays ~1 s.
COLLECTION_DT = 0.02
ATTITUDE_ALPHA_50HZ = 0.98


@dataclass
class OnboardSample:
    """One PI tick as the robot can measure it."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_torque: np.ndarray
    foot_pos_yawbase: np.ndarray
    foot_vel_yawbase: np.ndarray
    imu_acc_body: np.ndarray
    imu_gyro_body: np.ndarray
    base_quat_est: np.ndarray
    R_est: np.ndarray


def _attitude_alpha(dt: float) -> float:
    tau = COLLECTION_DT / (1.0 - ATTITUDE_ALPHA_50HZ)
    return float(np.clip(1.0 - float(dt) / tau, 0.0, 0.999999))


def _noise_config(dt: float, enabled: bool):
    scale = COLLECTION_DT / float(dt)
    latency = LatencyConfig(
        joint=max(0, int(round(sensor_noise_config.latency_steps.joint * scale))),
        imu=max(0, int(round(sensor_noise_config.latency_steps.imu * scale))),
    )
    return replace(sensor_noise_config, enabled=bool(enabled), latency_steps=latency)


class OnboardProprioception:
    """IMU → noise → FK(joints) → gravity-align(estimate)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        n_joints: int,
        dt: float,
        enable_noise: bool = True,
        foot_geom_names: Sequence[str] = ("FL", "FR", "RL", "RR"),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.n_joints = int(n_joints)
        self.dt = float(dt)
        self.foot_site_names = [f"{name}_foot" for name in foot_geom_names]
        self.imu = ImuReader.from_model(model)
        self.kinematics = MeasuredKinematics(model, self.foot_site_names, self.n_joints)
        self.sensor_noise = SensorNoise.from_config(
            dt=self.dt,
            cfg=_noise_config(self.dt, enable_noise),
            rng=rng,
        )
        self.attitude = ComplementaryAttitudeFilter(
            dt=self.dt, alpha=_attitude_alpha(self.dt)
        )
        self.sensor_noise.reset(self.n_joints)

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        *,
        n_joints: int,
        dt: float,
        enable_noise: bool = True,
        foot_geom_names: Sequence[str] = ("FL", "FR", "RL", "RR"),
        rng: np.random.Generator | None = None,
    ) -> "OnboardProprioception":
        return cls(
            model,
            n_joints=n_joints,
            dt=dt,
            enable_noise=enable_noise,
            foot_geom_names=foot_geom_names,
            rng=rng,
        )

    @classmethod
    def from_contact_ids(
        cls,
        model: mujoco.MjModel,
        contact_ids: Sequence[int],
        *,
        n_joints: int,
        dt: float,
        enable_noise: bool = True,
        rng: np.random.Generator | None = None,
    ) -> "OnboardProprioception":
        names = []
        for gid in contact_ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
            if not name:
                raise ValueError(f"No geom name for contact id {gid}")
            names.append(name)
        return cls(
            model,
            n_joints=n_joints,
            dt=dt,
            enable_noise=enable_noise,
            foot_geom_names=names,
            rng=rng,
        )

    def reset(self) -> None:
        self.sensor_noise.reset(self.n_joints)
        self.attitude.reset()

    def sample(self, data: mujoco.MjData) -> OnboardSample:
        q_true = np.asarray(data.qpos[7 : 7 + self.n_joints], dtype=np.float64)
        dq_true = np.asarray(data.qvel[6 : 6 + self.n_joints], dtype=np.float64)
        tau_applied = measured_joint_torque(data, self.n_joints)
        imu_true = self.imu.read(data)

        reading = self.sensor_noise.apply(
            joint_pos=q_true,
            joint_vel=dq_true,
            joint_torque=tau_applied,
            imu_acc=imu_true["acc"],
            imu_gyro=imu_true["gyro"],
        )
        quat_est = self.attitude.update(reading.imu_acc, reading.imu_gyro)
        foot_pos_base, foot_vel_base = self.kinematics(
            reading.joint_pos, reading.joint_vel
        )
        foot_pos_yaw = gravity_align(foot_pos_base, quat_est)
        foot_vel_yaw = gravity_align(foot_vel_base, quat_est)
        return OnboardSample(
            joint_pos=np.asarray(reading.joint_pos, dtype=np.float64),
            joint_vel=np.asarray(reading.joint_vel, dtype=np.float64),
            joint_torque=np.asarray(reading.joint_torque, dtype=np.float64),
            foot_pos_yawbase=np.asarray(foot_pos_yaw, dtype=np.float64),
            foot_vel_yawbase=np.asarray(foot_vel_yaw, dtype=np.float64),
            imu_acc_body=np.asarray(reading.imu_acc, dtype=np.float64),
            imu_gyro_body=np.asarray(reading.imu_gyro, dtype=np.float64),
            base_quat_est=np.asarray(quat_est, dtype=np.float64),
            R_est=quat_to_rotmat(quat_est),
        )
