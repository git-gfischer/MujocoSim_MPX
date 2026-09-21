"""Live PI observations must follow the same chain as dataset collection."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mpx.utils.simulation_utils.measured_kinematics import (
    ImuReader,
    MeasuredKinematics,
    gravity_align,
    quat_to_rotmat,
)
from mpx.utils.simulation_utils.onboard_proprioception import OnboardProprioception
from mpx.utils.simulation_utils.sim_utils import (
    feet_yaw_base_kinematics,
    geom_ids,
    geom_linear_velocities,
)


SCENE = "mpx/data/go2/scene_flat.xml"
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
FOOT_SITES = tuple(f"{name}_foot" for name in FOOT_GEOMS)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _pitched_go2():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    yaw = np.deg2rad(35.0)
    pitch = np.deg2rad(20.0)
    qy = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    qp = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    data.qpos[:] = 0.0
    data.qpos[:3] = [1.0, 2.0, 0.32]
    data.qpos[3:7] = _quat_mul(qy, qp)
    data.qpos[7:19] = np.tile([0.0, 0.9, -1.8], 4)
    data.qvel[:] = 0.0
    data.qvel[:3] = [1.2, -0.3, 0.05]
    data.qvel[3:6] = [0.15, 0.6, -0.25]
    data.qvel[6:18] = np.linspace(-0.8, 0.8, 12)
    mujoco.mj_forward(model, data)
    return model, data


def _yaw_align_world_vel(model, data, contact_ids):
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    ryaw_t = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    v_foot_w = geom_linear_velocities(model, data, contact_ids)
    v_body = ryaw_t @ np.asarray(data.qvel[:3], dtype=np.float64)
    return (ryaw_t @ v_foot_w.T).T, v_body


def test_foot_pos_is_fk_of_joints_gravity_aligned_with_estimate():
    model, data = _pitched_go2()
    observer = OnboardProprioception.from_model(
        model, n_joints=12, dt=0.02, enable_noise=False, foot_geom_names=FOOT_GEOMS
    )
    sample = observer.sample(data)

    kin = MeasuredKinematics(model, FOOT_SITES, 12)
    q = np.asarray(data.qpos[7:19], dtype=np.float64)
    dq = np.asarray(data.qvel[6:18], dtype=np.float64)
    pos_base, vel_base = kin(q, dq)
    pos_yaw = gravity_align(pos_base, sample.base_quat_est)
    vel_yaw = gravity_align(vel_base, sample.base_quat_est)

    np.testing.assert_allclose(sample.foot_pos_yawbase, pos_yaw, atol=1e-7)
    np.testing.assert_allclose(sample.foot_vel_yawbase, vel_yaw, atol=1e-7)
    np.testing.assert_allclose(sample.joint_pos, q, atol=1e-6)
    np.testing.assert_allclose(sample.joint_vel, dq, atol=1e-6)


def test_foot_pos_is_not_true_quat_geom_yaw_base():
    model, data = _pitched_go2()
    observer = OnboardProprioception.from_model(
        model, n_joints=12, dt=0.02, enable_noise=False, foot_geom_names=FOOT_GEOMS
    )
    sample = observer.sample(data)
    contact_ids = geom_ids(model, FOOT_GEOMS)
    geom_pos, _ = feet_yaw_base_kinematics(model, data, contact_ids)

    # 20 deg pitch: body vs true-quat yaw-base already disagrees by ~10 cm.
    assert np.max(np.abs(sample.foot_pos_yawbase - geom_pos)) > 0.01


def test_foot_vel_is_jacobian_not_no_slip_residual():
    model, data = _pitched_go2()
    observer = OnboardProprioception.from_model(
        model, n_joints=12, dt=0.02, enable_noise=False, foot_geom_names=FOOT_GEOMS
    )
    sample = observer.sample(data)

    contact_ids = geom_ids(model, FOOT_GEOMS)
    _, p_dot = feet_yaw_base_kinematics(model, data, contact_ids)
    v_foot_yaw, v_body = _yaw_align_world_vel(model, data, contact_ids)
    omega = np.asarray(data.qvel[3:6], dtype=np.float64)
    p = sample.foot_pos_yawbase.reshape(4, 3)
    residual = (p_dot.reshape(4, 3) + v_body + np.cross(omega, p)).reshape(-1)

    err_jac = np.max(np.abs(sample.foot_vel_yawbase - gravity_align(
        MeasuredKinematics(model, FOOT_SITES, 12)(
            data.qpos[7:19], data.qvel[6:18]
        )[1],
        sample.base_quat_est,
    )))
    err_residual = np.max(np.abs(sample.foot_vel_yawbase - residual))
    err_world = np.max(np.abs(sample.foot_vel_yawbase - v_foot_yaw.reshape(-1)))

    assert err_jac < 1e-7
    assert err_residual > 0.05
    assert err_world > 0.05


def test_imu_is_sensor_specific_force_not_qacc():
    model, data = _pitched_go2()
    observer = OnboardProprioception.from_model(
        model, n_joints=12, dt=0.02, enable_noise=False, foot_geom_names=FOOT_GEOMS
    )
    sample = observer.sample(data)
    imu = ImuReader.from_model(model).read(data)

    np.testing.assert_allclose(sample.imu_acc_body, imu["acc"], atol=1e-12)
    np.testing.assert_allclose(sample.imu_gyro_body, imu["gyro"], atol=1e-12)
    assert np.linalg.norm(sample.imu_acc_body - data.qacc[:3]) > 1.0


def test_noisy_feet_are_fk_of_the_logged_joints():
    model, data = _pitched_go2()
    observer = OnboardProprioception.from_model(
        model, n_joints=12, dt=0.02, enable_noise=True, foot_geom_names=FOOT_GEOMS,
        rng=np.random.default_rng(0),
    )
    # Joint latency is one 50 Hz step; the second sample is the delayed noisy q.
    observer.sample(data)
    sample = observer.sample(data)

    kin = MeasuredKinematics(model, FOOT_SITES, 12)
    pos_base, vel_base = kin(sample.joint_pos, sample.joint_vel)
    pos_yaw = gravity_align(pos_base, sample.base_quat_est)
    vel_yaw = gravity_align(vel_base, sample.base_quat_est)

    np.testing.assert_allclose(sample.foot_pos_yawbase, pos_yaw, atol=1e-7)
    np.testing.assert_allclose(sample.foot_vel_yawbase, vel_yaw, atol=1e-7)
    assert not np.allclose(sample.joint_pos, data.qpos[7:19], atol=1e-6)


def test_attitude_for_yaw_align_is_the_estimate_not_ground_truth():
    model, data = _pitched_go2()
    observer = OnboardProprioception.from_model(
        model, n_joints=12, dt=0.02, enable_noise=False, foot_geom_names=FOOT_GEOMS
    )
    sample = observer.sample(data)
    true_quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    kin = MeasuredKinematics(model, FOOT_SITES, 12)
    pos_base, _ = kin(data.qpos[7:19], data.qvel[6:18])
    from_true = gravity_align(pos_base, true_quat)
    from_est = gravity_align(pos_base, sample.base_quat_est)

    np.testing.assert_allclose(sample.foot_pos_yawbase, from_est, atol=1e-7)
    # First-sample seed comes from the accelerometer, not base_quat.
    assert np.max(np.abs(from_est - from_true)) > 1e-4
    np.testing.assert_allclose(
        sample.R_est, quat_to_rotmat(sample.base_quat_est), atol=1e-12
    )
