from pathlib import Path

import mujoco
import numpy as np
import pytest

from framework.config.sim_config.config_motor_model import MotorModelConfig, motor_model_config
from framework.utils.simulation_utils.measured_kinematics import measured_joint_torque
from framework.utils.simulation_utils.motor_model import MotorModel

SCENE = Path(__file__).resolve().parents[2] / "data" / "go2" / "scene_flat.xml"
DT = 0.002


def _on(**kw):
    base = dict(
        enabled=True,
        torque_time_constant_s=0.005,
        peak_torque_nm=(20.0, 20.0, 40.0),
        corner_speed_rad_s=(10.0, 10.0, 8.0),
        no_load_speed_rad_s=(30.0, 30.0, 20.0),
    )
    base.update(kw)
    return MotorModel(MotorModelConfig(**base), dt=DT, n_joints=12)


def test_default_is_off_and_passes_the_command_through():
    assert motor_model_config.enabled is False
    motor = MotorModel.from_config(dt=DT, n_joints=12)
    cmd = np.linspace(-30.0, 30.0, 12)
    np.testing.assert_array_equal(motor(cmd, np.full(12, 50.0)), cmd)


def test_go2_params_file_matches_the_mjx_force_limits():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert motor_model_config.peak_torque_nm == tuple(model.actuator_forcerange[:3, 1])
    assert motor_model_config.no_load_speed_rad_s == (30.1, 30.1, 15.70)


def test_enabled_with_unset_parameters_raises():
    with pytest.raises(ValueError, match="corner_speed_rad_s"):
        MotorModel(MotorModelConfig(enabled=True, torque_time_constant_s=0.005,
                                    no_load_speed_rad_s=(30.0, 30.0, 20.0)), dt=DT, n_joints=12)


def test_first_order_lag_step_response():
    motor = _on()
    cmd = np.full(12, 5.0)
    alpha = 1.0 - np.exp(-DT / 0.005)
    np.testing.assert_allclose(motor(cmd, np.zeros(12)), alpha * cmd)
    for _ in range(200):
        out = motor(cmd, np.zeros(12))
    np.testing.assert_allclose(out, cmd, rtol=1e-6)
    motor.reset()
    np.testing.assert_allclose(motor(cmd, np.zeros(12)), alpha * cmd)


def test_torque_speed_envelope_limits_motoring_not_braking():
    motor = _on(torque_time_constant_s=0.0)
    cmd = np.tile([20.0, 20.0, 40.0], 4)
    # thigh at 20 rad/s: (30 - 20) / (30 - 10) = half of 20 N*m
    vel = np.tile([0.0, 20.0, 20.0], 4)
    out = motor(cmd, vel)
    np.testing.assert_allclose(out[:3], [20.0, 10.0, 0.0])
    braking = motor(cmd, -vel)
    np.testing.assert_allclose(braking[:3], [20.0, 20.0, 40.0])


def test_measured_torque_reads_the_motor_output():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    motor = _on()
    data.ctrl = motor(np.full(12, 5.0), data.qvel[6:18])
    mujoco.mj_step(model, data)
    applied = measured_joint_torque(data, 12)
    np.testing.assert_allclose(applied, data.ctrl)
    assert np.all(applied < 5.0)
