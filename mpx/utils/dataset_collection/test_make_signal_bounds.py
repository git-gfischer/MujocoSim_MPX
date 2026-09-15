"""Signal bounds are built from the robot model, not from a collected run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mpx.config.sim_config.config_sensor_noise import sensor_noise_config
from mpx.utils.dataset_collection.make_signal_bounds import (
    build_signal_bounds,
    write_signal_bounds,
)
from mpx.utils.dataset_collection.signal_bounds import encoder_constraints


def test_go2_joint_pos_bounds_come_from_mjcf_with_margin():
    payload = build_signal_bounds("go2")
    haa = payload["signals"]["joint_pos"]["bounds"]["HAA"]
    assert haa[0] < -1.0472
    assert haa[1] > 1.0472
    stance = payload["stance_posture_rad"]
    assert stance["HAA"] == pytest.approx(0.0)
    assert stance["HFE"] == pytest.approx(0.9)
    assert stance["KFE"] == pytest.approx(-1.8)


def test_go2_torque_bounds_cover_actuator_limit_plus_sensing_error():
    payload = build_signal_bounds("go2")
    haa = payload["signals"]["joint_torque_measured"]["bounds"]["HAA"]
    kfe = payload["signals"]["joint_torque_measured"]["bounds"]["KFE"]
    scale = 1.0 + 2.0 * sensor_noise_config.joint_torque_scale_err_std
    noise = 5.0 * sensor_noise_config.joint_torque_std
    assert haa[1] >= 23.7 * scale + noise
    assert kfe[1] >= 45.43 * scale + noise


def test_go2_joint_vel_is_actuator_capability_not_a_dataset_percentile():
    payload = build_signal_bounds("go2")
    kfe = payload["signals"]["joint_vel"]["bounds"]["KFE"]
    assert kfe[1] >= 30.1
    assert kfe[0] <= -30.1


def test_imu_z_stance_is_gravity():
    payload = build_signal_bounds("go2")
    stance = payload["signals"]["imu_acc_body"]["stance_value"]
    assert stance["z"] == pytest.approx(9.81)


def test_foot_bounds_cover_stance_foot_pose():
    payload = build_signal_bounds("go2")
    pos = payload["signals"]["foot_pos_base"]["bounds"]
    # Nominal Go2 stance puts the feet ~0.19 m forward/back and ~0.14 m left/right.
    assert pos["x"][0] < -0.15 < pos["x"][1]
    assert pos["y"][0] < -0.10 < pos["y"][1]
    assert pos["z"][0] < -0.20 < pos["z"][1]


def test_write_signal_bounds_round_trips(tmp_path: Path):
    payload = build_signal_bounds("go2")
    path = tmp_path / "signal_bounds.json"
    write_signal_bounds(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["robot"] == "go2"
    enc = encoder_constraints("joint_pos", 12, loaded)
    assert enc["constraints"].shape == (2, 12)


def test_spot_uses_spot_mjcf_ranges():
    payload = build_signal_bounds("spot")
    haa = payload["signals"]["joint_pos"]["bounds"]["HAA"]
    assert haa[0] < -0.785
    assert haa[1] > 0.785
    tau = payload["signals"]["joint_torque_measured"]["bounds"]["HAA"]
    assert tau[1] >= 144.4
