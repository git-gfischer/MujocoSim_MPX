"""Tests for measured (applied) joint torque, not the MPC command."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mpx.utils.simulation_utils.measured_kinematics import measured_joint_torque


def test_measured_joint_torque_uses_actuator_force_not_command():
    n = 12
    applied = np.linspace(-20.0, 20.0, n)
    data = SimpleNamespace(
        actuator_force=applied,
        qfrc_actuator=np.full(6 + n, 99.0),
    )

    got = measured_joint_torque(data, n)

    np.testing.assert_array_equal(got, applied)
    assert got.dtype == np.float64


def test_measured_joint_torque_falls_back_to_qfrc_actuator():
    n = 12
    qfrc = np.arange(6 + n, dtype=np.float64)
    data = SimpleNamespace(
        actuator_force=np.array([]),
        qfrc_actuator=qfrc,
    )

    got = measured_joint_torque(data, n)

    np.testing.assert_array_equal(got, qfrc[6 : 6 + n])
