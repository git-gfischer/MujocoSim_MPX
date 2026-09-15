"""
Tests for the round-2 fixes (DATASET_FIX_TASKS_R2).

Each test names the defect it locks down, so a future regression reports what
broke rather than only which assert failed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpx.utils.dataset_collection.dataset_schema import (
    COLUMNS_BY_NAME,
    INPUT_COLUMNS,
    PRIVILEGED_COLUMNS,
)
from mpx.utils.dataset_collection.signal_bounds import (
    CLIPPING_LIMIT,
    clipping_audit,
    load_signal_bounds,
    merge_clipping_audits,
    signal_bounds_version,
    worst_clipping,
)
from mpx.utils.simulation_utils.attitude_estimator import ComplementaryAttitudeFilter
from mpx.utils.simulation_utils.velocity_command import (
    VelocityCommandConfig,
    VelocityCommandSampler,
    command_from_mpc_input,
)


# ── R2-1: frames and roles ───────────────────────────────────────────────────

def test_measured_and_true_foot_columns_share_a_frame():
    """A *_true twin must differ from its measured counterpart only by noise."""
    for measured, truth in (
        ("foot_pos_base", "foot_pos_base_true"),
        ("foot_vel_base", "foot_vel_base_true"),
        ("foot_pos_yawbase", "foot_pos_yawbase_true"),
        ("foot_vel_yawbase", "foot_vel_yawbase_true"),
    ):
        assert COLUMNS_BY_NAME[measured].frame == COLUMNS_BY_NAME[truth].frame, (
            f"{measured} and {truth} are in different frames"
        )


def test_body_frame_columns_are_labelled_base():
    assert COLUMNS_BY_NAME["foot_pos_base"].frame == "base"
    assert COLUMNS_BY_NAME["foot_pos_base_true"].frame == "base"
    assert COLUMNS_BY_NAME["foot_pos_yawbase"].frame == "yaw_base"


def test_ground_truth_attitude_is_privileged_and_the_estimate_is_the_input():
    assert COLUMNS_BY_NAME["base_quat"].role == "privileged"
    assert COLUMNS_BY_NAME["base_quat_est"].role == "input"


def test_no_privileged_column_leaks_into_the_input_set():
    input_names = {c.name for c in INPUT_COLUMNS}
    privileged_names = {c.name for c in PRIVILEGED_COLUMNS}
    assert not (input_names & privileged_names)
    # The specific channels the audit flagged.
    for name in ("base_quat", "joint_torque_cmd", "contact_schedule",
                 "foot_pos_base_true", "foot_pos_yawbase_true"):
        assert name not in input_names


# ── R2-1: the attitude estimator uses only the IMU ───────────────────────────

def test_attitude_filter_recovers_a_static_tilt():
    """A level robot reads +9.81 on z; a tilted one shifts the gravity vector."""
    filter_ = ComplementaryAttitudeFilter(dt=0.02)
    filter_.reset()
    # 0.2 rad of roll: gravity rotates into the y axis.
    roll = 0.2
    acc = np.array([0.0, 9.81 * np.sin(roll), 9.81 * np.cos(roll)])
    for _ in range(200):
        filter_.update(acc, np.zeros(3))
    assert abs(filter_.roll - roll) < 0.02
    assert abs(filter_.pitch) < 0.02


def test_attitude_filter_ignores_the_accelerometer_under_high_acceleration():
    """While the robot accelerates hard the accelerometer is not gravity."""
    filter_ = ComplementaryAttitudeFilter(dt=0.02)
    filter_.reset()
    filter_.update(np.array([0.0, 0.0, 9.81]), np.zeros(3))
    before = filter_.roll
    # 4 g sideways: far outside the gravity tolerance, so it must be rejected.
    for _ in range(10):
        filter_.update(np.array([0.0, 40.0, 9.81]), np.zeros(3))
    assert abs(filter_.roll - before) < 1e-6


def test_attitude_filter_quaternion_is_unit_norm():
    filter_ = ComplementaryAttitudeFilter(dt=0.02)
    filter_.reset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        quat = filter_.update(
            np.array([0.0, 0.5, 9.7]) + rng.normal(0, 0.3, 3),
            rng.normal(0, 0.1, 3),
        )
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-9


# ── R2-2: the command vector, and the index-5 trap ───────────────────────────

def test_yaw_rate_is_read_from_index_five_not_index_two():
    """
    The MPC command is [vx, vy, 0, 0, 0, yaw_rate, height].

    Slicing ``[:3]`` off it logged a structural zero for yaw across the whole of
    the audited v4 run.
    """
    mpc_input = np.array([0.4, -0.1, 0.0, 0.0, 0.0, 0.9, 0.27])
    command = command_from_mpc_input(mpc_input)
    np.testing.assert_allclose(command, [0.4, -0.1, 0.9])
    assert command[2] != 0.0


def test_command_sampler_covers_reverse_and_turning():
    sampler = VelocityCommandSampler(
        dt=0.02, rng=np.random.default_rng(0),
        config=VelocityCommandConfig(hold_s=(0.2, 0.4)),
    )
    sampler.reset()
    commands = np.stack([sampler.step() for _ in range(5000)])

    assert commands[:, 0].min() < -0.5, "no commanded reverse"
    assert np.abs(commands[:, 2]).max() > 0.5, "no commanded yaw"
    assert (commands[:, 0] < -0.1).mean() > 0.10, "reverse under-sampled"
    assert (np.abs(commands[:, 2]) > 0.3).mean() > 0.15, "turning under-sampled"


def test_command_sampler_segments_and_ramps():
    sampler = VelocityCommandSampler(
        dt=0.02, rng=np.random.default_rng(1),
        config=VelocityCommandConfig(hold_s=(2.0, 5.0), ramp_s=0.3),
    )
    sampler.reset()
    segments, commands = [], []
    for _ in range(3000):          # 60 s at 50 Hz
        commands.append(sampler.step())
        segments.append(sampler.segment_id)

    n_segments = len(set(segments))
    assert 10 <= n_segments <= 35, f"{n_segments} segments in 60 s (want 12-30)"

    # The ramp keeps step-to-step changes small: no discontinuity at a boundary.
    steps = np.abs(np.diff(np.stack(commands), axis=0)).max()
    assert steps < 0.25, f"command jumps by {steps:.3f} in one step — ramp missing"


def test_command_sampler_includes_standing_segments():
    sampler = VelocityCommandSampler(
        dt=0.02, rng=np.random.default_rng(2),
        config=VelocityCommandConfig(hold_s=(0.2, 0.3), zero_command_prob=0.10),
    )
    sampler.reset()
    commands = np.stack([sampler.step() for _ in range(20000)])
    standing = (np.abs(commands).max(axis=1) < 1e-9).mean()
    assert standing > 0.02, "no standing-in-place segments"


# ── R2-5: the clipping audit ─────────────────────────────────────────────────

def _audit_arrays(n_rows: int, out_of_range_rows: int):
    """
    A clean joint_pos block with a known number of out-of-range elements.

    Built from the stance posture, not from zeros: the knee range is entirely
    negative, so an all-zero array is out of bounds on every KFE element and
    would make the fixture measure itself.
    """
    bounds = load_signal_bounds()
    stance = bounds["stance_posture_rad"]
    row = np.tile([stance["HAA"], stance["HFE"], stance["KFE"]], 4)
    arrays = {"joint_pos": np.tile(row, (n_rows, 1)).astype(np.float32)}

    # Push one element of the first ``out_of_range_rows`` rows past its bound.
    high_haa = bounds["signals"]["joint_pos"]["bounds"]["HAA"][1]
    arrays["joint_pos"][:out_of_range_rows, 0] = high_haa + 1.0
    return arrays, bounds


def test_clipping_audit_reports_both_definitions():
    arrays, bounds = _audit_arrays(100, 10)
    audit = clipping_audit(arrays, bounds)

    assert "definition" in audit
    # 10 of 1200 elements, but 10 of 100 rows: the two definitions must differ.
    assert audit["per_element"]["joint_pos"] == pytest.approx(10 / 1200)
    assert audit["per_row_any"]["joint_pos"] == pytest.approx(0.10)


def test_clipping_audits_merge_by_count_not_by_average():
    """A 61-step episode must not weigh as much as a 3,000-step one."""
    big, bounds = _audit_arrays(1000, 0)       # clean, long
    small, _ = _audit_arrays(10, 10)           # dirty, short
    merged = merge_clipping_audits(
        [clipping_audit(big, bounds), clipping_audit(small, bounds)]
    )
    # 10 bad elements out of (1000 + 10) * 12.
    assert merged["per_element"]["joint_pos"] == pytest.approx(10 / (1010 * 12))
    # A naive mean of the two episode fractions would give ~0.042 instead.
    assert merged["per_element"]["joint_pos"] < 0.002


def test_worst_clipping_picks_the_worst_channel():
    arrays, bounds = _audit_arrays(100, 50)
    name, fraction = worst_clipping(clipping_audit(arrays, bounds))
    assert name == "joint_pos"
    assert fraction > CLIPPING_LIMIT


def test_joint_pos_bounds_carry_a_noise_margin():
    """
    A joint resting on its limit must not saturate once noise is added.

    The MJCF range is +-1.0472; the normalisation bound has to sit outside it.
    """
    bounds = load_signal_bounds()
    haa = bounds["signals"]["joint_pos"]["bounds"]["HAA"]
    assert haa[0] < -1.0472 and haa[1] > 1.0472
    assert signal_bounds_version(bounds) >= 2


def test_knee_velocity_bound_covers_the_observed_motion():
    """Go2 actuator speed is 30.1 rad/s; observed 22.61 rad/s must sit inside it."""
    bounds = load_signal_bounds()
    kfe = bounds["signals"]["joint_vel"]["bounds"]["KFE"]
    assert kfe[1] >= 30.1


# ── R2-7: clearance columns ──────────────────────────────────────────────────

def test_clearance_and_height_label_columns_exist():
    assert COLUMNS_BY_NAME["foot_clearance"].role == "privileged"
    assert COLUMNS_BY_NAME["contact_from_height"].role == "context"
    # The site-measured height must say so, or someone will threshold it.
    assert "site" in COLUMNS_BY_NAME["foot_height_terrain"].doc.lower()
