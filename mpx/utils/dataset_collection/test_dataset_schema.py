"""Tests for the per-timestep episode schema and its derived labels."""

from __future__ import annotations

import numpy as np
import pytest

from mpx.utils.dataset_collection.dataset_bucket_system import (
    RARE_CONTACT_STATE,
    contact_state_name,
)
from mpx.utils.dataset_collection.dataset_schema import (
    EPISODE_COLUMNS,
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    EpisodeMetadata,
    EpisodeOutcome,
    EpisodeRecord,
    assign_split,
    classify_termination,
    contact_bits_from_grf,
    dt_since_transition,
    empty_arrays,
    flatten_inputs,
    input_channel_slices,
    structured_dtype,
    validate_arrays,
)


# ── column layout ────────────────────────────────────────────────────────────

def test_every_requested_channel_has_a_column():
    names = {c.name for c in EPISODE_COLUMNS}
    assert {"joint_pos", "joint_vel", "joint_torque"} <= names
    assert {"imu_acc", "imu_gyro"} <= names
    assert {"foot_pos_base", "foot_vel_base"} <= names
    assert {"contact", "grf_world", "external_force", "base_lin_vel"} <= names
    assert {"rare_contact", "dt_since_transition"} <= names


def test_ground_truth_channels_are_targets():
    target_names = {c.name for c in TARGET_COLUMNS}
    assert target_names == {"contact", "grf_world", "external_force", "base_lin_vel"}


def test_contact_target_is_four_binaries():
    contact = next(c for c in EPISODE_COLUMNS if c.name == "contact")
    assert contact.width == 4
    assert np.dtype(contact.dtype) == np.uint8


def test_structured_dtype_covers_all_columns():
    dtype = structured_dtype()
    assert list(dtype.names) == [c.name for c in EPISODE_COLUMNS]


def test_flatten_inputs_matches_channel_slices():
    arrays = empty_arrays(5)
    for column in INPUT_COLUMNS:
        arrays[column.name][:] = np.arange(
            np.prod((5, *column.shape))
        ).reshape((5, *column.shape))

    flat = flatten_inputs(arrays)
    slices = input_channel_slices()
    assert flat.shape == (5, sum(c.width for c in INPUT_COLUMNS))
    for column in INPUT_COLUMNS:
        start, end = slices[column.name]
        expected = np.asarray(arrays[column.name], dtype=np.float32).reshape(5, -1)
        np.testing.assert_allclose(flat[:, start:end], expected)


def test_validate_arrays_rejects_a_ragged_column():
    arrays = empty_arrays(4)
    arrays["joint_pos"] = arrays["joint_pos"][:3]
    with pytest.raises(ValueError, match="rows"):
        validate_arrays(arrays)


# ── contact derivation ───────────────────────────────────────────────────────

def test_contact_bits_threshold_force_magnitude():
    grf = np.zeros((3, 4, 3))
    grf[0, :, 2] = 40.0          # all four feet loaded
    grf[1, 0, 2] = 40.0          # FL only
    grf[2, 0, 2] = 1.0           # FL below threshold

    bits = contact_bits_from_grf(grf, contact_force_threshold=5.0)
    np.testing.assert_array_equal(bits[0], [1, 1, 1, 1])
    np.testing.assert_array_equal(bits[1], [1, 0, 0, 0])
    np.testing.assert_array_equal(bits[2], [0, 0, 0, 0])


def test_single_foot_stance_is_rare_not_dropped():
    assert contact_state_name((1, 1, 1, 1)) == "FULL"
    assert contact_state_name((1, 0, 0, 1)) == "DIAG_FL_RR"
    for pattern in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)):
        assert contact_state_name(pattern) == RARE_CONTACT_STATE


# ── dt since transition ──────────────────────────────────────────────────────

def test_dt_resets_at_touchdown_and_at_liftoff():
    # One foot: stance for 3 steps, swing for 3, stance again.
    contact = np.zeros((9, 4), dtype=np.uint8)
    contact[0:3, 0] = 1
    contact[6:9, 0] = 1

    elapsed = dt_since_transition(contact, dt=0.02)[:, 0]

    # First step has no observed history, so it starts at zero.
    np.testing.assert_allclose(elapsed[0:3], [0.0, 0.02, 0.04])
    # Liftoff at step 3 resets the timer...
    np.testing.assert_allclose(elapsed[3:6], [0.0, 0.02, 0.04])
    # ...and so does touchdown at step 6.
    np.testing.assert_allclose(elapsed[6:9], [0.0, 0.02, 0.04])


def test_dt_is_tracked_per_foot_independently():
    contact = np.ones((4, 4), dtype=np.uint8)
    contact[2:, 1] = 0            # only FR lifts off, at step 2

    elapsed = dt_since_transition(contact, dt=0.02)
    np.testing.assert_allclose(elapsed[3, 0], 0.06)   # FL never changed
    np.testing.assert_allclose(elapsed[3, 1], 0.02)   # FR reset one step ago


def test_dt_of_empty_episode_is_empty():
    assert dt_since_transition(np.zeros((0, 4), dtype=np.uint8), dt=0.02).shape == (0, 4)


# ── termination ──────────────────────────────────────────────────────────────

def test_goal_is_success_crash_and_pose_loss_are_failures():
    assert classify_termination("goal_reached") is EpisodeOutcome.SUCCESS
    assert classify_termination("crash") is EpisodeOutcome.FAILURE
    assert classify_termination("pose_lost") is EpisodeOutcome.FAILURE


def test_timer_and_shutdown_truncate_rather_than_judge():
    assert classify_termination("duration_cap") is EpisodeOutcome.TRUNCATED
    assert classify_termination("shutdown") is EpisodeOutcome.TRUNCATED
    assert classify_termination("something_new") is EpisodeOutcome.TRUNCATED


# ── episode split ────────────────────────────────────────────────────────────

def test_split_is_deterministic_for_an_episode_id():
    first = assign_split("run_00042")
    assert all(assign_split("run_00042") == first for _ in range(5))


def test_split_seed_changes_the_assignment():
    ids = [f"ep_{i:04d}" for i in range(400)]
    a = [assign_split(e, seed=0) for e in ids]
    b = [assign_split(e, seed=1) for e in ids]
    assert a != b


def test_split_ratios_are_approximately_respected():
    ids = [f"ep_{i:05d}" for i in range(4000)]
    splits = [assign_split(e, val_ratio=0.15, test_ratio=0.10) for e in ids]
    assert abs(splits.count("test") / len(ids) - 0.10) < 0.02
    assert abs(splits.count("val") / len(ids) - 0.15) < 0.02
    assert abs(splits.count("train") / len(ids) - 0.75) < 0.02


def test_invalid_split_ratios_are_rejected():
    with pytest.raises(ValueError):
        assign_split("ep", val_ratio=0.8, test_ratio=0.4)


# ── record ───────────────────────────────────────────────────────────────────

def test_record_fills_in_step_count_and_duration():
    record = EpisodeRecord(
        metadata=EpisodeMetadata(episode_id="ep_1", control_hz=50.0),
        arrays=empty_arrays(100),
    )
    assert record.n_steps == 100
    assert record.metadata.duration_s == pytest.approx(2.0)


def test_metadata_row_carries_every_requested_field():
    row = EpisodeMetadata(
        episode_id="ep_1",
        terrain="flat",
        gait="trot",
        timestamp="2026-09-14T10:00:00+02:00",
        friction=0.6,
        payload_kg=1.5,
        terminate_by="success",
    ).to_row()
    for key in (
        "episode_id", "terrain", "gait", "timestamp",
        "friction", "payload_kg", "terminate_by",
    ):
        assert key in row
    assert row["friction"] == pytest.approx(0.6)
    assert row["payload_kg"] == pytest.approx(1.5)
