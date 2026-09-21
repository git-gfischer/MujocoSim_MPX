"""Tests for the per-timestep episode schema and its derived labels."""

from __future__ import annotations

import numpy as np
import pytest

from mpx.utils.dataset_collection.contact_labeling import debounce_sequence

from mpx.utils.dataset_collection.dataset_bucket_system import (
    contact_state_name,
    is_rare_contact,
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
    assert {"imu_acc_body", "imu_gyro_body"} <= names
    assert {"foot_pos_base", "foot_vel_base"} <= names
    assert {"contact", "grf_world", "external_force", "base_lin_vel"} <= names
    assert {"rare_contact", "dt_since_transition"} <= names
    assert {"joint_torque_measured", "base_quat", "contact_raw"} <= names


def test_ground_truth_channels_are_targets():
    target_names = {c.name for c in TARGET_COLUMNS}
    assert target_names == {
        "contact", "grf_base", "grf_yawbase", "external_force", "base_lin_vel",
    }


def test_the_grf_target_is_body_frame_and_the_world_one_is_deprecated():
    """
    A world-frame GRF cannot be the target of a symmetry-equivariant model.

    It is not equivariant under the robot's morphological symmetry group, so
    every yaw-invariance argument an MI-HGNN / ECNN-style method makes breaks on
    it. The instantaneous world-frame column is also aliased (0 N on 1.9% of
    in-contact frames), which is a separate reason not to regress against it.
    """
    from mpx.utils.dataset_collection.dataset_schema import COLUMNS_BY_NAME

    assert COLUMNS_BY_NAME["grf_base"].role == "target"
    assert COLUMNS_BY_NAME["grf_base"].frame == "base"
    assert COLUMNS_BY_NAME["grf_yawbase"].frame == "yaw_base"
    assert COLUMNS_BY_NAME["grf_world"].role == "deprecated"
    assert COLUMNS_BY_NAME["grf_mean_world"].role == "privileged"


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

def test_contact_labels_come_from_the_debouncer_not_a_second_threshold():
    """
    There is exactly one definition of contact, and it is not in this module.

    ``contact_bits_from_grf`` was a plain ``|GRF| > 5 N`` rule that production
    never used — the recorder runs the debouncer — but it stayed reachable, kept
    a third contact threshold alive in the config, and labelled the synthetic
    test fixtures, so those fixtures disagreed with production on 8.6% of rows.
    """
    import mpx.utils.dataset_collection.dataset_schema as schema

    assert not hasattr(schema, "contact_bits_from_grf")

    # The surviving path: substep force -> Schmitt trigger -> minimum dwell.
    force = np.zeros((40, 4))
    force[:, 0] = 40.0                        # FL loaded throughout
    force[10:30, 1] = 40.0                    # FR loaded in the middle
    bits = debounce_sequence(force)
    assert bits.shape == (40, 4)
    np.testing.assert_array_equal(bits[:, 0], np.ones(40, dtype=np.uint8))
    assert bits[20, 1] == 1 and bits[39, 1] == 0


def test_single_foot_stance_gets_its_own_name():
    """v4 names all 16 patterns, so no two share a label."""
    assert contact_state_name((1, 1, 1, 1)) == "FULL"
    assert contact_state_name((1, 0, 0, 1)) == "DIAG_FL_RR"
    expected = {
        (1, 0, 0, 0): "SINGLE_FL",
        (0, 1, 0, 0): "SINGLE_FR",
        (0, 0, 1, 0): "SINGLE_RL",
        (0, 0, 0, 1): "SINGLE_RR",
    }
    for pattern, name in expected.items():
        assert contact_state_name(pattern) == name
        assert is_rare_contact(pattern)
    assert not is_rare_contact((1, 1, 1, 1))


def test_every_bit_pattern_has_a_unique_name():
    names = [
        contact_state_name((a, b, c, d))
        for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)
    ]
    assert len(set(names)) == 16


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

def test_split_is_deterministic_for_a_group_id():
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
