"""Round-trip tests for the parquet episode store and the balanced label index."""

from __future__ import annotations

import numpy as np
import pytest

from mpx.utils.dataset_collection.dataset_bucket_system import (
    DatasetBucketSystem,
    GaitType,
    TerrainType,
)
from mpx.utils.dataset_collection.dataset_schema import (
    EPISODE_COLUMNS,
    EpisodeMetadata,
    EpisodeRecord,
    dt_since_transition,
    empty_arrays,
)
from mpx.utils.dataset_collection.episode_storage import (
    EpisodeStore,
    pyarrow_available,
)

pytestmark = pytest.mark.skipif(
    not pyarrow_available(), reason="parquet storage needs pyarrow"
)

CONTROL_HZ = 50.0


def make_record(
    episode_id: str = "ep_00001",
    n_steps: int = 60,
    *,
    stance=(1, 1, 1, 1),
    ext_force: float = 0.0,
    terrain: TerrainType = TerrainType.FLAT,
    gait: GaitType = GaitType.TROT,
    terminate_by: str = "success",
    split: str = "train",
    seed: int = 0,
) -> EpisodeRecord:
    """One synthetic episode with a constant contact pattern."""
    rng = np.random.default_rng(seed)
    arrays = empty_arrays(n_steps)
    arrays["t"][:] = np.arange(n_steps, dtype=np.int32)
    arrays["time_s"][:] = np.arange(n_steps, dtype=np.float32) / CONTROL_HZ
    for name in ("joint_pos", "joint_vel", "joint_torque", "imu_acc", "imu_gyro",
                 "foot_pos_base", "foot_vel_base", "base_lin_vel"):
        arrays[name][:] = rng.standard_normal(arrays[name].shape).astype(np.float32)

    grf = np.zeros((n_steps, 4, 3), dtype=np.float32)
    for foot, loaded in enumerate(stance):
        if loaded:
            grf[:, foot, 2] = 50.0
    arrays["grf_world"][:] = grf.reshape(n_steps, 12)
    arrays["external_force"][:, 0] = ext_force

    contact = np.tile(np.asarray(stance, dtype=np.uint8), (n_steps, 1))
    arrays["contact"][:] = contact
    arrays["dt_since_transition"][:] = dt_since_transition(contact, 1.0 / CONTROL_HZ)

    metadata = EpisodeMetadata(
        episode_id=episode_id,
        robot="go2",
        scene="flat",
        terrain=terrain.value,
        gait=gait.value,
        timestamp="2026-09-14T10:00:00+02:00",
        ended_at="2026-09-14T10:01:00+02:00",
        control_hz=CONTROL_HZ,
        friction=0.65,
        payload_kg=1.25,
        terminate_by=terminate_by,
        terminate_reason="goal_reached",
        split=split,
        reset_randomization={"step_freq": 1.4},
    )
    return EpisodeRecord(metadata=metadata, arrays=arrays)


# ── per-timestep episode table ───────────────────────────────────────────────

def test_episode_round_trips_every_column(tmp_path):
    store = EpisodeStore(tmp_path)
    record = make_record()
    path = store.write_episode(record)

    assert path.exists()
    restored = store.read_episode(path)
    for column in EPISODE_COLUMNS:
        np.testing.assert_allclose(
            np.asarray(restored[column.name], dtype=np.float64),
            np.asarray(record.arrays[column.name], dtype=np.float64),
            err_msg=f"column {column.name} did not round-trip",
        )


def test_episode_metadata_travels_with_the_file(tmp_path):
    store = EpisodeStore(tmp_path)
    store.write_episode(make_record())

    metadata = store.read_episode_metadata(store.episode_path("ep_00001"))
    assert metadata["terrain"] == "flat"
    assert metadata["gait"] == "trot"
    assert metadata["friction"] == pytest.approx(0.65)
    assert metadata["payload_kg"] == pytest.approx(1.25)
    assert metadata["terminate_by"] == "success"
    assert metadata["timestamp"] == "2026-09-14T10:00:00+02:00"


def test_window_slice_has_the_requested_length(tmp_path):
    store = EpisodeStore(tmp_path)
    store.write_episode(make_record(n_steps=60))

    window = store.window("ep_00001", t=41, window_size=30)
    assert window["joint_pos"].shape == (30, 12)
    np.testing.assert_array_equal(window["t"], np.arange(12, 42))


def test_the_same_episode_serves_any_window_length(tmp_path):
    """Window length is the caller's choice, not a property of the stored data."""
    store = EpisodeStore(tmp_path)
    store.write_episode(make_record(n_steps=60))

    for window_size in (5, 30, 50):
        window = store.window("ep_00001", t=55, window_size=window_size)
        assert window["joint_pos"].shape == (window_size, 12)
        assert int(window["t"][-1]) == 55


def test_window_before_the_episode_start_is_rejected(tmp_path):
    store = EpisodeStore(tmp_path)
    store.write_episode(make_record(n_steps=60))
    with pytest.raises(ValueError, match="before the episode begins"):
        store.window("ep_00001", t=5, window_size=30)


# ── episode + index tables ───────────────────────────────────────────────────

def test_episode_table_round_trips_metadata(tmp_path):
    store = EpisodeStore(tmp_path)
    records = [make_record("ep_a"), make_record("ep_b", terminate_by="failure")]
    store.write_episode_table([r.metadata for r in records])

    rows = {row["episode_id"]: row for row in store.read_episode_table()}
    assert set(rows) == {"ep_a", "ep_b"}
    assert rows["ep_b"]["terminate_by"] == "failure"
    assert rows["ep_a"]["reset_randomization"] == {"step_freq": 1.4}


def test_collection_writes_the_three_tables(tmp_path):
    store = EpisodeStore(tmp_path)
    bucket = DatasetBucketSystem(
        bucket_capacity=1_000, dataset_summary_path=None, store=store
    )
    bucket.add_episode(make_record("ep_00001", n_steps=60))
    bucket.add_episode(
        make_record("ep_00002", n_steps=60, stance=(1, 0, 0, 1), ext_force=40.0)
    )
    index_path = bucket.save_dataset(tmp_path, metadata={"robot": "go2"})

    assert index_path.exists()
    assert store.episode_table_path.exists()
    assert store.episode_path("ep_00001").exists()
    assert store.episode_path("ep_00002").exists()
    assert (tmp_path / "run_metadata.json").exists()

    index = store.read_index()
    assert len(index) == bucket.total_samples_stored
    states = {row["contact_state"] for row in index}
    assert states == {"FULL", "DIAG_FL_RR"}


def test_index_carries_no_window_length(tmp_path):
    """The index addresses label timesteps; W is chosen downstream."""
    store = EpisodeStore(tmp_path)
    bucket = DatasetBucketSystem(
        bucket_capacity=1_000, dataset_summary_path=None, store=store
    )
    bucket.add_episode(make_record("ep_00001", n_steps=60))
    bucket.save_dataset(tmp_path)

    index = store.read_index()
    assert "window_size" not in index[0]
    assert "t_start" not in index[0]
    # Every timestep is indexable, including the first.
    assert {row["t"] for row in index} == set(range(60))


def test_index_rows_point_at_readable_windows(tmp_path):
    store = EpisodeStore(tmp_path)
    bucket = DatasetBucketSystem(
        bucket_capacity=1_000, dataset_summary_path=None, store=store
    )
    bucket.add_episode(make_record("ep_00001", n_steps=60))
    bucket.save_dataset(tmp_path)

    window_size = 30
    usable = [row for row in store.read_index() if row["t"] >= window_size - 1]
    assert usable
    for row in usable[:5]:
        window = store.window(row["episode_id"], row["t"], window_size)
        assert window["joint_pos"].shape == (window_size, 12)
        assert int(window["t"][-1]) == row["t"]


def test_perturbation_flag_follows_the_external_force(tmp_path):
    store = EpisodeStore(tmp_path)
    bucket = DatasetBucketSystem(
        bucket_capacity=1_000,
        perturbation_force_threshold=5.0,
        dataset_summary_path=None,
        store=store,
    )
    bucket.add_episode(make_record("ep_quiet", n_steps=60, ext_force=0.0))
    bucket.add_episode(make_record("ep_pushed", n_steps=60, ext_force=40.0))
    bucket.save_dataset(tmp_path)

    by_episode = {}
    for row in store.read_index():
        by_episode.setdefault(row["episode_id"], set()).add(row["perturbation_active"])
    assert by_episode["ep_quiet"] == {False}
    assert by_episode["ep_pushed"] == {True}


# ── episode-level split ──────────────────────────────────────────────────────

def test_windows_of_one_episode_never_straddle_a_split(tmp_path):
    store = EpisodeStore(tmp_path)
    bucket = DatasetBucketSystem(
        bucket_capacity=1_000, dataset_summary_path=None, store=store
    )
    bucket.add_episode(make_record("ep_train", n_steps=80, split="train"))
    bucket.add_episode(make_record("ep_test", n_steps=80, split="test"))
    bucket.save_dataset(tmp_path)

    splits_per_episode = {}
    for row in store.read_index():
        splits_per_episode.setdefault(row["episode_id"], set()).add(row["split"])
    assert splits_per_episode == {"ep_train": {"train"}, "ep_test": {"test"}}


def test_split_index_rows_groups_by_split(tmp_path):
    bucket = DatasetBucketSystem(
        bucket_capacity=1_000, dataset_summary_path=None
    )
    bucket.add_episode(make_record("ep_train", n_steps=80, split="train"))
    bucket.add_episode(make_record("ep_val", n_steps=80, split="val"))

    grouped = bucket.split_index_rows()
    assert set(grouped) == {"train", "val"}
    assert all(r["episode_id"] == "ep_train" for r in grouped["train"])
    assert all(r["episode_id"] == "ep_val" for r in grouped["val"])


# ── rare contacts ────────────────────────────────────────────────────────────

def test_single_foot_samples_are_kept_and_flagged(tmp_path):
    store = EpisodeStore(tmp_path)
    bucket = DatasetBucketSystem(
        bucket_capacity=1_000, dataset_summary_path=None, store=store
    )
    result = bucket.add_episode(make_record("ep_rare", n_steps=60, stance=(1, 0, 0, 0)))
    bucket.save_dataset(tmp_path)

    assert result["added"] == 60
    assert result["rare"] == 60
    index = store.read_index()
    assert all(row["rare_contact"] for row in index)
    assert all(row["contact_state"] == "RARE" for row in index)
    assert all(row["contact_bits"] == "1000" for row in index)


# ── episode registry ─────────────────────────────────────────────────────────

def test_empty_episode_is_rejected():
    bucket = DatasetBucketSystem(bucket_capacity=10, dataset_summary_path=None)
    with pytest.raises(ValueError, match="no timesteps"):
        bucket.add_episode(make_record("ep_empty", n_steps=0))


def test_duplicate_episode_id_is_rejected():
    bucket = DatasetBucketSystem(bucket_capacity=10, dataset_summary_path=None)
    bucket.add_episode(make_record("ep_dup", n_steps=60))
    with pytest.raises(ValueError, match="already added"):
        bucket.add_episode(make_record("ep_dup", n_steps=60))


# ── persistent dataset memory ────────────────────────────────────────────────

def test_dataset_summary_records_the_run(tmp_path):
    run_dir = tmp_path / "run_a"
    store = EpisodeStore(run_dir)
    bucket = DatasetBucketSystem(
        bucket_capacity=1_000,
        dataset_summary_path=tmp_path / "dataset_summary.json",
        store=store,
    )
    bucket.add_episode(make_record("ep_00001", n_steps=60, terminate_by="success"))
    bucket.add_episode(
        make_record("ep_00002", n_steps=60, stance=(1, 1, 1, 0), terminate_by="failure")
    )
    index_path = bucket.save_dataset(run_dir, metadata={"robot": "go2"})
    summary_path = bucket.update_dataset_summary(index_path, run_dir=run_dir)

    assert summary_path.exists()
    whole = bucket.dataset_memory["summary"]
    assert whole["episodes"] == 2
    assert whole["samples"] == bucket.total_samples_stored
    assert whole["terminate_by_counts"] == {"failure": 1, "success": 1}
    assert set(whole["split_counts"]) <= {"train", "val", "test"}


def test_dataset_summary_bootstraps_from_an_existing_run(tmp_path):
    run_dir = tmp_path / "run_a"
    store = EpisodeStore(run_dir)
    first = DatasetBucketSystem(
        bucket_capacity=1_000, dataset_summary_path=None, store=store
    )
    first.add_episode(make_record("ep_00001", n_steps=60))
    first.save_dataset(run_dir, metadata={"robot": "go2"})

    # A fresh session with no memory file must find the run already on disk.
    second = DatasetBucketSystem(
        bucket_capacity=1_000,
        dataset_summary_path=tmp_path / "dataset_summary.json",
    )
    assert second.dataset_memory["summary"]["dataset_files"] == 1
    assert second.dataset_memory["summary"]["samples"] == first.total_samples_stored


def test_old_schema_summary_is_archived_not_overwritten(tmp_path):
    summary_path = tmp_path / "dataset_summary.json"
    summary_path.write_text('{"schema_version": 2, "datasets": {}}', encoding="utf-8")

    DatasetBucketSystem(dataset_summary_path=summary_path)

    archived = tmp_path / "dataset_summary.v2.json"
    assert archived.exists()
    assert "schema_version" in archived.read_text(encoding="utf-8")
