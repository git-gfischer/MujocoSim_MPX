"""Tests for episode boundaries, outcomes and the derived per-step labels."""

from __future__ import annotations

import numpy as np
import pytest

from mpx.config.sim_config.config_dataset_bucket import DatasetCollectionConfig
from mpx.utils.dataset_collection.dataset_bucket_system import (
    DatasetBucketSystem,
    GaitType,
    TerrainType,
)
from mpx.utils.dataset_collection.dataset_bucket_system import is_rare_contact
from mpx.utils.dataset_collection.dataset_schema import (
    EPISODE_COLUMNS,
    EpisodeMetadata,
    EpisodeRecord,
)
from mpx.utils.dataset_collection.episode_recorder import (
    EpisodeRecorder,
    EpisodeRecorderConfig,
    _ActiveCollectionHooks,
    scene_to_terrain,
)

CONTROL_HZ = 50.0


def make_recorder(
    *,
    store_failed_episodes: bool = True,
    min_episode_duration_s: float = 0.0,
    episode_duration_s: float = 10.0,
) -> EpisodeRecorder:
    bucket = DatasetBucketSystem(bucket_capacity=100, dataset_summary_path=None)
    return EpisodeRecorder(
        bucket,
        gait_type=GaitType.TROT,
        terrain_type=TerrainType.FLAT,
        sim_hz=CONTROL_HZ,
        config=EpisodeRecorderConfig(
            control_hz=CONTROL_HZ,
            episode_duration_s=episode_duration_s,
            min_episode_duration_s=min_episode_duration_s,
            store_failed_episodes=store_failed_episodes,
        ),
        episode_prefix="run_x",
        robot="go2",
        scene="flat",
    )


def feed(recorder: EpisodeRecorder, stances) -> None:
    """
    Push one buffered row per entry of ``stances`` (each a 4-foot pattern).

    Writes every v4 column so ``_build_arrays`` sees a complete table, which is
    what the recorder would have produced from a real step.
    """
    for stance in stances:
        loaded = np.asarray(stance, dtype=np.float64) * 50.0
        grf = np.zeros((4, 3), dtype=np.float32)
        grf[:, 2] = loaded

        row = {c.name: np.zeros(c.shape, dtype=c.dtype) for c in EPISODE_COLUMNS}
        row["contact"] = np.asarray(stance, dtype=np.uint8)
        row["contact_raw"] = np.asarray(stance, dtype=np.uint8)
        row["grf_world"] = grf.reshape(-1)
        row["grf_mean_n"] = loaded.astype(np.float32)
        row["grf_max_n"] = loaded.astype(np.float32)
        # Unit quaternion: the post-failure predicate reads this.
        row["base_quat"] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        row["base_height_terrain"] = np.float32(0.30)
        row["rare_contact"] = np.bool_(is_rare_contact(stance))

        for name, value in row.items():
            recorder._buffer[name].append(value)
        recorder._buffer["t"][-1] = np.int32(recorder._control_step)
        recorder._buffer["time_s"][-1] = np.float32(
            recorder._control_step / CONTROL_HZ
        )
        recorder._control_step += 1


# ── outcomes ─────────────────────────────────────────────────────────────────

def test_goal_reached_is_stored_as_a_success():
    recorder = make_recorder()
    feed(recorder, [(1, 1, 1, 1)] * 4)
    assert recorder.end_episode(reason="goal_reached") is True

    episode = next(iter(recorder.bucket.episodes.values()))
    assert episode.terminate_by == "success"
    assert episode.terminate_reason == "goal_reached"


def test_crash_is_stored_as_a_failure_by_default():
    recorder = make_recorder()
    feed(recorder, [(1, 1, 1, 1)] * 4)
    assert recorder.end_episode(reason="crash") is True
    assert recorder.bucket.episode_outcome_counts() == {"failure": 1}


def test_failures_can_be_dropped_by_config():
    recorder = make_recorder(store_failed_episodes=False)
    feed(recorder, [(1, 1, 1, 1)] * 4)
    assert recorder.end_episode(reason="crash") is False
    assert recorder.bucket.episodes == {}
    assert recorder.episodes_discarded == 1


def test_duration_cap_truncates_rather_than_judging():
    recorder = make_recorder(episode_duration_s=4 / CONTROL_HZ)
    feed(recorder, [(1, 1, 1, 1)] * 4)
    assert recorder.end_episode(reason="duration_cap") is True
    assert recorder.bucket.episode_outcome_counts() == {"truncated": 1}


def test_manual_respawn_discards_while_a_crash_stores(tmp_path):
    recorder = make_recorder()
    hooks = _ActiveCollectionHooks(
        recorder, run_dir=str(tmp_path), profile=DatasetCollectionConfig(), metadata={}
    )

    feed(recorder, [(1, 1, 1, 1)] * 4)
    hooks.on_respawn(manual=True)
    assert recorder.bucket.episodes == {}

    feed(recorder, [(1, 1, 1, 1)] * 4)
    hooks.on_respawn(crashed=True)
    assert recorder.bucket.episode_outcome_counts() == {"failure": 1}


def test_episode_shorter_than_the_minimum_duration_is_dropped():
    recorder = make_recorder(min_episode_duration_s=1.0)   # 50 steps @ 50 Hz
    feed(recorder, [(1, 1, 1, 1)] * 4)
    assert recorder.end_episode(reason="goal_reached") is False
    assert recorder.bucket.episodes == {}


def test_every_timestep_is_indexed_including_the_first():
    """Collection stores whole trajectories; W is a training-time choice."""
    recorder = make_recorder()
    feed(recorder, [(1, 1, 1, 1)] * 6)
    record = EpisodeRecord(
        metadata=EpisodeMetadata(
            episode_id="ep_all", gait="trot", terrain="flat", control_hz=CONTROL_HZ
        ),
        arrays=recorder._build_arrays(),
    )

    result = recorder.bucket.add_episode(record)
    assert result["added"] == 6
    indexed = {row["t"] for row in recorder.bucket.index_rows(shuffle=False)}
    assert indexed == set(range(6))


# ── derived per-step labels ──────────────────────────────────────────────────

def test_recorded_episode_carries_contact_and_dt_columns():
    recorder = make_recorder()
    feed(recorder, [
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (0, 1, 1, 1),
        (0, 1, 1, 1),
    ])
    arrays = recorder._build_arrays()

    np.testing.assert_array_equal(arrays["contact"][0], [1, 1, 1, 1])
    np.testing.assert_array_equal(arrays["contact"][2], [0, 1, 1, 1])
    # FL's timer resets at the liftoff on step 2, FR's keeps running.
    assert arrays["dt_since_transition"][2, 0] == pytest.approx(0.0)
    assert arrays["dt_since_transition"][2, 1] == pytest.approx(2 / CONTROL_HZ)
    assert not arrays["rare_contact"].any()


def test_single_foot_stance_sets_the_rare_flag():
    recorder = make_recorder()
    feed(recorder, [(1, 0, 0, 0)] * 3)
    arrays = recorder._build_arrays()
    assert arrays["rare_contact"].all()


def test_time_column_follows_the_control_rate():
    recorder = make_recorder()
    feed(recorder, [(1, 1, 1, 1)] * 3)
    arrays = recorder._build_arrays()
    np.testing.assert_allclose(arrays["time_s"], [0.0, 0.02, 0.04])
    np.testing.assert_array_equal(arrays["t"], [0, 1, 2])


# ── episode identity and split ───────────────────────────────────────────────

def test_episode_ids_are_prefixed_by_the_run_and_numbered():
    recorder = make_recorder()
    for _ in range(2):
        feed(recorder, [(1, 1, 1, 1)] * 4)
        recorder.end_episode(reason="goal_reached")
    assert recorder.bucket.episodes_collected == ["run_x_00001", "run_x_00002"]


def test_every_episode_gets_a_split():
    recorder = make_recorder()
    feed(recorder, [(1, 1, 1, 1)] * 4)
    recorder.end_episode(reason="goal_reached")
    episode = next(iter(recorder.bucket.episodes.values()))
    assert episode.split_assigned in ("train", "val", "test")


# ── scene mapping ────────────────────────────────────────────────────────────

def test_scene_names_map_to_terrain_types():
    assert scene_to_terrain("flat") is TerrainType.FLAT
    assert scene_to_terrain("slippery") is TerrainType.FLAT
    assert scene_to_terrain("stairs") is TerrainType.STAIRS
    assert scene_to_terrain("rough") is TerrainType.ROUGH
