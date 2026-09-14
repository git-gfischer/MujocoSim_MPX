"""Tests for stamping per-episode conditions onto collected episodes.

Friction and payload are read from the live model at respawn, so they are
recorded whether or not reset randomization is enabled; the sampled knobs
themselves travel alongside them.
"""

from __future__ import annotations

import numpy as np

from mpx.utils.dataset_collection.dataset_bucket_system import (
    DatasetBucketSystem,
    GaitType,
    TerrainType,
)
from mpx.utils.dataset_collection.episode_recorder import (
    EpisodeRecorder,
    EpisodeRecorderConfig,
    SimCollectionHooks,
    _ActiveCollectionHooks,
    _NullCollectionHooks,
)
from mpx.config.sim_config.config_dataset_bucket import DatasetCollectionConfig


def _make_recorder(bucket: DatasetBucketSystem | None = None) -> EpisodeRecorder:
    bucket = bucket or DatasetBucketSystem(
        bucket_capacity=10, dataset_summary_path=None
    )
    return EpisodeRecorder(
        bucket,
        gait_type=GaitType.TROT,
        terrain_type=TerrainType.FLAT,
        sim_hz=50.0,
        config=EpisodeRecorderConfig(
            control_hz=50.0,
            episode_duration_s=10.0,
            min_episode_duration_s=0.0,
        ),
    )


def _feed_full_contact(recorder: EpisodeRecorder, n_steps: int = 3) -> None:
    """Push ``n_steps`` four-foot-stance rows into the recorder's buffer."""
    grf = np.zeros((4, 3), dtype=np.float32)
    grf[:, 2] = 20.0
    for _ in range(n_steps):
        for name in recorder.SAMPLED_COLUMNS:
            if name == "grf_world":
                value = grf.reshape(-1).copy()
            elif name == "external_force":
                value = np.zeros(3, dtype=np.float32)
            elif name in ("joint_pos", "joint_vel", "joint_torque"):
                value = np.zeros(12, dtype=np.float32)
            elif name in ("foot_pos_base", "foot_vel_base"):
                value = np.zeros(12, dtype=np.float32)
            else:
                value = np.zeros(3, dtype=np.float32)
            recorder._buffer[name].append(value)
        recorder._control_step += 1


def test_recorder_stamps_conditions_on_the_episode():
    recorder = _make_recorder()
    meta = {"payload_kg": 4.0, "solref_timeconst": 0.02}
    recorder.set_episode_conditions(friction=0.42, payload_kg=4.0, randomization=meta)
    _feed_full_contact(recorder)

    assert recorder.end_episode(reason="goal_reached") is True
    episode = next(iter(recorder.bucket.episodes.values()))
    assert episode.friction == 0.42
    assert episode.payload_kg == 4.0
    assert episode.reset_randomization == meta


def test_conditions_without_randomization_still_record_friction_and_payload():
    recorder = _make_recorder()
    recorder.set_episode_conditions(friction=1.2, payload_kg=0.0)
    _feed_full_contact(recorder)
    recorder.end_episode(reason="goal_reached")

    episode = next(iter(recorder.bucket.episodes.values()))
    assert episode.friction == 1.2
    assert episode.payload_kg == 0.0
    assert episode.reset_randomization == {}


def test_recorder_keeps_a_map_of_episode_randomization():
    recorder = _make_recorder()
    meta = {"max_speed": 0.4}
    recorder.set_episode_conditions(randomization=meta)
    _feed_full_contact(recorder)
    recorder.end_episode(reason="goal_reached")

    episode_id = next(iter(recorder.episode_randomization))
    assert recorder.episode_randomization[episode_id] == meta


def test_null_hooks_set_episode_conditions_is_noop():
    _NullCollectionHooks().set_episode_conditions(friction=1.0, payload_kg=1.0)


def test_base_hooks_have_set_episode_conditions():
    SimCollectionHooks().set_episode_conditions(randomization={"payload_kg": 1.0})


def test_active_hooks_forward_randomization_into_run_metadata(tmp_path):
    recorder = _make_recorder()
    hooks = _ActiveCollectionHooks(
        recorder,
        run_dir=str(tmp_path),
        profile=DatasetCollectionConfig(),
        metadata={"gait": "trot"},
    )
    meta = {"max_speed": 0.4}
    hooks.set_episode_conditions(friction=0.8, payload_kg=2.0, randomization=meta)
    _feed_full_contact(recorder)
    recorder.end_episode(reason="goal_reached")

    run_meta = hooks._run_metadata()
    assert "episode_randomization" in run_meta
    assert list(run_meta["episode_randomization"].values()) == [meta]
