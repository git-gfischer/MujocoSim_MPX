"""Collection logs one row per physics step; control-side 50 Hz timing is unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from mpx.utils.dataset_collection.contact_labeling import (
    ContactLabelConfig,
    debounce_sequence,
)

from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config
from mpx.utils.dataset_collection.dataset_bucket_system import GaitType, TerrainType
from mpx.utils.dataset_collection.episode_recorder import (
    EpisodeRecorder,
    EpisodeRecorderConfig,
    create_collection_session,
    setup_sim_collection,
)
from mpx.utils.dataset_collection.test_episode_recorder import CONTROL_HZ, make_recorder
from mpx.utils.simulation_utils.sensor_noise import SensorNoise


def test_joint_latency_stays_20ms_when_sample_dt_changes():
    n50 = SensorNoise.from_config(dt=0.02)
    n500 = SensorNoise.from_config(dt=0.002)
    assert n50.latency_steps.joint == 1
    assert n500.latency_steps.joint == 10
    assert n50.latency_steps.imu == 0
    assert n500.latency_steps.imu == 0


def test_collection_session_logs_every_physics_step():
    rec = create_collection_session(
        gait_type=GaitType.TROT,
        terrain_type=TerrainType.FLAT,
        sim_hz=500.0,
        episode_prefix="rate_test",
    )
    assert rec.config.control_hz == 500.0
    assert rec.config.sim_hz == 500.0
    assert rec._decim == 1
    assert rec.config.contact_labeling.min_dwell_steps == 30
    assert rec.config.contact_labeling.schmitt_average_steps == 10
    assert rec.config.operating_regime.dwell_steps == 30
    assert rec.bucket.config.mismatch_edge_radius == 20
    assert rec.bucket.config.mismatch_sustained_min_run == 30
    assert rec.bucket.collapse_dwell_steps == 250
    assert rec.sensor_noise.dt == pytest.approx(0.002)
    assert rec.sensor_noise.latency_steps.joint == 10
    assert dataset_collection_config.episode.control_hz == 50.0
    assert dataset_collection_config.rates.control_hz == 50.0
    assert dataset_collection_config.contact_labeling.min_dwell_steps == 3


def test_fifty_hz_recorder_tests_are_unchanged():
    rec = make_recorder()
    assert rec.config.control_hz == CONTROL_HZ
    assert rec._decim == 1
    assert rec.config.contact_labeling.min_dwell_steps == 3


def test_old_decimation_still_computes_when_rates_differ():
    rec = EpisodeRecorder(
        make_recorder().bucket,
        gait_type=GaitType.TROT,
        terrain_type=TerrainType.FLAT,
        sim_hz=500.0,
        config=EpisodeRecorderConfig(control_hz=50.0, sim_hz=500.0),
        episode_prefix="decim",
    )
    assert rec._decim == 10


def test_setup_stamps_scaled_contact_dwell(tmp_path):
    hooks = setup_sim_collection(
        True,
        gait_type=GaitType.TROT,
        scene="flat",
        sim_hz=500.0,
        robot="go2",
        collect_out=str(tmp_path / "run"),
        register_atexit=False,
    )
    assert hooks._metadata["contact_labeling"]["min_dwell_steps"] == 30
    assert hooks._metadata["contact_labeling"]["schmitt_average_steps"] == 10
    assert dataset_collection_config.contact_labeling.min_dwell_steps == 3
    assert dataset_collection_config.contact_labeling.schmitt_average_steps == 1


def test_one_physics_step_dropout_does_not_start_a_liftoff():
    """A 2 ms zero must not lift a loaded foot when the Schmitt sees 20 ms.

    At 500 Hz the label used to compare one physics sample to the 5 N rail.
    That sample is often 0 N while the foot is still down; the 60 ms dwell
    then held the false swing, and crawl's 80% stance plan disagreed on most
    rows.
    """
    force = np.full((80, 4), 60.0)
    force[40, 0] = 0.0
    bits = debounce_sequence(
        force,
        ContactLabelConfig(min_dwell_steps=30, schmitt_average_steps=10),
    )
    assert bits[:, 0].min() == 1


def test_a_real_unload_still_lifts_off():
    force = np.full((80, 4), 60.0)
    force[40:, 0] = 0.0
    bits = debounce_sequence(
        force,
        ContactLabelConfig(min_dwell_steps=30, schmitt_average_steps=10),
    )
    assert bits[39, 0] == 1
    assert bits[55, 0] == 0
    assert bits[-1, 0] == 0
