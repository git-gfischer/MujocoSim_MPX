"""One pickle per proprioceptive episode, data only."""

from __future__ import annotations

import argparse
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from mpx.utils.dataset_collection.dataset_pickle import (
    EpisodePickleWriter,
    add_collect_argument,
    default_pickle_output_root,
    new_pickle_run_dir,
    sample_to_proprioceptive,
)


def _sample():
    return {
        "JointPos": {"LF": {"HAA": deque(maxlen=4)}},
        "Lin_acc": {"X": deque(maxlen=4)},
    }


def _load(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def test_run_dir_is_under_pickle_datasets():
    path = new_pickle_run_dir()
    assert path.parent == default_pickle_output_root()
    assert path.parent.name == "pickle_datasets"
    assert len(path.name) == len("20261012_135501_123456")
    assert not path.exists()


def test_two_ticks_write_one_episode(tmp_path):
    sample = _sample()
    writer = EpisodePickleWriter(tmp_path, enable=True)
    sample["JointPos"]["LF"]["HAA"].append(0.1)
    sample["Lin_acc"]["X"].append(1.0)
    writer.append(sample, labels=[1, 0, 0, 0])
    sample["JointPos"]["LF"]["HAA"].append(0.2)
    sample["Lin_acc"]["X"].append(2.0)
    writer.append(sample, labels=[0, 1, 0, 0])
    writer.save()

    payload = _load(tmp_path / "00001.pkl")
    haa = payload["JointPos"]["LF"]["HAA"]
    acc = payload["Lin_acc"]["X"]
    assert haa.dtype == np.float32 and haa.shape == (2,)
    assert acc.dtype == np.float32 and acc.shape == (2,)
    assert haa.tolist() == pytest.approx([0.1, 0.2])
    assert acc.tolist() == pytest.approx([1.0, 2.0])
    assert "labels" not in payload

    writer.save()
    assert not (tmp_path / "00002.pkl").exists()


def test_empty_deque_is_absent(tmp_path):
    sample = {
        "JointPos": {"LF": {"HAA": deque([0.5]), "HFE": deque()}},
        "Lin_acc": {"X": deque()},
    }
    writer = EpisodePickleWriter(tmp_path, enable=True)
    writer.append(sample)
    writer.save()
    payload = _load(tmp_path / "00001.pkl")
    assert "HFE" not in payload["JointPos"]["LF"]
    assert "Lin_acc" not in payload


def test_stale_deque_repeats_last_value(tmp_path):
    sample = {"JointPos": {"LF": {"HAA": deque([1.0])}}}
    writer = EpisodePickleWriter(tmp_path, enable=True)
    writer.append(sample)
    writer.append(sample)
    writer.save()
    payload = _load(tmp_path / "00001.pkl")
    assert payload["JointPos"]["LF"]["HAA"].tolist() == pytest.approx([1.0, 1.0])


def test_disabled_writes_nothing(tmp_path):
    dest = tmp_path / "run"
    writer = EpisodePickleWriter(dest, enable=False)
    writer.append({"JointPos": {"LF": {"HAA": deque([1.0])}}})
    writer.save()
    writer.close()
    assert not dest.exists()


def test_collect_flag_parquet_is_default_and_pickle_is_a_choice():
    parser = argparse.ArgumentParser()
    add_collect_argument(parser)
    assert parser.parse_args([]).collect is None
    assert parser.parse_args(["--collect"]).collect == "parquet"
    assert parser.parse_args(["--collect", "pickle"]).collect == "pickle"


def test_onboard_sample_becomes_one_tick_tree():
    from types import SimpleNamespace

    sample = SimpleNamespace(
        joint_pos=np.arange(12, dtype=float),
        joint_vel=np.arange(12, dtype=float) + 0.5,
        joint_torque=np.arange(12, dtype=float) + 1.0,
        foot_pos_yawbase=np.arange(12, dtype=float) + 2.0,
        foot_vel_yawbase=np.arange(12, dtype=float) + 3.0,
        imu_acc_body=np.array([0.1, 0.2, 0.3]),
        imu_gyro_body=np.array([0.4, 0.5, 0.6]),
    )
    tree = sample_to_proprioceptive(sample)
    assert list(tree["JointPos"]["LF"]["HAA"]) == [0.0]
    assert list(tree["JointPos"]["RH"]["KFE"]) == [11.0]
    assert list(tree["Lin_acc"]["Z"]) == pytest.approx([0.3])
    assert list(tree["Ang_vel"]["X"]) == pytest.approx([0.4])
