"""One pickle per proprioceptive episode. The file is the time series and nothing else."""

from __future__ import annotations

import argparse
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

_LEGS = ("LF", "RF", "LH", "RH")
_JOINTS = ("HAA", "HFE", "KFE")
_AXES = ("X", "Y", "Z")
COLLECT_PARQUET = "parquet"
COLLECT_PICKLE = "pickle"


def default_pickle_output_root() -> Path:
    """``<repo>/pickle_datasets``, sibling to the parquet ``datasets`` root."""
    return Path(__file__).resolve().parents[3] / "pickle_datasets"


def new_pickle_run_dir() -> Path:
    """A new run folder. Does not create it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return default_pickle_output_root() / stamp


def add_collect_argument(parser: argparse.ArgumentParser) -> None:
    """``--collect`` writes parquet. ``--collect pickle`` writes episode pickles."""
    parser.add_argument(
        "--collect",
        nargs="?",
        const=COLLECT_PARQUET,
        default=None,
        choices=(COLLECT_PARQUET, COLLECT_PICKLE),
        help=(
            "Collect a dataset. --collect and --collect parquet write the "
            "bucketed parquet run. --collect pickle writes one proprioceptive "
            "pickle per episode under pickle_datasets/."
        ),
    )


def sample_to_proprioceptive(sample) -> dict:
    """One onboard sample as the nested deque tree ``EpisodePickleWriter`` stores."""
    tree = {}
    series = {
        "JointPos": sample.joint_pos,
        "JointVel": sample.joint_vel,
        "Torque": sample.joint_torque,
        "FootPos": sample.foot_pos_yawbase,
        "FootVel": sample.foot_vel_yawbase,
    }
    for name, values in series.items():
        legs = {}
        for leg_i, leg in enumerate(_LEGS):
            sl = slice(3 * leg_i, 3 * leg_i + 3)
            legs[leg] = {
                joint: deque([float(values[sl][joint_i])])
                for joint_i, joint in enumerate(_JOINTS)
            }
        tree[name] = legs
    tree["Lin_acc"] = {
        axis: deque([float(sample.imu_acc_body[i])]) for i, axis in enumerate(_AXES)
    }
    tree["Ang_vel"] = {
        axis: deque([float(sample.imu_gyro_body[i])]) for i, axis in enumerate(_AXES)
    }
    return tree


class EpisodePickleWriter:
    def __init__(self, directory: str | Path, enable: bool = False):
        self.enable = bool(enable)
        self.directory = Path(directory)
        self._buffer: dict = {}
        self._n = 0
        self._episode = 0
        if self.enable:
            self.directory.mkdir(parents=True, exist_ok=True)

    def append(self, proprioceptive_data: dict, labels=None) -> None:
        if not self.enable or not proprioceptive_data:
            return
        if _snapshot(proprioceptive_data, self._buffer):
            self._n += 1

    def save(self) -> None:
        if not self.enable or self._n == 0:
            return
        self._episode += 1
        path = self.directory / f"{self._episode:05d}.pkl"
        with open(path, "wb") as handle:
            pickle.dump(_lists_to_arrays(self._buffer), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[EpisodePickleWriter] Wrote {path}")
        self._buffer = {}
        self._n = 0

    def close(self) -> None:
        self.save()


def _snapshot(node: dict, buf: dict) -> bool:
    """Copy the last value of every non-empty deque. Return whether any value was stored."""
    added = False
    for key, value in node.items():
        if isinstance(value, dict):
            child = buf.setdefault(key, {})
            if _snapshot(value, child):
                added = True
            elif not child:
                buf.pop(key, None)
        elif isinstance(value, deque) and value:
            buf.setdefault(key, []).append(np.float32(value[-1]))
            added = True
    return added


class PickleSimCollector:
    """Record onboard proprioception for ``--collect pickle``."""

    def __init__(self, model, contact_ids, n_joints: int, sim_hz: float, episode_duration_s: float):
        from mpx.utils.simulation_utils.onboard_proprioception import OnboardProprioception

        self.writer = EpisodePickleWriter(new_pickle_run_dir(), enable=True)
        self.observer = OnboardProprioception.from_contact_ids(
            model, contact_ids, n_joints=int(n_joints), dt=1.0 / float(sim_hz),
        )
        self._limit = max(1, int(round(float(episode_duration_s) * float(sim_hz))))
        self._steps = 0

    @property
    def seconds(self) -> float:
        return self._steps * self.observer.dt

    def step(self, data) -> bool:
        """Append one tick. Return True when the duration cap closed the episode."""
        self.writer.append(sample_to_proprioceptive(self.observer.sample(data)))
        self._steps += 1
        if self._steps < self._limit:
            return False
        self.end_episode()
        return True

    def end_episode(self) -> None:
        self.writer.save()
        self.observer.reset()
        self._steps = 0

    def close(self) -> None:
        self.writer.close()


def _lists_to_arrays(node: dict) -> dict:
    out = {}
    for key, value in node.items():
        if isinstance(value, dict):
            child = _lists_to_arrays(value)
            if child:
                out[key] = child
        elif isinstance(value, list) and value:
            out[key] = np.asarray(value, dtype=np.float32)
    return out
