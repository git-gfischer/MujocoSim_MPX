"""
Episode buffer and routing into :class:`DatasetBucketSystem`.

Records proprioception + ground truth at a fixed control rate (default 50 Hz)
while the simulator runs at a higher rate (e.g. 200 Hz). One row is buffered per
control step; the finished episode is written once as a parquet table, whole and
untrimmed, and each of its labelled timesteps is registered as an
``(episode_id, t)`` reference in the buckets. No window length is involved —
that is the training-time dataset's choice.

An *episode* is one continuous stretch of simulation between two task events, so
its length varies. The simulator decides the boundaries:

* ``quad_locomotion`` closes an episode when the robot reaches its navigation goal.
* ``quad_4balance`` records only while the robot holds the desired pose and closes
  the episode when that pose is lost.

Each closed episode carries an outcome (``terminate_by``): reaching the goal is a
success, falling or losing the commanded pose is a failure, and the duration cap
or shutdown truncates an otherwise healthy episode. Failures are kept by default
(``EpisodeCollectionConfig.store_failed_episodes``) — the steps leading into a
fall are the ones a contact/GRF estimator gets wrong. A manual respawn is not a
task outcome, so it always discards the buffer.

``episode_duration_s`` is only a safety cap in this mode. Set
``EpisodeCollectionConfig.episode_mode = "fixed_duration"`` to go back to closing
every episode on a timer instead.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import mujoco
import numpy as np

from mpx.estimators.quad_contact_estimation import estimate_foot_grf
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation
from mpx.config.sim_config.config_dataset_bucket import (
    DatasetCollectionConfig,
    dataset_collection_config,
)
from mpx.config.sim_config.config_sensor_noise import (
    SensorNoiseConfig,
    sensor_noise_config,
)
from mpx.utils.dataset_collection.dataset_bucket_system import (
    DATASET_SUMMARY_FILENAME,
    RARE_CONTACT_STATE,
    DatasetBucketSystem,
    GaitType,
    TerrainType,
    contact_state_name,
    resolve_run_directory,
)
from mpx.utils.dataset_collection.dataset_schema import (
    EPISODE_COLUMNS,
    EpisodeMetadata,
    EpisodeRecord,
    EpisodeOutcome,
    assign_split,
    classify_termination,
    describe_schema,
    dt_since_transition,
    episode_timestamp,
)
from mpx.utils.dataset_collection.episode_storage import EpisodeStore
from mpx.utils.simulation_utils.sensor_noise import SensorNoise
from mpx.utils.simulation_utils.sim_utils import feet_yaw_base_kinematics

if TYPE_CHECKING:
    from numpy.typing import NDArray


def scene_to_terrain(scene: str) -> TerrainType:
    """Map simulator ``--scene`` name to a :class:`TerrainType`."""
    if scene in ("flat", "slippery"):
        return TerrainType.FLAT
    if scene == "stairs":
        return TerrainType.STAIRS
    return TerrainType.ROUGH


def base_linear_velocity(data: mujoco.MjData) -> NDArray[np.float64]:
    """
    Base linear velocity expressed in the base frame.

    MuJoCo reports a free joint's linear velocity in world coordinates, so it is
    rotated by the transpose of the base orientation.
    """
    rot = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot, np.asarray(data.qpos[3:7], dtype=np.float64))
    return rot.reshape(3, 3).T @ np.asarray(data.qvel[:3], dtype=np.float64)


def read_episode_conditions(
    model: mujoco.MjModel,
    foot_geom_ids,
    base_weight=None,
) -> Dict[str, float]:
    """
    Read the physical conditions in effect for the next episode off the model.

    ``friction`` is the sliding coefficient on the foot geoms (they share one
    value; the mean covers the case where they ever diverge) and ``payload_kg``
    the extra base mass currently applied. Reading the live state rather than the
    sampled reset knobs keeps both populated when randomization is disabled.

    Note that MuJoCo's contact friction is ``min(foot, floor)``; this records the
    foot side, which is the one reset randomization controls.
    """
    ids = np.asarray(foot_geom_ids, dtype=np.int32).reshape(-1)
    return {
        "friction": (
            float(np.mean(model.geom_friction[ids, 0])) if ids.size else None
        ),
        "payload_kg": (
            float(base_weight.extra_mass_kg)
            if base_weight is not None and getattr(base_weight, "enabled", False)
            else 0.0
        ),
    }


def sample_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau: NDArray,
    contact_ids: NDArray,
    n_joints: int,
    sensor_noise: SensorNoise | None = None,
) -> Dict[str, NDArray]:
    """
    Sample one control step into the per-timestep columns of the episode table.

    Joint positions and the IMU channels can be corrupted by ``sensor_noise``;
    everything else (joint velocity, torque, foot kinematics, and every ground
    truth channel) stays exact.
    """
    foot_pos_base, foot_vel_base = feet_yaw_base_kinematics(
        model, data, np.asarray(contact_ids, dtype=np.int32),
    )
    joint_pos = np.asarray(data.qpos[7 : 7 + n_joints], dtype=np.float32)
    imu_acc = np.asarray(data.qacc[:3], dtype=np.float32)
    imu_gyro = np.asarray(data.qvel[3:6], dtype=np.float32)
    if sensor_noise is not None:
        joint_pos, imu_acc, imu_gyro = sensor_noise.apply(
            joint_pos, imu_acc, imu_gyro,
        )
    return {
        "joint_pos": joint_pos,
        "joint_vel": np.asarray(data.qvel[6 : 6 + n_joints], dtype=np.float32),
        "joint_torque": np.asarray(tau, dtype=np.float32).ravel()[:n_joints],
        "imu_acc": imu_acc,
        "imu_gyro": imu_gyro,
        "foot_pos_base": np.asarray(foot_pos_base, dtype=np.float32),
        "foot_vel_base": np.asarray(foot_vel_base, dtype=np.float32),
        "grf_world": np.asarray(
            estimate_foot_grf(model, data, contact_ids), dtype=np.float32
        ).reshape(-1),
        "base_lin_vel": base_linear_velocity(data).astype(np.float32),
    }


@dataclass
class EpisodeRecorderConfig:
    """Timing and naming defaults for on-the-fly dataset collection."""

    control_hz: float = 50.0
    # Event mode: safety cap. Fixed-duration mode: exact episode length.
    episode_duration_s: float = 60.0
    min_episode_duration_s: float = 1.0
    episode_mode: str = "event"
    label_stride: int = 1
    store_failed_episodes: bool = True
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    split_seed: int = 0


class EpisodeRecorder:
    """
    Buffers one variable-length episode at ``control_hz`` and commits it to a bucket system.

    Call :meth:`step_sim` once per MuJoCo step (after ``mj_step``), then close the
    episode from the simulator with :meth:`end_episode` on a task event, or gate
    recording with :meth:`set_recording` when only part of the run is of interest.
    :meth:`discard` drops the buffer without storing it.
    """

    # Buffered columns sampled directly from the simulator each control step.
    # The rest (``t``, ``time_s``, ``contact``, ``rare_contact``,
    # ``dt_since_transition``) are derived when the episode closes.
    SAMPLED_COLUMNS = (
        "joint_pos", "joint_vel", "joint_torque", "imu_acc", "imu_gyro",
        "foot_pos_base", "foot_vel_base", "grf_world", "base_lin_vel",
        "external_force",
    )

    def __init__(
        self,
        bucket_system: DatasetBucketSystem,
        *,
        gait_type: GaitType,
        terrain_type: TerrainType,
        sim_hz: float,
        config: EpisodeRecorderConfig | None = None,
        episode_prefix: str = "ep",
        sensor_noise: SensorNoise | None = None,
        robot: str = "",
        scene: str = "",
    ) -> None:
        self.bucket = bucket_system
        self.gait_type = gait_type
        self.terrain_type = terrain_type
        self.config = config if config is not None else EpisodeRecorderConfig()
        self.episode_prefix = episode_prefix
        self.robot = robot
        self.scene = scene

        if sim_hz <= 0 or self.config.control_hz <= 0:
            raise ValueError("sim_hz and control_hz must be positive")
        self.sensor_noise = sensor_noise
        if self.sensor_noise is None:
            self.sensor_noise = SensorNoise.from_config(
                dt=1.0 / self.config.control_hz,
                cfg=sensor_noise_config,
            )
        self._decim = max(1, int(round(sim_hz / self.config.control_hz)))
        self._event_mode = self.config.episode_mode != "fixed_duration"
        self._max_control_steps = max(
            1, int(round(self.config.episode_duration_s * self.config.control_hz))
        )
        # Episodes shorter than this are dropped as too short to be worth keeping.
        # It must cover the longest window a downstream dataset will cut.
        self._min_control_steps = max(
            1, int(round(self.config.min_episode_duration_s * self.config.control_hz))
        )

        self._episode_index = 0
        self._sim_step = 0
        self._control_step = 0
        self._recording = True
        self._buffer: Dict[str, List[NDArray]] = {
            name: [] for name in self.SAMPLED_COLUMNS
        }
        self._episode_started_at = episode_timestamp()

        self.episodes_stored = 0
        self.episodes_discarded = 0
        self.last_add_result: dict | None = None
        # Physical conditions in effect for the episode being buffered.
        self._friction: float | None = None
        self._payload_kg: float | None = None
        self._episode_randomization: dict | None = None
        self.episode_randomization: Dict[str, dict] = {}

    @property
    def decimation(self) -> int:
        return self._decim

    @property
    def buffer_control_steps(self) -> int:
        return self._control_step

    @property
    def event_mode(self) -> bool:
        """True when episode boundaries come from simulator events, not a timer."""
        return self._event_mode

    @property
    def min_control_steps(self) -> int:
        return self._min_control_steps

    @property
    def max_control_steps(self) -> int:
        return self._max_control_steps

    def begin_episode(self) -> None:
        """Start a fresh episode buffer (does not reset the robot)."""
        self._sim_step = 0
        self._control_step = 0
        for values in self._buffer.values():
            values.clear()
        self._episode_started_at = episode_timestamp()
        if self.sensor_noise is not None:
            self.sensor_noise.reset()

    def set_episode_conditions(
        self,
        *,
        friction: float | None = None,
        payload_kg: float | None = None,
        randomization: dict | None = None,
    ) -> None:
        """
        Record the physical conditions of the episode currently being buffered.

        ``friction`` and ``payload_kg`` are read from the live model by the
        simulator, so they are populated whether or not reset randomization is
        enabled. ``randomization`` carries the sampled reset knobs when it is.
        """
        self._friction = None if friction is None else float(friction)
        self._payload_kg = None if payload_kg is None else float(payload_kg)
        self._episode_randomization = None if not randomization else dict(randomization)

    def set_recording(self, active: bool, *, reason: str = "gate") -> bool:
        """
        Gate buffering on a task condition (e.g. "robot holds the desired pose").

        Turning the gate off closes the current episode, so the stored episode
        covers exactly the stretch where the condition held. Returns ``True`` when
        that close committed an episode to the buckets.
        """
        active = bool(active)
        if active == self._recording:
            return False

        self._recording = active
        if active:
            self.begin_episode()
            return False
        return self.end_episode(reason=reason)

    def end_episode(self, *, reason: str = "event") -> bool:
        """
        Close the current episode: commit it when long enough, otherwise drop it.

        Returns ``True`` if the episode reached the bucket system.
        """
        if self._control_step == 0:
            return False
        outcome = classify_termination(reason)
        if (
            outcome is EpisodeOutcome.FAILURE
            and not self.config.store_failed_episodes
        ):
            self.discard(reason=f"{reason}_not_stored")
            return False
        if self._control_step < self._min_control_steps:
            self.discard(reason=f"{reason}_too_short")
            return False
        return self._finalize_episode(reason=reason)

    def discard(self, *, reason: str = "discarded") -> None:
        """Drop the current buffer without storing."""
        if self._control_step > 0:
            self.episodes_discarded += 1
            print(
                f"[collect] sequence removed  reason={reason}  "
                f"steps={self._control_step}  (not saved to buckets)",
                flush=True,
            )
            self.bucket.print_bucket_snapshot(
                event="after remove",
                detail=f"reason={reason}, discarded_episodes={self.episodes_discarded}",
            )
        self.begin_episode()

    def step_sim(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tau: NDArray,
        contact_ids: NDArray,
        base_force_pert: RandomBaseForcePerturbation,
        n_joints: int,
    ) -> bool:
        """
        Record one simulation step; decimates to ``control_hz``.

        Buffering is skipped while the recording gate is closed. Returns ``True``
        if this step closed an episode and stored it in the bucket system, which
        happens on the duration cap here and on task events via :meth:`end_episode`.
        """
        if not self._recording:
            return False

        self._sim_step += 1
        if (self._sim_step - 1) % self._decim != 0:
            return False

        sample = sample_step(
            model, data, tau, contact_ids, n_joints,
            sensor_noise=self.sensor_noise,
        )
        sample["external_force"] = np.asarray(
            base_force_pert.force, dtype=np.float32
        ).reshape(3)
        for name in self.SAMPLED_COLUMNS:
            self._buffer[name].append(sample[name])
        self._control_step += 1

        if self._control_step < self._max_control_steps:
            return False
        return self._finalize_episode(
            reason="duration_cap" if self._event_mode else "fixed_duration"
        )

    def flush_partial(self) -> bool:
        """Store the current buffer if it is long enough (e.g. at shutdown)."""
        return self.end_episode(reason="shutdown")

    def _build_arrays(self) -> Dict[str, NDArray]:
        """Stack the buffered columns and derive the contact-based labels."""
        n_steps = self._control_step
        dt = 1.0 / self.config.control_hz

        arrays: Dict[str, NDArray] = {}
        for column in EPISODE_COLUMNS:
            if column.name in self._buffer:
                arrays[column.name] = np.stack(
                    self._buffer[column.name], axis=0
                ).astype(column.dtype, copy=False)

        arrays["t"] = np.arange(n_steps, dtype=np.int32)
        arrays["time_s"] = (np.arange(n_steps, dtype=np.float32) * dt).astype(np.float32)

        contact = self.bucket.derive_contacts(arrays["grf_world"])
        arrays["contact"] = contact
        arrays["rare_contact"] = np.asarray(
            [contact_state_name(bits) == RARE_CONTACT_STATE for bits in contact],
            dtype=np.bool_,
        )
        arrays["dt_since_transition"] = dt_since_transition(contact, dt)
        return arrays

    def _finalize_episode(self, *, reason: str = "event") -> bool:
        if self._control_step < self._min_control_steps:
            self.discard(reason=f"{reason}_too_short")
            return False

        steps = self._control_step
        self._episode_index += 1
        episode_id = f"{self.episode_prefix}_{self._episode_index:05d}"
        outcome = classify_termination(reason)

        metadata = EpisodeMetadata(
            episode_id=episode_id,
            robot=self.robot,
            scene=self.scene,
            terrain=self.terrain_type.value,
            gait=self.gait_type.value,
            timestamp=self._episode_started_at,
            ended_at=episode_timestamp(),
            control_hz=self.config.control_hz,
            friction=self._friction,
            payload_kg=self._payload_kg,
            terminate_by=outcome.value,
            terminate_reason=reason,
            split=assign_split(
                episode_id,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                seed=self.config.split_seed,
            ),
            reset_randomization=dict(self._episode_randomization or {}),
        )
        record = EpisodeRecord(metadata=metadata, arrays=self._build_arrays())

        self.last_add_result = self.bucket.add_episode(
            record, stride=self.config.label_stride
        )
        if self._episode_randomization:
            self.episode_randomization[episode_id] = dict(self._episode_randomization)
        self.episodes_stored += 1
        r = self.last_add_result or {}
        print(
            f"[collect] episode committed to buckets  episode={episode_id}  "
            f"terminate_by={metadata.terminate_by} ({reason})  "
            f"split={metadata.split}  "
            f"samples_added={r.get('added', 0)}  "
            f"samples_rejected={r.get('rejected', 0)}  "
            f"samples_rare={r.get('rare', 0)}  "
            f"steps={steps} ({steps / self.config.control_hz:.1f}s)  "
            f"episodes_total={self.episodes_stored}",
            flush=True,
        )
        self.bucket.print_bucket_snapshot(
            event="after store",
            detail=(
                f"{episode_id}, +{r.get('added', 0)} samples, "
                f"rare={r.get('rare', 0)}"
            ),
        )
        self.begin_episode()
        return True


class SimCollectionHooks:
    """
    Null-safe hooks for minimal simulator integration (one import, few call sites).

    When disabled, every method is a no-op so simulators can call hooks unconditionally.
    """

    enabled: bool = False

    def on_ready(self) -> None:
        """Call once after sim warm-up before the main loop."""

    def on_respawn(self, *, manual: bool = False, crashed: bool = False) -> None:
        """Call at the start of each respawn (manual key or crash)."""

    def after_physics_step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tau: NDArray,
        contact_ids: NDArray,
        base_force_pert: RandomBaseForcePerturbation,
        n_joints: int,
    ) -> None:
        """Call once per sim step, after ``mj_step``."""

    def end_episode(self, *, reason: str = "event") -> bool:
        """Close the current episode on a task event (e.g. goal reached)."""
        return False

    def set_recording(self, active: bool, *, reason: str = "gate") -> bool:
        """Record only while ``active``; closing the gate ends the episode."""
        return False

    def finish(self, default_out: str) -> None:
        """Flush buffer, print summary, write the dataset (no-op when disabled)."""

    def set_episode_conditions(
        self,
        *,
        friction: float | None = None,
        payload_kg: float | None = None,
        randomization: dict | None = None,
    ) -> None:
        """Record this episode's physical conditions (no-op when disabled)."""


class _NullCollectionHooks(SimCollectionHooks):
    enabled = False


class _ActiveCollectionHooks(SimCollectionHooks):
    enabled = True

    def __init__(
        self,
        recorder: EpisodeRecorder,
        run_dir: str,
        profile: DatasetCollectionConfig,
        metadata: dict,
    ) -> None:
        self._recorder = recorder
        self._run_dir = run_dir
        self._profile = profile
        self._metadata = metadata
        self._finished = False

    def _run_metadata(self) -> dict:
        return {
            **self._metadata,
            "episodes_stored": self._recorder.episodes_stored,
            "episodes_discarded": self._recorder.episodes_discarded,
            "episode_randomization": dict(self._recorder.episode_randomization),
        }

    def on_ready(self) -> None:
        self._recorder.begin_episode()

    def on_respawn(self, *, manual: bool = False, crashed: bool = False) -> None:
        """
        A crash is a task failure, so its episode is closed and stored; a manual
        respawn is the operator interrupting, so its buffer is dropped.
        """
        if manual:
            self._recorder.discard(reason="manual_respawn")
        elif crashed:
            self._on_episode_boundary(self._recorder.end_episode(reason="crash"))
        self._recorder.begin_episode()

    def set_episode_conditions(
        self,
        *,
        friction: float | None = None,
        payload_kg: float | None = None,
        randomization: dict | None = None,
    ) -> None:
        self._recorder.set_episode_conditions(
            friction=friction, payload_kg=payload_kg, randomization=randomization,
        )

    def after_physics_step(
        self,
        model,
        data,
        tau,
        contact_ids,
        base_force_pert,
        n_joints,
    ) -> None:
        self._on_episode_boundary(
            self._recorder.step_sim(
                model, data, tau, contact_ids, base_force_pert, n_joints,
            )
        )

    def end_episode(self, *, reason: str = "event") -> bool:
        return self._on_episode_boundary(self._recorder.end_episode(reason=reason))

    def set_recording(self, active: bool, *, reason: str = "gate") -> bool:
        return self._on_episode_boundary(
            self._recorder.set_recording(active, reason=reason)
        )

    def _on_episode_boundary(self, stored: bool) -> bool:
        """Refresh the episode table and label index whenever one was committed."""
        if stored and self._profile.output.save_after_each_episode:
            self._write_disk()
        return stored

    def _write_disk(self) -> Path:
        """Write the episode table + label index and refresh the dataset memory."""
        exp = self._profile.export
        index_path = self._recorder.bucket.save_dataset(
            self._run_dir,
            metadata=self._run_metadata(),
            write_metadata=self._profile.output.write_metadata_json,
            max_per_bucket=exp.max_per_bucket,
            shuffle=True,
            seed=exp.shuffle_seed,
        )
        print(
            f"[collect] index written → {index_path.resolve()}  "
            f"samples={self._recorder.bucket.total_samples_stored}  "
            f"episodes={len(self._recorder.bucket.episodes)}",
            flush=True,
        )
        if self._recorder.bucket.dataset_summary_path is not None:
            memory_path = self._recorder.bucket.update_dataset_summary(
                index_path,
                run_dir=self._run_dir,
                metadata=self._run_metadata(),
                max_per_bucket=exp.max_per_bucket,
            )
            whole = self._recorder.bucket.dataset_memory["summary"]
            print(
                f"[collect] dataset memory updated → {memory_path.resolve()}  "
                f"runs={whole['dataset_files']}  samples={whole['samples']}",
                flush=True,
            )
        return index_path

    def finish(self, default_out: str) -> None:
        del default_out
        if self._finished:
            return
        self._finished = True

        self._recorder.flush_partial()
        self._recorder.bucket.print_summary()
        if self._recorder.bucket.total_samples_stored > 0:
            self._write_disk()
        else:
            print("[collect] no samples stored — dataset not written", flush=True)


_NULL_HOOKS = _NullCollectionHooks()


def setup_sim_collection(
    enabled: bool,
    *,
    gait_type: GaitType,
    scene: str,
    sim_hz: float,
    robot: str,
    episode_duration_s: float | None = None,
    collect_out: str | None = None,
    name_prefix: str = "loco",
    cfg: DatasetCollectionConfig | None = None,
) -> SimCollectionHooks:
    """Create collection hooks for a simulator, or a no-op stub when disabled."""
    profile = cfg if cfg is not None else dataset_collection_config
    active = enabled or profile.enabled
    if not active:
        return _NULL_HOOKS

    terrain = scene_to_terrain(scene)
    ep_duration = (
        profile.episode.episode_duration_s
        if episode_duration_s is None
        else episode_duration_s
    )

    run_dir = resolve_run_directory(
        prefix=name_prefix,
        robot=robot,
        scene=scene,
        gait=gait_type.value,
        terrain=terrain.value,
        output_root_dir=profile.output.output_root_dir,
        run_folder_pattern=profile.output.run_folder_pattern,
        use_timestamp=profile.output.use_timestamp,
        collect_out=collect_out,
    )

    recorder = create_collection_session(
        gait_type=gait_type,
        terrain_type=terrain,
        sim_hz=sim_hz,
        # The run folder name is unique per run, so episode ids stay unique
        # across runs — the split hash and any multi-run sampler depend on that.
        episode_prefix=run_dir.name,
        cfg=profile,
        episode_duration_s=ep_duration,
        run_dir=run_dir,
        robot=robot,
        scene=scene,
    )
    ctrl_hz = recorder.config.control_hz
    boundary = (
        f"episodes closed by task events (cap {ep_duration:.0f}s)"
        if recorder.event_mode
        else f"episodes closed every {ep_duration:.0f}s"
    )
    print(
        f"[collect] recording @ {ctrl_hz:.0f} Hz "
        f"(decim={recorder.decimation}), {boundary}, "
        f"min episode {recorder.min_control_steps / ctrl_hz:.1f}s",
        flush=True,
    )
    print(
        f"[collect] episode split: val={recorder.config.val_ratio:.0%} "
        f"test={recorder.config.test_ratio:.0%} (seed {recorder.config.split_seed}), "
        f"failed episodes "
        f"{'stored' if recorder.config.store_failed_episodes else 'discarded'}",
        flush=True,
    )
    print(describe_schema(), flush=True)
    sn = recorder.sensor_noise
    if sn.enabled:
        rw = "on" if sn.imu_random_walk else "off"
        print(
            f"[collect] sensor noise: joint_pos σ={sn.joint_pos_std:.4g} rad, "
            f"imu_acc σ={sn.imu_acc_std:.4g} m/s^2, "
            f"imu_gyro σ={sn.imu_gyro_std:.4g} rad/s, "
            f"imu_random_walk={rw}",
            flush=True,
        )
    else:
        print("[collect] sensor noise: disabled (ground-truth samples)", flush=True)
    print(f"[collect] run dir → {run_dir.resolve()}", flush=True)
    memory = recorder.bucket.dataset_memory["summary"]
    memory_path = recorder.bucket.dataset_summary_path
    if memory_path is not None and memory_path.exists():
        print(
            f"[collect] dataset memory loaded ← {memory_path.resolve()}  "
            f"runs={memory['dataset_files']}  samples={memory['samples']}",
            flush=True,
        )
    elif memory_path is not None and memory["dataset_files"] > 0:
        print(
            f"[collect] dataset memory bootstrapped from existing runs  "
            f"runs={memory['dataset_files']}  samples={memory['samples']}  "
            f"(will write → {memory_path.resolve()})",
            flush=True,
        )
    elif memory_path is not None:
        print(
            f"[collect] dataset memory will be created → {memory_path.resolve()}",
            flush=True,
        )
    hooks = _ActiveCollectionHooks(
        recorder,
        run_dir=str(run_dir),
        profile=profile,
        metadata={
            "prefix": name_prefix,
            "robot": robot,
            "scene": scene,
            "gait": gait_type.value,
            "terrain": terrain.value,
            "sim_hz": sim_hz,
            "control_hz": recorder.config.control_hz,
            "episode_duration_s": ep_duration,
            "sensor_noise": recorder.sensor_noise.to_metadata(),
        },
    )
    atexit.register(hooks.finish, "")
    return hooks


def create_collection_session(
    *,
    gait_type: GaitType,
    terrain_type: TerrainType,
    sim_hz: float,
    episode_prefix: str = "ep",
    cfg: DatasetCollectionConfig | None = None,
    run_dir: str | Path | None = None,
    robot: str = "",
    scene: str = "",
    # Optional overrides (take precedence over ``cfg`` when set).
    bucket_capacity: int | None = None,
    episode_duration_s: float | None = None,
    control_hz: float | None = None,
    sensor_noise_cfg: SensorNoiseConfig | None = None,
) -> EpisodeRecorder:
    """Build a recorder wired to a new :class:`DatasetBucketSystem`."""
    profile = cfg if cfg is not None else dataset_collection_config
    b = profile.bucket
    e = profile.episode
    x = profile.export

    bucket = DatasetBucketSystem(
        bucket_capacity=bucket_capacity if bucket_capacity is not None else b.bucket_capacity,
        contact_force_threshold=b.contact_force_threshold,
        perturbation_force_threshold=b.perturbation_force_threshold,
        min_perturbation_ratio=b.min_perturbation_ratio,
        dataset_summary_path=(
            Path(profile.output.output_root_dir) / DATASET_SUMMARY_FILENAME
        ),
        store=(
            None
            if run_dir is None
            else EpisodeStore(
                Path(run_dir), compression=profile.output.parquet_compression
            )
        ),
    )
    ep_duration = episode_duration_s if episode_duration_s is not None else e.episode_duration_s
    ctrl_hz = control_hz if control_hz is not None else e.control_hz
    noise_cfg = sensor_noise_cfg if sensor_noise_cfg is not None else sensor_noise_config
    return EpisodeRecorder(
        bucket,
        gait_type=gait_type,
        terrain_type=terrain_type,
        sim_hz=sim_hz,
        config=EpisodeRecorderConfig(
            control_hz=ctrl_hz,
            episode_duration_s=ep_duration,
            min_episode_duration_s=e.min_episode_duration_s,
            episode_mode=e.episode_mode,
            label_stride=e.label_stride,
            store_failed_episodes=e.store_failed_episodes,
            val_ratio=x.val_ratio,
            test_ratio=x.test_ratio,
            split_seed=x.split_seed,
        ),
        episode_prefix=episode_prefix,
        sensor_noise=SensorNoise.from_config(dt=1.0 / ctrl_hz, cfg=noise_cfg),
        robot=robot,
        scene=scene,
    )
