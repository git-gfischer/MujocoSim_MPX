"""
Episode buffer and routing into :class:`DatasetBucketSystem`.

Records proprioception + labels at a fixed control rate (default 50 Hz) while the
simulator runs at a higher rate (e.g. 200 Hz). Episodes are stored only when they
complete without a crash; crash or manual respawn discards the current buffer.

An *episode* is one continuous stretch of simulation between two task events, so
its length varies. The simulator decides the boundaries:

* ``quad_locomotion`` closes an episode when the robot reaches its navigation goal.
* ``quad_4balance`` records only while the robot holds the desired pose and closes
  the episode when that pose is lost.

``episode_duration_s`` is only a safety cap in this mode. Set
``EpisodeCollectionConfig.episode_mode = "fixed_duration"`` to go back to closing
every episode on a timer instead.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mpx.estimators.quad_contact_estimation import estimate_foot_grf
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation
from mpx.config.sim_config.config_dataset_bucket import (
    DatasetCollectionConfig,
    dataset_collection_config,
)
from mpx.utils.dataset_collection.dataset_bucket_system import (
    DATASET_SUMMARY_FILENAME,
    DatasetBucketSystem,
    GaitType,
    TerrainType,
    resolve_dataset_output_path,
)
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


def pack_proprioceptive_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau: NDArray,
    contact_ids: NDArray,
    n_joints: int,
) -> NDArray[np.float32]:
    """
    Pack one control-step proprioceptive vector (66,).

    Layout: joint_pos[12] | joint_vel[12] | torque[12] | imu_acc[3] | imu_gyro[3]
            | foot_pos_base[12] | foot_vel_base[12]
    """
    foot_pos_base, foot_vel_base = feet_yaw_base_kinematics(
        model, data, np.asarray(contact_ids, dtype=np.int32),
    )
    return np.concatenate(
        [
            np.asarray(data.qpos[7 : 7 + n_joints], dtype=np.float32),
            np.asarray(data.qvel[6 : 6 + n_joints], dtype=np.float32),
            np.asarray(tau, dtype=np.float32).ravel()[:n_joints],
            np.asarray(data.qacc[:3], dtype=np.float32),
            np.asarray(data.qvel[3:6], dtype=np.float32),
            np.asarray(foot_pos_base, dtype=np.float32),
            np.asarray(foot_vel_base, dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


@dataclass
class EpisodeRecorderConfig:
    """Timing and naming defaults for on-the-fly dataset collection."""

    control_hz: float = 50.0
    # Event mode: safety cap. Fixed-duration mode: exact episode length.
    episode_duration_s: float = 60.0
    min_episode_duration_s: float = 1.0
    episode_mode: str = "event"
    stride: int = 1


class EpisodeRecorder:
    """
    Buffers one variable-length episode at ``control_hz`` and commits it to a bucket system.

    Call :meth:`step_sim` once per MuJoCo step (after ``mj_step``), then close the
    episode from the simulator with :meth:`end_episode` on a task event, or gate
    recording with :meth:`set_recording` when only part of the run is of interest.
    On crash or :meth:`discard`, the buffer is dropped without reaching the buckets.
    """

    def __init__(
        self,
        bucket_system: DatasetBucketSystem,
        *,
        gait_type: GaitType,
        terrain_type: TerrainType,
        sim_hz: float,
        config: EpisodeRecorderConfig | None = None,
        episode_prefix: str = "ep",
    ) -> None:
        self.bucket = bucket_system
        self.gait_type = gait_type
        self.terrain_type = terrain_type
        self.config = config if config is not None else EpisodeRecorderConfig()
        self.episode_prefix = episode_prefix

        if sim_hz <= 0 or self.config.control_hz <= 0:
            raise ValueError("sim_hz and control_hz must be positive")
        self._decim = max(1, int(round(sim_hz / self.config.control_hz)))
        self._event_mode = self.config.episode_mode != "fixed_duration"
        self._max_control_steps = max(
            1, int(round(self.config.episode_duration_s * self.config.control_hz))
        )
        # An episode must span at least one full window to produce a sample.
        self._min_control_steps = max(
            self.bucket.window_size,
            int(round(self.config.min_episode_duration_s * self.config.control_hz)),
        )

        self._episode_index = 0
        self._sim_step = 0
        self._control_step = 0
        self._recording = True
        self._prop: list[NDArray[np.float32]] = []
        self._grf: list[NDArray[np.float64]] = []
        self._ext: list[NDArray[np.float64]] = []

        self.episodes_stored = 0
        self.episodes_discarded = 0
        self.last_add_result: dict | None = None

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
        self._prop.clear()
        self._grf.clear()
        self._ext.clear()

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

        self._prop.append(
            pack_proprioceptive_sample(model, data, tau, contact_ids, n_joints)
        )
        self._grf.append(
            np.asarray(estimate_foot_grf(model, data, contact_ids), dtype=np.float64)
        )
        self._ext.append(np.asarray(base_force_pert.force, dtype=np.float64).reshape(3))
        self._control_step += 1

        if self._control_step < self._max_control_steps:
            return False
        return self._finalize_episode(
            reason="duration_cap" if self._event_mode else "fixed_duration"
        )

    def flush_partial(self) -> bool:
        """Store the current buffer if it is long enough (e.g. at shutdown)."""
        return self.end_episode(reason="shutdown")

    def _finalize_episode(self, *, reason: str = "event") -> bool:
        if self._control_step < self._min_control_steps:
            self.discard(reason=f"{reason}_too_short")
            return False

        steps = self._control_step
        self._episode_index += 1
        episode_id = f"{self.episode_prefix}_{self.gait_type.value}_{self.terrain_type.value}_{self._episode_index:05d}"

        prop = np.stack(self._prop, axis=0)
        grf = np.stack(self._grf, axis=0)
        ext = np.stack(self._ext, axis=0)

        self.last_add_result = self.bucket.add_episode(
            episode_id=episode_id,
            proprioceptive_data=prop,
            grf_world=grf,
            external_force=ext,
            gait_type=self.gait_type,
            terrain_type=self.terrain_type,
            stride=self.config.stride,
        )
        self.episodes_stored += 1
        r = self.last_add_result or {}
        print(
            f"[collect] episode committed to buckets  episode={episode_id}  "
            f"reason={reason}  "
            f"windows_added={r.get('added', 0)}  "
            f"windows_invalid={r.get('discarded_invalid_state', 0)}  "
            f"steps={steps} ({steps / self.config.control_hz:.1f}s)  "
            f"episodes_total={self.episodes_stored}",
            flush=True,
        )
        self.bucket.print_bucket_snapshot(
            event="after store",
            detail=(
                f"{episode_id}, +{r.get('added', 0)} windows, "
                f"invalid={r.get('discarded_invalid_state', 0)}"
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
        """Flush buffer, print summary, save ``.npz`` (no-op when disabled)."""


class _NullCollectionHooks(SimCollectionHooks):
    enabled = False


class _ActiveCollectionHooks(SimCollectionHooks):
    enabled = True

    def __init__(
        self,
        recorder: EpisodeRecorder,
        npz_path: str,
        run_dir: str,
        profile: DatasetCollectionConfig,
        metadata: dict,
    ) -> None:
        self._recorder = recorder
        self._npz_path = npz_path
        self._run_dir = run_dir
        self._profile = profile
        self._metadata = metadata
        self._finished = False

    def _run_metadata(self) -> dict:
        return {
            **self._metadata,
            "episodes_stored": self._recorder.episodes_stored,
            "episodes_discarded": self._recorder.episodes_discarded,
        }

    def on_ready(self) -> None:
        self._recorder.begin_episode()

    def on_respawn(self, *, manual: bool = False, crashed: bool = False) -> None:
        if manual or crashed:
            reason = "crash" if crashed else "manual_respawn"
            self._recorder.discard(reason=reason)
        self._recorder.begin_episode()

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
        """Persist the dataset whenever an episode was just committed."""
        if stored and self._profile.output.save_after_each_episode:
            self._write_disk()
        return stored

    def _write_disk(self) -> Path:
        """Save the current dataset and refresh its persistent summary record."""
        exp = self._profile.export
        saved = self._recorder.bucket.save_dataset(
            self._run_dir,
            self._npz_path,
            metadata=self._run_metadata(),
            write_metadata=self._profile.output.write_metadata_json,
            max_per_bucket=exp.max_per_bucket,
            shuffle=True,
            seed=exp.shuffle_seed,
        )
        print(
            f"[collect] file written → {saved.resolve()}  "
            f"windows={self._recorder.bucket.total_windows_stored}",
            flush=True,
        )
        memory_path = self._recorder.bucket.update_dataset_summary(
            saved,
            run_dir=self._run_dir,
            metadata=self._run_metadata(),
            max_per_bucket=exp.max_per_bucket,
        )
        whole = self._recorder.bucket.dataset_memory["summary"]
        print(
            f"[collect] dataset memory updated → {memory_path.resolve()}  "
            f"datasets={whole['dataset_files']}  windows={whole['windows']}",
            flush=True,
        )
        return saved

    def finish(self, default_out: str) -> None:
        del default_out
        if self._finished:
            return
        self._finished = True

        self._recorder.flush_partial()
        self._recorder.bucket.print_summary()
        if self._recorder.bucket.total_windows_stored > 0:
            self._write_disk()
        else:
            print("[collect] no windows stored — file not written", flush=True)


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

    run_dir, npz_path = resolve_dataset_output_path(
        prefix=name_prefix,
        robot=robot,
        scene=scene,
        gait=gait_type.value,
        terrain=terrain.value,
        output_root_dir=profile.output.output_root_dir,
        run_folder_pattern=profile.output.run_folder_pattern,
        filename_pattern=profile.output.filename_pattern,
        use_timestamp=profile.output.use_timestamp,
        collect_out=collect_out,
    )

    recorder = create_collection_session(
        gait_type=gait_type,
        terrain_type=terrain,
        sim_hz=sim_hz,
        episode_duration_s=ep_duration,
        episode_prefix=f"{name_prefix}_{robot}_{scene}",
        cfg=profile,
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
    print(f"[collect] output dir → {run_dir.resolve()}", flush=True)
    print(f"[collect] npz file  → {npz_path.resolve()}", flush=True)
    memory = recorder.bucket.dataset_memory["summary"]
    memory_path = recorder.bucket.dataset_summary_path
    if memory_path is not None and memory_path.exists():
        print(
            f"[collect] dataset memory loaded ← {memory_path.resolve()}  "
            f"datasets={memory['dataset_files']}  windows={memory['windows']}",
            flush=True,
        )
    elif memory_path is not None and memory["dataset_files"] > 0:
        print(
            f"[collect] dataset memory bootstrapped from existing files  "
            f"datasets={memory['dataset_files']}  windows={memory['windows']}  "
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
        npz_path=str(npz_path),
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
    # Optional overrides (take precedence over ``cfg`` when set).
    window_size: int | None = None,
    bucket_capacity: int | None = None,
    episode_duration_s: float | None = None,
    control_hz: float | None = None,
) -> EpisodeRecorder:
    """Build a recorder wired to a new :class:`DatasetBucketSystem`."""
    profile = cfg if cfg is not None else dataset_collection_config
    b = profile.bucket
    e = profile.episode

    bucket = DatasetBucketSystem(
        window_size=window_size if window_size is not None else b.window_size,
        bucket_capacity=bucket_capacity if bucket_capacity is not None else b.bucket_capacity,
        contact_force_threshold=b.contact_force_threshold,
        perturbation_force_threshold=b.perturbation_force_threshold,
        min_perturbation_ratio=b.min_perturbation_ratio,
        dataset_summary_path=(
            Path(profile.output.output_root_dir) / DATASET_SUMMARY_FILENAME
        ),
    )
    ep_duration = episode_duration_s if episode_duration_s is not None else e.episode_duration_s
    ctrl_hz = control_hz if control_hz is not None else e.control_hz
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
            stride=e.window_stride,
        ),
        episode_prefix=episode_prefix,
    )
