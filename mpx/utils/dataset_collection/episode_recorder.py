"""
Episode buffer and routing into :class:`DatasetBucketSystem` (schema v4).

Records proprioception + ground truth at a fixed control rate (default 50 Hz)
while the simulator runs at a higher rate (default 500 Hz). One row is buffered
per control step; the finished episode is written once as a parquet table, whole
and untrimmed, and each of its labelled timesteps is registered as an
``(episode_id, t)`` reference in the buckets. No window length is involved —
that is the training-time dataset's choice.

What v4 changed, and why
------------------------
*Contact labels are reduced from the substeps, not sampled at an instant.*
Per-foot force accumulates on every sim step; at the control step the majority
of substeps decides ``contact_raw``, and a Schmitt trigger with a minimum dwell
turns that into the primary ``contact`` label. v3 sampled one instant and
produced 31.6% one-frame contact runs.

*Everything the robot "measures" flows from one noisy state.* The encoder
reading is corrupted first, and foot kinematics are then computed by forward
kinematics **on that reading**. v3 computed foot pose from the simulator's own
geom poses and added encoder noise afterwards, which left ``foot_pos_base`` a
privileged signal wearing an input's name.

*The IMU is read from the model's sensors.* ``imu_acc_body`` is specific force
in the IMU frame with gravity included — what an accelerometer actually reports
— and ``base_quat`` is stored so the world frame can be recovered.

*Commanded torque is privileged; measured actuator force is the input.* The MPC
solves for desired GRFs with zero force at scheduled swing feet, so its
commanded torque is close to a direct readout of the planned contact schedule.
The plan itself is logged as ``contact_schedule`` so the leak is auditable
rather than hidden inside the torques.

Episode boundaries
------------------
An *episode* is one continuous stretch of simulation between two task events, so
its length varies. Each closed episode carries an outcome (``terminate_by``):
reaching the goal is a success, falling or losing the commanded pose is a
failure, and the duration cap or shutdown truncates an otherwise healthy
episode. Failures are kept by default — the steps leading into a fall are the
ones a contact/GRF estimator gets wrong. A manual respawn is not a task outcome,
so it always discards the buffer.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

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
from mpx.utils.dataset_collection.contact_labeling import (
    ContactDebouncer,
    ContactLabelConfig,
    SubstepForceAccumulator,
    contact_run_statistics,
)
from mpx.utils.dataset_collection.dataset_bucket_system import (
    DATASET_SUMMARY_FILENAME,
    DatasetBucketSystem,
    GaitType,
    TerrainType,
    is_rare_contact,
    resolve_run_directory,
)
from mpx.utils.dataset_collection.dataset_schema import (
    EPISODE_COLUMNS,
    EpisodeMetadata,
    EpisodeOutcome,
    EpisodeRecord,
    assign_split,
    classify_termination,
    describe_schema,
    dt_since_transition,
    episode_timestamp,
    randomization_group_id,
)
from mpx.utils.dataset_collection.episode_storage import EpisodeStore
from mpx.utils.dataset_collection.signal_bounds import (
    CLIPPING_LIMIT,
    clipping_audit,
    default_signal_bounds_path,
    describe_bounds,
    load_signal_bounds,
    merge_clipping_audits,
    signal_bounds_sha256,
    signal_bounds_version,
    worst_clipping,
)
from mpx.utils.simulation_utils.attitude_estimator import ComplementaryAttitudeFilter
from mpx.utils.simulation_utils.measured_kinematics import (
    ImuReader,
    MeasuredKinematics,
    TerrainProbe,
    gravity_align,
    measured_joint_torque,
)
from mpx.utils.simulation_utils.sensor_noise import SensorNoise

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Go2 body weight [N], recorded in the contact-labelling metadata block so the
# force thresholds can be read as a fraction of it.
DEFAULT_BODY_WEIGHT_N = 176.0


def scene_to_terrain(scene: str) -> TerrainType:
    """Map simulator ``--scene`` name to a :class:`TerrainType`."""
    if scene in ("flat", "slippery"):
        return TerrainType.FLAT
    if scene in ("stairs", "staired_pyramid"):
        return TerrainType.STAIRS
    return TerrainType.ROUGH


def base_linear_velocity(data: mujoco.MjData) -> "NDArray[np.float64]":
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


@dataclass
class ControlSample:
    """
    What the controller knows this step; the simulator fills it in.

    ``leg_phase`` and ``duty_factor`` come straight off the MPC's gait timer
    (``mpc_data.contact_time`` and ``mpc_data.duty_factor``); the recorder turns
    them into the privileged ``gait_phase`` and ``contact_schedule`` columns, so
    the controller's plan is on the record instead of only implicit in the
    torques.
    """

    tau_cmd: Any
    leg_phase: Any = None
    duty_factor: float | None = None
    cmd_base_vel: Any = None
    cmd_segment_id: int = 0

    def schedule(self, n_feet: int = 4) -> np.ndarray:
        """The contact pattern the controller planned for this step."""
        if self.leg_phase is None or self.duty_factor is None:
            return np.zeros(n_feet, dtype=np.uint8)
        phase = np.asarray(self.leg_phase, dtype=np.float64).reshape(-1)[:n_feet]
        planned = (phase < float(self.duty_factor)).astype(np.uint8)
        if planned.size < n_feet:
            planned = np.pad(planned, (0, n_feet - planned.size))
        return planned

    def phase(self) -> float:
        """Reference gait phase in [0, 1) — the first leg's timer."""
        if self.leg_phase is None:
            return 0.0
        values = np.asarray(self.leg_phase, dtype=np.float64).reshape(-1)
        return float(values[0]) if values.size else 0.0


class StepSampler:
    """
    Turns one control step of simulator state into a v4 row.

    Owns every model-derived helper (IMU sensor slices, kinematics on the
    measured state, terrain rays, actuator limits) so the recorder itself stays
    about buffering and episode boundaries.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        contact_ids,
        n_joints: int,
        sensor_noise: SensorNoise,
        *,
        control_dt: float = 0.02,
        contact_config: ContactLabelConfig | None = None,
        foot_geom_names=("FL", "FR", "RL", "RR"),
        foot_site_suffix: str = "_foot",
    ) -> None:
        contact_config = contact_config or ContactLabelConfig()
        self.model = model
        self.contact_ids = np.asarray(contact_ids, dtype=np.int32).reshape(-1)
        self.n_joints = int(n_joints)
        self.sensor_noise = sensor_noise
        self.n_feet = int(self.contact_ids.size)

        self.foot_site_names = [
            f"{name}{foot_site_suffix}" for name in foot_geom_names
        ][: self.n_feet]
        self.foot_site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.foot_site_names
        ]
        self.imu = ImuReader.from_model(model)
        self.kinematics = MeasuredKinematics(model, self.foot_site_names, n_joints)
        self.terrain = TerrainProbe(model)

        # Attitude from the IMU alone. A gravity-aligned *input* channel must not
        # be built from base_quat: ground-truth attitude is privileged.
        self.attitude = ComplementaryAttitudeFilter(dt=float(control_dt))

        # Foot collision radius read from the model, not hard-coded, so
        # foot_clearance is zero when the foot SURFACE touches the terrain.
        self.foot_collision_radius = float(
            np.mean([model.geom_size[int(g), 0] for g in self.contact_ids])
        )
        # Height-derived contact, labelled the way the real-world self-supervised
        # benchmark labels it: a clearance threshold plus the same dwell as the
        # GRF label, so sim and real numbers are comparable.
        self.height_contact_threshold_m = 0.005
        self.height_debouncer = ContactDebouncer(
            ContactLabelConfig(
                # Reused as a generic two-rail trigger on clearance: "loaded"
                # means clearance BELOW the threshold, so the rails are inverted
                # by negating the signal in ``row``.
                on_threshold_n=-self.height_contact_threshold_m,
                off_threshold_n=-2.0 * self.height_contact_threshold_m,
                min_dwell_steps=contact_config.min_dwell_steps,
            )
        )

        # Applied-force limit per actuator; an all-zero row means "unlimited".
        limits = np.asarray(model.actuator_forcerange, dtype=np.float64)
        unlimited = np.all(limits == 0.0, axis=1)
        self.force_limit = np.where(unlimited, np.inf, limits[:, 1])[: self.n_joints]

    def foot_normal_force(self, data: mujoco.MjData) -> np.ndarray:
        """
        Per-foot GRF magnitude ``(n_feet,)`` [N] at one sim substep.

        Magnitude rather than the world Z component, so the value stays correct
        on a slope where the contact normal is not vertical.
        """
        grf = estimate_foot_grf(self.model, data, self.contact_ids)
        return np.linalg.norm(np.asarray(grf, dtype=np.float64), axis=1)

    def row(
        self,
        data: mujoco.MjData,
        control: ControlSample,
        base_force_pert: RandomBaseForcePerturbation,
        reduced: Dict[str, np.ndarray],
        contact: np.ndarray,
    ) -> Dict[str, Any]:
        """Pack one control step. ``reduced`` comes from the substep accumulator."""
        n_joints = self.n_joints

        # ── true state ───────────────────────────────────────────────────────
        q_true = np.asarray(data.qpos[7 : 7 + n_joints], dtype=np.float64)
        dq_true = np.asarray(data.qvel[6 : 6 + n_joints], dtype=np.float64)
        tau_cmd = np.asarray(control.tau_cmd, dtype=np.float64).ravel()[:n_joints]
        tau_applied = measured_joint_torque(data, n_joints)

        imu_true = self.imu.read(data)

        # ── measured state: one noisy source, everything derived from it ──────
        reading = self.sensor_noise.apply(
            joint_pos=q_true,
            joint_vel=dq_true,
            joint_torque=tau_applied,
            imu_acc=imu_true["acc"],
            imu_gyro=imu_true["gyro"],
        )
        # Attitude from the measured IMU only — never from base_quat.
        quat_est = self.attitude.update(reading.imu_acc, reading.imu_gyro)

        # Body frame: pure FK, no attitude needed. Measured and true differ only
        # by the noise model, which is what makes an ablation between them mean
        # something.
        # ONE FK function for all four body-frame variants; the only difference
        # is which joint state goes in (DATASET_FIX_TASKS_R2, Task R2-1 rule 2).
        # Deriving the true twin a second way left a 1 cm disagreement against
        # FK of its own joint state, so the pair was not comparable.
        foot_pos_base, foot_vel_base = self.kinematics(
            reading.joint_pos, reading.joint_vel
        )
        foot_pos_true, foot_vel_true = self.kinematics(q_true, dq_true)
        # Gravity-aligned: the input pair uses the ESTIMATE, the privileged pair
        # uses ground truth.
        foot_pos_yaw = gravity_align(foot_pos_base, quat_est)
        foot_vel_yaw = gravity_align(foot_vel_base, quat_est)
        foot_pos_yaw_true = gravity_align(foot_pos_true, imu_true["quat"])
        foot_vel_yaw_true = gravity_align(foot_vel_true, imu_true["quat"])

        # ── terrain geometry ─────────────────────────────────────────────────
        foot_world = np.asarray(
            [data.site_xpos[int(sid)] for sid in self.foot_site_ids], dtype=np.float64
        )
        terrain_under_foot = self.terrain.heights_under(data, foot_world)
        base_pos = np.asarray(data.qpos[:3], dtype=np.float64)
        terrain_under_base = self.terrain.height_at(data, base_pos[:2])

        # foot_height_terrain measures to the site (sphere centre), so a foot in
        # contact sits one collision radius above the terrain — 1.8 cm on the Go2.
        # Thresholding it directly yields zero contacts; threshold clearance.
        foot_height = foot_world[:, 2] - terrain_under_foot
        foot_clearance = foot_height - self.foot_collision_radius
        contact_from_height = self.height_debouncer.step(-foot_clearance)

        grf_world = np.asarray(
            estimate_foot_grf(self.model, data, self.contact_ids), dtype=np.float64
        )

        return {
            # ── inputs ───────────────────────────────────────────────────────
            "joint_pos": reading.joint_pos,
            "joint_vel": reading.joint_vel,
            "joint_torque_measured": reading.joint_torque,
            "imu_acc_body": reading.imu_acc,
            "imu_gyro_body": reading.imu_gyro,
            "foot_pos_base": foot_pos_base.astype(np.float32),
            "foot_vel_base": foot_vel_base.astype(np.float32),
            "base_quat_est": quat_est.astype(np.float32),
            "foot_pos_yawbase": foot_pos_yaw.astype(np.float32),
            "foot_vel_yawbase": foot_vel_yaw.astype(np.float32),
            # ── targets ──────────────────────────────────────────────────────
            "contact": np.asarray(contact, dtype=np.uint8),
            "grf_world": grf_world.reshape(-1).astype(np.float32),
            "external_force": np.asarray(
                base_force_pert.force, dtype=np.float32
            ).reshape(3),
            "base_lin_vel": base_linear_velocity(data).astype(np.float32),
            # ── context ──────────────────────────────────────────────────────
            "contact_raw": reduced["contact_raw"].astype(np.uint8),
            "contact_from_height": contact_from_height.astype(np.uint8),
            "cmd_segment_id": np.int32(control.cmd_segment_id),
            "torque_saturated": (
                np.abs(tau_cmd) >= self.force_limit - 1e-9
            ).astype(np.uint8),
            "cmd_base_vel": (
                np.zeros(3, dtype=np.float32)
                if control.cmd_base_vel is None
                else np.asarray(control.cmd_base_vel, dtype=np.float32).reshape(-1)[:3]
            ),
            "sensor_stale": np.bool_(reading.stale),
            # ── privileged ───────────────────────────────────────────────────
            "base_quat": np.asarray(imu_true["quat"], dtype=np.float32),
            "base_pos": base_pos.astype(np.float32),
            "grf_mean_n": reduced["mean"].astype(np.float32),
            "grf_max_n": reduced["max"].astype(np.float32),
            "joint_torque_cmd": tau_cmd.astype(np.float32),
            "gait_phase": np.float32(control.phase()),
            "contact_schedule": control.schedule(self.n_feet),
            "joint_pos_true": q_true.astype(np.float32),
            "joint_vel_true": dq_true.astype(np.float32),
            "foot_pos_base_true": foot_pos_true.astype(np.float32),
            "foot_vel_base_true": foot_vel_true.astype(np.float32),
            "foot_pos_yawbase_true": foot_pos_yaw_true.astype(np.float32),
            "foot_vel_yawbase_true": foot_vel_yaw_true.astype(np.float32),
            "imu_acc_bias": reading.imu_acc_bias,
            "imu_gyro_bias": reading.imu_gyro_bias,
            "foot_pos_world": foot_world.reshape(-1).astype(np.float32),
            "terrain_height_under_foot": terrain_under_foot.astype(np.float32),
            "foot_height_terrain": foot_height.astype(np.float32),
            "foot_clearance": foot_clearance.astype(np.float32),
            "base_height_terrain": np.float32(base_pos[2] - terrain_under_base),
            # ── deprecated v3 aliases, kept so old loaders keep working ───────
            "imu_acc": np.asarray(data.qacc[:3], dtype=np.float32),
            "imu_gyro": np.asarray(data.qvel[3:6], dtype=np.float32),
            "joint_torque": tau_cmd.astype(np.float32),
            "rare_contact": np.bool_(is_rare_contact(contact)),
        }

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "imu": self.imu.to_metadata(),
            "attitude_estimator": self.attitude.to_metadata(),
            "foot_collision_radius_m": float(self.foot_collision_radius),
            "height_contact_threshold_m": float(self.height_contact_threshold_m),
            "foot_sites": list(self.foot_site_names),
            "actuator_force_limit_n_m": [
                None if not np.isfinite(v) else float(v) for v in self.force_limit
            ],
        }


@dataclass
class EpisodeRecorderConfig:
    """Timing and naming defaults for on-the-fly dataset collection."""

    control_hz: float = 50.0
    sim_hz: float = 500.0
    # Event mode: safety cap. Fixed-duration mode: exact episode length.
    episode_duration_s: float = 60.0
    min_episode_duration_s: float = 1.0
    episode_mode: str = "event"
    label_stride: int = 1
    store_failed_episodes: bool = True
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    split_seed: int = 0
    contact_labeling: ContactLabelConfig = field(default_factory=ContactLabelConfig)


class EpisodeRecorder:
    """
    Buffers one variable-length episode at ``control_hz`` and commits it to a bucket system.

    Call :meth:`step_sim` once per MuJoCo step (after ``mj_step``). Per-foot force
    accumulates on every one of those calls; a full row is packed only on the
    control step, from the reduced interval. Close the episode from the simulator
    with :meth:`end_episode` on a task event, or gate recording with
    :meth:`set_recording`. :meth:`discard` drops the buffer without storing it.
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
        sensor_noise: SensorNoise | None = None,
        robot: str = "",
        scene: str = "",
        run_id: str = "",
        mode: str = "",
    ) -> None:
        self.bucket = bucket_system
        self.gait_type = gait_type
        self.terrain_type = terrain_type
        self.config = config if config is not None else EpisodeRecorderConfig()
        self.episode_prefix = episode_prefix
        self.robot = robot
        self.scene = scene
        self.run_id = run_id or episode_prefix
        self.mode = mode

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

        self._sampler: StepSampler | None = None
        self._accumulator = SubstepForceAccumulator(config=self.config.contact_labeling)
        self._debouncer = ContactDebouncer(self.config.contact_labeling)
        self._control: ControlSample | None = None

        self._episode_index = 0
        self._sim_step = 0
        self._control_step = 0
        self._recording = True
        self._buffer: Dict[str, List[Any]] = {c.name: [] for c in EPISODE_COLUMNS}
        self._episode_started_at = episode_timestamp()

        self.episodes_stored = 0
        self.episodes_discarded = 0
        self.last_add_result: dict | None = None
        # Physical conditions in effect for the episode being buffered.
        self._friction: float | None = None
        self._payload_kg: float | None = None
        self._seed: int | None = None
        self._episode_randomization: dict | None = None
        self.episode_randomization: Dict[str, dict] = {}
        self.clipping_audits: List[Dict[str, float]] = []
        self.contact_statistics: List[Dict[str, float]] = []
        self._bounds = load_signal_bounds()

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def decimation(self) -> int:
        return self._decim

    @property
    def substeps_per_control(self) -> int:
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

    @property
    def sampler(self) -> StepSampler | None:
        return self._sampler

    # ── episode lifecycle ────────────────────────────────────────────────────

    def begin_episode(self) -> None:
        """Start a fresh episode buffer (does not reset the robot)."""
        self._sim_step = 0
        self._control_step = 0
        for values in self._buffer.values():
            values.clear()
        self._accumulator.clear()
        self._debouncer.reset()
        if self._sampler is not None:
            self._sampler.attitude.reset()
            self._sampler.height_debouncer.reset()
        self._episode_started_at = episode_timestamp()
        if self.sensor_noise is not None:
            self.sensor_noise.reset()

    def set_episode_conditions(
        self,
        *,
        friction: float | None = None,
        payload_kg: float | None = None,
        randomization: dict | None = None,
        seed: int | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Record the physical conditions of the episode currently being buffered.

        ``friction`` and ``payload_kg`` are read from the live model by the
        simulator, so they are populated whether or not reset randomization is
        enabled. ``randomization`` carries the sampled reset knobs, and its hash
        becomes the episode's ``randomization_group_id`` — the key the manifest
        splits on, so near-duplicate episodes cannot straddle train and test.
        """
        self._friction = None if friction is None else float(friction)
        self._payload_kg = None if payload_kg is None else float(payload_kg)
        self._episode_randomization = None if not randomization else dict(randomization)
        self._seed = None if seed is None else int(seed)
        if mode is not None:
            self.mode = str(mode)

    def set_control(self, control: ControlSample | None) -> None:
        """Hand the recorder this step's controller state."""
        self._control = control

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
        self.begin_episode()

    # ── per-step recording ───────────────────────────────────────────────────

    def step_sim(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tau: "NDArray",
        contact_ids: "NDArray",
        base_force_pert: RandomBaseForcePerturbation,
        n_joints: int,
        control: ControlSample | None = None,
    ) -> bool:
        """
        Record one simulation step.

        Per-foot force is accumulated on **every** call, so the contact label sees
        the whole control interval rather than the instant the control step lands
        on. A full row is packed only every ``decimation`` calls.

        Returns ``True`` if this step closed an episode and stored it.
        """
        if not self._recording:
            return False

        if self._sampler is None:
            self._sampler = StepSampler(
                model, contact_ids, n_joints, self.sensor_noise,
                control_dt=1.0 / self.config.control_hz,
                contact_config=self.config.contact_labeling,
            )
        sampler = self._sampler

        # Every sim substep contributes to the interval's force statistics.
        self._accumulator.push(sampler.foot_normal_force(data))

        self._sim_step += 1
        if self._sim_step % self._decim != 0:
            return False

        if control is None:
            control = self._control
        if control is None:
            control = ControlSample(tau_cmd=tau)

        reduced = self._accumulator.reduce()
        contact = self._debouncer.step(reduced["mean"])
        row = sampler.row(data, control, base_force_pert, reduced, contact)

        for name, value in row.items():
            self._buffer[name].append(value)
        self._buffer["t"].append(np.int32(self._control_step))
        self._buffer["time_s"].append(
            np.float32(self._control_step / self.config.control_hz)
        )
        # Filled in at finalize, from the whole episode's debounced contact.
        self._buffer["dt_since_transition"].append(np.zeros(4, dtype=np.float32))
        self._control_step += 1

        if self._control_step < self._max_control_steps:
            return False
        return self._finalize_episode(
            reason="duration_cap" if self._event_mode else "fixed_duration"
        )

    def flush_partial(self) -> bool:
        """Store the current buffer if it is long enough (e.g. at shutdown)."""
        return self.end_episode(reason="shutdown")

    def _build_arrays(self) -> Dict[str, "NDArray"]:
        """Stack the buffered columns and fill in the whole-episode derivations."""
        arrays: Dict[str, Any] = {}
        for column in EPISODE_COLUMNS:
            arrays[column.name] = np.stack(
                self._buffer[column.name], axis=0
            ).astype(column.dtype, copy=False)

        # Derived from the DEBOUNCED contact: a chattering label would reset this
        # timer on every one-frame glitch and destroy the signal.
        arrays["dt_since_transition"] = dt_since_transition(
            arrays["contact"], 1.0 / self.config.control_hz
        )
        return arrays

    def _finalize_episode(self, *, reason: str = "event") -> bool:
        if self._control_step < self._min_control_steps:
            self.discard(reason=f"{reason}_too_short")
            return False

        steps = self._control_step
        self._episode_index += 1
        episode_id = f"{self.episode_prefix}_{self._episode_index:05d}"
        outcome = classify_termination(reason)
        knobs = dict(self._episode_randomization or {})
        group_id = randomization_group_id(knobs)

        metadata = EpisodeMetadata(
            episode_id=episode_id,
            run_id=self.run_id,
            robot=self.robot,
            scene=self.scene,
            terrain=self.terrain_type.value,
            gait=self.gait_type.value,
            mode=self.mode,
            timestamp=self._episode_started_at,
            ended_at=episode_timestamp(),
            control_hz=self.config.control_hz,
            sim_hz=self.config.sim_hz,
            substeps_per_control=self._decim,
            friction=self._friction,
            payload_kg=self._payload_kg,
            terminate_by=outcome.value,
            terminate_reason=reason,
            seed=self._seed,
            randomization_group_id=group_id,
            # Cached convenience only: datasets/manifest.json is the authority,
            # and it splits on randomization_group_id so near-duplicate episodes
            # cannot straddle train and test.
            split_assigned=assign_split(
                group_id,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                seed=self.config.split_seed,
            ),
            reset_randomization=knobs,
        )
        arrays = self._build_arrays()
        record = EpisodeRecord(metadata=metadata, arrays=arrays)

        # Clipping is audited over EVERY row: the gate asks how much of the
        # encoded image saturates, and a loader that keeps crash frames encodes
        # them too. (Chatter is different — a tumbling robot genuinely has
        # erratic contacts, so that statistic excludes post-failure frames.)
        usable = ~self.bucket.post_failure_mask(record)
        self.clipping_audits.append(clipping_audit(arrays, self._bounds))
        stats = contact_run_statistics(
            arrays["contact"][usable], self.config.control_hz
        )
        self.contact_statistics.append(stats)

        self.last_add_result = self.bucket.add_episode(
            record, stride=self.config.label_stride
        )
        if knobs:
            self.episode_randomization[episode_id] = dict(knobs)
        self.episodes_stored += 1
        r = self.last_add_result or {}
        print(
            f"[collect] episode committed  episode={episode_id}  "
            f"terminate_by={metadata.terminate_by} ({reason})  "
            f"split={metadata.split_assigned}  group={group_id}  "
            f"samples={r.get('added', 0)}  "
            f"steps={steps} ({steps / self.config.control_hz:.1f}s)  "
            f"chatter={stats['fraction_runs_length_1']:.1%}  "
            f"transitions/foot/s={stats['transitions_per_foot_per_s']:.2f}  "
            f"episodes_total={self.episodes_stored}",
            flush=True,
        )
        self.begin_episode()
        return True

    # ── run-level reporting ──────────────────────────────────────────────────

    def clipping_audit(self) -> Dict[str, Any]:
        """Run-level clipping audit: counts summed, fractions recomputed."""
        return merge_clipping_audits(self.clipping_audits)

    def contact_quality(self) -> Dict[str, float]:
        """
        Chatter statistics pooled over the run's usable frames.

        Pooled rather than worst-episode: a 73-step episode that ended in a fall
        contributes a handful of contact runs, so its one-step fraction is mostly
        sampling noise. Pooling is also what the acceptance check computes.
        """
        if not self.contact_statistics:
            return {}
        total_runs = sum(s["runs"] for s in self.contact_statistics)
        if total_runs == 0:
            return {}
        one_step = sum(
            s["fraction_runs_length_1"] * s["runs"] for s in self.contact_statistics
        )
        total_steps = sum(s["steps"] for s in self.contact_statistics)
        transitions = sum(s["transitions"] for s in self.contact_statistics)
        return {
            "fraction_runs_length_1": one_step / total_runs,
            "transitions_per_foot_per_s": (
                transitions / 4 / (total_steps / self.config.control_hz)
                if total_steps
                else 0.0
            ),
            "median_stance_steps": float(
                np.median([s["median_stance_steps"] for s in self.contact_statistics])
            ),
            "median_swing_steps": float(
                np.median([s["median_swing_steps"] for s in self.contact_statistics])
            ),
            "episodes": len(self.contact_statistics),
        }


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

    def set_control(self, control: ControlSample | None) -> None:
        """Hand over this step's controller state (commanded torque, gait plan)."""

    def after_physics_step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tau: "NDArray",
        contact_ids: "NDArray",
        base_force_pert: RandomBaseForcePerturbation,
        n_joints: int,
        control: ControlSample | None = None,
    ) -> bool:
        """
        Call once per sim step, after ``mj_step``.

        Returns ``True`` when this step closed an episode (the duration cap), so
        the simulator can resample its domain randomization for the next one.
        """
        return False

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
        seed: int | None = None,
        mode: str | None = None,
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
        sampler = self._recorder.sampler
        return {
            **self._metadata,
            "episodes_stored": self._recorder.episodes_stored,
            "episodes_discarded": self._recorder.episodes_discarded,
            "episode_randomization": dict(self._recorder.episode_randomization),
            "clipping_audit": self._recorder.clipping_audit(),
            "contact_quality": self._recorder.contact_quality(),
            **({"sampler": sampler.to_metadata()} if sampler is not None else {}),
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
        seed: int | None = None,
        mode: str | None = None,
    ) -> None:
        self._recorder.set_episode_conditions(
            friction=friction,
            payload_kg=payload_kg,
            randomization=randomization,
            seed=seed,
            mode=mode,
        )

    def set_control(self, control: ControlSample | None) -> None:
        self._recorder.set_control(control)

    def after_physics_step(
        self,
        model,
        data,
        tau,
        contact_ids,
        base_force_pert,
        n_joints,
        control: ControlSample | None = None,
    ) -> bool:
        return self._on_episode_boundary(
            self._recorder.step_sim(
                model, data, tau, contact_ids, base_force_pert, n_joints,
                control=control,
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
        if self._recorder.bucket.dataset_summary_path is not None:
            self._recorder.bucket.update_dataset_summary(
                index_path,
                run_dir=self._run_dir,
                metadata=self._run_metadata(),
                max_per_bucket=exp.max_per_bucket,
            )
        return index_path

    def finish(self, default_out: str) -> None:
        del default_out
        if self._finished:
            return
        self._finished = True

        self._recorder.flush_partial()
        self._recorder.bucket.print_summary()

        quality = self._recorder.contact_quality()
        if quality:
            chatter = quality["fraction_runs_length_1"]
            rate = quality["transitions_per_foot_per_s"]
            flag = "ok" if chatter < 0.02 and rate < 3.5 else "FAIL"
            print(
                f"[collect] contact quality [{flag}]  "
                f"1-step runs={chatter:.2%} (limit 2%)  "
                f"transitions/foot/s={rate:.2f} (limit 3.5)  "
                f"median stance={quality['median_stance_steps']:.0f} steps  "
                f"median swing={quality['median_swing_steps']:.0f} steps",
                flush=True,
            )
        audit = self._recorder.clipping_audit()
        if audit.get("per_element"):
            name, fraction = worst_clipping(audit)
            row_any = audit["per_row_any"].get(name, 0.0)
            flag = "ok" if fraction < CLIPPING_LIMIT else "FAIL"
            print(
                f"[collect] clipping audit [{flag}]  worst={name} "
                f"per_element={100 * fraction:.4f}% "
                f"per_row_any={100 * row_any:.4f}% "
                f"(limit {100 * CLIPPING_LIMIT:.3f}% per_element)",
                flush=True,
            )

        if self._recorder.bucket.total_samples_stored > 0:
            index_path = self._write_disk()
            print(
                f"[collect] dataset written → {index_path.parent.resolve()}",
                flush=True,
            )
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
        # across runs — any multi-run sampler depends on that.
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
        f"[collect] recording @ {ctrl_hz:.0f} Hz from {sim_hz:.0f} Hz physics "
        f"({recorder.substeps_per_control} substeps/control), {boundary}, "
        f"min episode {recorder.min_control_steps / ctrl_hz:.1f}s",
        flush=True,
    )
    labels = profile.contact_labeling
    print(
        f"[collect] contact labels: substep majority >= {labels.substep_majority:.0%}, "
        f"Schmitt on={labels.on_threshold_n:.1f} N off={labels.off_threshold_n:.1f} N, "
        f"min dwell {labels.min_dwell_steps} steps "
        f"({labels.min_dwell_steps / ctrl_hz * 1e3:.0f} ms)",
        flush=True,
    )
    print(
        f"[collect] episode split: val={recorder.config.val_ratio:.0%} "
        f"test={recorder.config.test_ratio:.0%} (seed {recorder.config.split_seed}), "
        f"keyed on randomization_group_id; failed episodes "
        f"{'stored' if recorder.config.store_failed_episodes else 'discarded'}",
        flush=True,
    )
    print(describe_schema(), flush=True)
    print(describe_bounds(), flush=True)
    sn = recorder.sensor_noise
    if sn.enabled:
        print(
            f"[collect] sensor noise: joint_pos s={sn.joint_pos_std:.4g} rad "
            f"(quantised {sn.joint_pos_quantization_rad:.3g}), "
            f"joint_vel s={sn.joint_vel_std:.4g} rad/s, "
            f"torque s={sn.joint_torque_std:.4g} N*m, "
            f"imu_acc s={sn.imu_acc_std:.4g} m/s^2, "
            f"imu_gyro s={sn.imu_gyro_std:.4g} rad/s, "
            f"imu_random_walk={'on' if sn.imu_random_walk else 'OFF'}",
            flush=True,
        )
    else:
        print("[collect] sensor noise: disabled (ground-truth samples)", flush=True)
    print(f"[collect] run dir -> {run_dir.resolve()}", flush=True)

    hooks = _ActiveCollectionHooks(
        recorder,
        run_dir=str(run_dir),
        profile=profile,
        metadata={
            "prefix": name_prefix,
            "run_id": run_dir.name,
            "robot": robot,
            "scene": scene,
            "gait": gait_type.value,
            "terrain": terrain.value,
            "sim_hz": sim_hz,
            "control_hz": recorder.config.control_hz,
            "substeps_per_control": recorder.substeps_per_control,
            "episode_duration_s": ep_duration,
            "sensor_noise": recorder.sensor_noise.to_metadata(),
            "contact_labeling": profile.contact_labeling.to_metadata(
                recorder.substeps_per_control, DEFAULT_BODY_WEIGHT_N
            ),
            "signal_bounds_sha256": signal_bounds_sha256(),
            "signal_bounds_version": signal_bounds_version(),
            "signal_bounds": load_signal_bounds(),
            "signal_bounds_path": str(default_signal_bounds_path()),
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
            sim_hz=sim_hz,
            episode_duration_s=ep_duration,
            min_episode_duration_s=e.min_episode_duration_s,
            episode_mode=e.episode_mode,
            label_stride=e.label_stride,
            store_failed_episodes=e.store_failed_episodes,
            val_ratio=x.val_ratio,
            test_ratio=x.test_ratio,
            split_seed=x.split_seed,
            contact_labeling=profile.contact_labeling,
        ),
        episode_prefix=episode_prefix,
        sensor_noise=SensorNoise.from_config(dt=1.0 / ctrl_hz, cfg=noise_cfg),
        robot=robot,
        scene=scene,
        run_id=episode_prefix,
    )
