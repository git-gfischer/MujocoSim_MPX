"""
Per-timestep record layout for collected quadruped episodes (schema v4).

One collected episode is a table with **one row per control step**, not a stack
of pre-cut windows. Windows are assembled at sampling time from
``(episode_id, t)``, so the same timestep is stored once no matter how many
windows overlap it.

Roles
-----
Every column declares a ``role``, and the role is the contract:

``input``
    Obtainable on hardware and safe to feed to a network: measured joint state,
    measured actuator force, IMU specific force, and foot kinematics computed by
    forward kinematics **from the measured joint state**.
``target``
    Ground truth the network is trained against.
``context``
    Per-step labels for sampling, stratification and error analysis. Available in
    sim and usually on hardware, but not a network input by default.
``privileged``
    Available in simulation, **not obtainable on hardware, never a network
    input**. Base pose, per-substep force statistics, the controller's own plan,
    the noise-free state, terrain geometry.
``deprecated``
    A v3 column kept so old loaders keep working. Superseded; the ``doc`` names
    the replacement.

``flatten_inputs`` and :data:`INPUT_COLUMNS` select on ``role == "input"``, so
demoting a channel to ``privileged`` removes it from the feature vector by
construction rather than by remembering to update a list somewhere else.

Frames
------
``imu_acc_body``/``imu_gyro_body``   IMU site frame (rotationally identical to
                                     the base; the site is offset in translation)
``base_lin_vel``                     base frame
``foot_pos_base``/``foot_vel_base``  yaw-aligned base frame (X fwd, Y left, Z up)
``grf_world``, ``external_force``    world frame
``base_quat``                        world→base, **wxyz** (MuJoCo convention)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

# Bumped whenever the on-disk format changes. Stamped into every parquet file
# of a run (episodes, episode table, index) so a reader can reject data it does
# not understand.
#
# v4: body-frame IMU + base_quat, debounced contacts from per-substep force,
#     forward kinematics on the measured joint state, measured actuator force,
#     privileged/deprecated roles, terrain geometry, per-episode randomization.
EPISODE_SCHEMA_VERSION = 4

# Foot order used by every per-foot column in this module.
FOOT_ORDER: Tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")

# Joint order within a leg, repeated four times across the 12-vectors.
JOINT_ORDER: Tuple[str, str, str] = ("HAA", "HFE", "KFE")

N_FEET = 4
N_JOINTS = 12

VALID_ROLES = frozenset(
    {"input", "target", "context", "privileged", "deprecated"}
)


@dataclass(frozen=True)
class Column:
    """One column of the per-timestep episode table."""

    name: str
    width: int          # 1 = scalar column, >1 = fixed-size list of that width
    dtype: Any          # NumPy scalar type stored in the column
    role: str           # one of VALID_ROLES
    unit: str
    frame: str
    doc: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"Column '{self.name}' has role {self.role!r}; "
                f"expected one of {sorted(VALID_ROLES)}"
            )
        if self.width < 1:
            raise ValueError(f"Column '{self.name}' has width {self.width}")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Per-episode array shape suffix: ``()`` for scalars, ``(width,)`` otherwise."""
        return () if self.width == 1 else (self.width,)


EPISODE_COLUMNS: Tuple[Column, ...] = (
    # ── keys ────────────────────────────────────────────────────────────────
    Column("t", 1, np.int32, "context", "step", "-",
           "Control-step index within the episode; sampling key with episode_id"),
    Column("time_s", 1, np.float32, "context", "s", "-",
           "Seconds since the first recorded step of this episode"),

    # ── inputs: what the robot can actually measure ─────────────────────────
    Column("joint_pos", N_JOINTS, np.float32, "input", "rad", "joint",
           "Measured joint positions: encoder noise then quantisation applied"),
    Column("joint_vel", N_JOINTS, np.float32, "input", "rad/s", "joint",
           "Measured joint velocities: differentiation noise then low-pass filter"),
    Column("joint_torque_measured", N_JOINTS, np.float32, "input", "N*m", "joint",
           "Applied actuator force after saturation, with current-sensing noise"),
    Column("imu_acc_body", 3, np.float32, "input", "m/s^2", "imu",
           "Specific force measured by the IMU, gravity included"),
    Column("imu_gyro_body", 3, np.float32, "input", "rad/s", "imu",
           "Angular velocity in the IMU frame"),
    Column("foot_pos_base", N_FEET * 3, np.float32, "input", "m", "base",
           "Foot positions in the BODY frame, R_base^T (p_foot_w - p_base_w), "
           "from FK on the MEASURED joint state. Needs no attitude estimate. "
           "FL FR RL RR x xyz"),
    Column("foot_vel_base", N_FEET * 3, np.float32, "input", "m/s", "base",
           "Foot velocities in the BODY frame, J(q_measured) @ dq_measured. "
           "FL FR RL RR x xyz"),
    Column("base_quat_est", 4, np.float32, "input", "-", "world",
           "Attitude ESTIMATED from imu_acc_body/imu_gyro_body by a complementary "
           "filter, wxyz. Yaw is unobservable from an accelerometer and drifts; "
           "use it only for gravity-aligned frames, where yaw cancels"),
    Column("foot_pos_yawbase", N_FEET * 3, np.float32, "input", "m", "yaw_base",
           "Foot positions gravity-aligned with the ESTIMATED attitude (yaw "
           "removed), FL FR RL RR x xyz"),
    Column("foot_vel_yawbase", N_FEET * 3, np.float32, "input", "m/s", "yaw_base",
           "Foot velocities gravity-aligned with the ESTIMATED attitude, "
           "FL FR RL RR x xyz"),

    # ── targets ─────────────────────────────────────────────────────────────
    Column("contact", N_FEET, np.uint8, "target", "-", "-",
           "Debounced per-foot contact FL FR RL RR (PRIMARY LABEL): per-substep "
           "majority, then Schmitt trigger, then minimum dwell"),
    Column("grf_world", N_FEET * 3, np.float32, "target", "N", "world",
           "Ground reaction force per foot at the control instant, FL FR RL RR x xyz"),
    Column("external_force", 3, np.float32, "target", "N", "world",
           "External perturbation force applied to the base"),
    Column("base_lin_vel", 3, np.float32, "target", "m/s", "base",
           "Base linear velocity expressed in the base frame"),

    # ── context ─────────────────────────────────────────────────────────────
    Column("contact_from_height", N_FEET, np.uint8, "context", "-", "-",
           "Contact derived from foot_clearance < 5 mm with the same hysteresis "
           "and dwell as the GRF label. Generated the way the real-world "
           "self-supervised benchmark generates it, so sim/real numbers compare"),
    Column("cmd_segment_id", 1, np.int32, "context", "-", "-",
           "Index of the current velocity-command segment within the episode"),
    Column("contact_raw", N_FEET, np.uint8, "context", "-", "-",
           "Per-substep-majority threshold, no hysteresis and no dwell"),
    Column("dt_since_transition", N_FEET, np.float32, "context", "s", "-",
           "Per foot: seconds since its last touchdown or liftoff, from the "
           "debounced contact"),
    Column("torque_saturated", N_JOINTS, np.uint8, "context", "-", "joint",
           "1 where the commanded torque hit the actuator forcerange"),
    Column("cmd_base_vel", 3, np.float32, "context", "m/s, rad/s", "base",
           "Commanded base velocity (vx, vy, wz)"),
    Column("sensor_stale", 1, np.bool_, "context", "-", "-",
           "True when a dropped packet made this row repeat the previous sample"),

    # ── privileged: simulation-only, never a network input ───────────────────
    Column("base_pos", 3, np.float32, "privileged", "m", "world",
           "Base position in world"),
    Column("grf_mean_n", N_FEET, np.float32, "privileged", "N", "-",
           "Mean per-foot normal force over the control interval"),
    Column("grf_max_n", N_FEET, np.float32, "privileged", "N", "-",
           "Max per-foot normal force over the control interval"),
    Column("joint_torque_cmd", N_JOINTS, np.float32, "privileged", "N*m", "joint",
           "MPC commanded torque before saturation; encodes the planned contact "
           "schedule, so it must not be used as an input"),
    Column("gait_phase", 1, np.float32, "privileged", "-", "-",
           "Controller gait phase in [0, 1)"),
    Column("contact_schedule", N_FEET, np.uint8, "privileged", "-", "-",
           "Contact pattern the controller PLANNED for this step; logged so the "
           "leak is auditable instead of hidden inside the torques"),
    Column("joint_pos_true", N_JOINTS, np.float32, "privileged", "rad", "joint",
           "Noise-free joint positions"),
    Column("joint_vel_true", N_JOINTS, np.float32, "privileged", "rad/s", "joint",
           "Noise-free joint velocities"),
    Column("base_quat", 4, np.float32, "privileged", "-", "world",
           "TRUE base orientation, wxyz (MuJoCo convention), world->base. "
           "Ground-truth attitude is not available on hardware: use "
           "base_quat_est as an input"),
    Column("foot_pos_base_true", N_FEET * 3, np.float32, "privileged", "m", "base",
           "Noise-free twin of foot_pos_base: same BODY frame, same formula, "
           "FK on joint_pos_true"),
    Column("foot_vel_base_true", N_FEET * 3, np.float32, "privileged", "m/s", "base",
           "Noise-free twin of foot_vel_base: J(q_true) @ dq_true, BODY frame"),
    Column("foot_pos_yawbase_true", N_FEET * 3, np.float32, "privileged", "m", "yaw_base",
           "Noise-free twin of foot_pos_yawbase, gravity-aligned with the TRUE "
           "attitude"),
    Column("foot_vel_yawbase_true", N_FEET * 3, np.float32, "privileged", "m/s", "yaw_base",
           "Noise-free twin of foot_vel_yawbase, gravity-aligned with the TRUE "
           "attitude"),
    Column("imu_acc_bias", 3, np.float32, "privileged", "m/s^2", "imu",
           "Realised accelerometer bias (random walk) at this step"),
    Column("imu_gyro_bias", 3, np.float32, "privileged", "rad/s", "imu",
           "Realised gyroscope bias (random walk) at this step"),
    Column("foot_pos_world", N_FEET * 3, np.float32, "privileged", "m", "world",
           "Foot positions in world, FL FR RL RR x xyz"),
    Column("terrain_height_under_foot", N_FEET, np.float32, "privileged", "m", "world",
           "Terrain elevation directly beneath each foot"),
    Column("foot_height_terrain", N_FEET, np.float32, "privileged", "m", "world",
           "Height of the foot SITE (sphere centre) above the terrain beneath it. "
           "A foot in contact reads ~= the collision radius, not 0 — threshold "
           "foot_clearance instead"),
    Column("foot_clearance", N_FEET, np.float32, "privileged", "m", "world",
           "foot_height_terrain minus the foot collision radius: 0 when the foot "
           "surface touches the terrain"),
    Column("base_height_terrain", 1, np.float32, "privileged", "m", "world",
           "Base height above the terrain beneath it"),

    # ── deprecated: v3 columns kept so old loaders keep working ──────────────
    Column("imu_acc", 3, np.float32, "deprecated", "m/s^2", "world",
           "v3 world-frame gravity-free base acceleration. Superseded by "
           "imu_acc_body, which is what an IMU measures. Do not use as input."),
    Column("imu_gyro", 3, np.float32, "deprecated", "rad/s", "base",
           "v3 base angular velocity. Superseded by imu_gyro_body."),
    Column("joint_torque", N_JOINTS, np.float32, "deprecated", "N*m", "joint",
           "v3 alias of joint_torque_cmd (commanded, pre-saturation). Superseded "
           "by joint_torque_measured. Do not use as input."),
    Column("rare_contact", 1, np.bool_, "deprecated", "-", "-",
           "v3 flag for a contact pattern outside the 12 named gait states. "
           "Superseded by the SINGLE_<foot> contact_state names."),
)

COLUMNS_BY_NAME: Dict[str, Column] = {c.name: c for c in EPISODE_COLUMNS}

INPUT_COLUMNS: Tuple[Column, ...] = tuple(c for c in EPISODE_COLUMNS if c.role == "input")
TARGET_COLUMNS: Tuple[Column, ...] = tuple(c for c in EPISODE_COLUMNS if c.role == "target")
CONTEXT_COLUMNS: Tuple[Column, ...] = tuple(c for c in EPISODE_COLUMNS if c.role == "context")
PRIVILEGED_COLUMNS: Tuple[Column, ...] = tuple(
    c for c in EPISODE_COLUMNS if c.role == "privileged"
)
DEPRECATED_COLUMNS: Tuple[Column, ...] = tuple(
    c for c in EPISODE_COLUMNS if c.role == "deprecated"
)

# Width of one flattened input vector, for models that want a single feature row.
INPUT_DIM: int = sum(c.width for c in INPUT_COLUMNS)


def columns_by_role(role: str) -> Tuple[Column, ...]:
    """Columns carrying ``role``."""
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown role {role!r}; expected one of {sorted(VALID_ROLES)}")
    return tuple(c for c in EPISODE_COLUMNS if c.role == role)


def structured_dtype() -> np.dtype:
    """NumPy structured dtype for one episode row (used by tests and tooling)."""
    return np.dtype([(c.name, c.dtype, c.shape) for c in EPISODE_COLUMNS])


def empty_arrays(n_steps: int) -> Dict[str, np.ndarray]:
    """Allocate a zeroed column dict for ``n_steps`` rows."""
    return {
        c.name: np.zeros((n_steps, *c.shape), dtype=c.dtype)
        for c in EPISODE_COLUMNS
    }


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    """
    Check that ``arrays`` holds every column with a consistent row count.

    Returns the row count. Raises ``ValueError`` on a missing column, a wrong
    per-row width, or a length mismatch between columns.
    """
    missing = [c.name for c in EPISODE_COLUMNS if c.name not in arrays]
    if missing:
        raise ValueError(f"Episode arrays are missing columns: {missing}")

    n_steps = -1
    for column in EPISODE_COLUMNS:
        array = np.asarray(arrays[column.name])
        expected = (array.shape[0], *column.shape)
        if array.shape != expected:
            raise ValueError(
                f"Column '{column.name}' has shape {array.shape}, expected "
                f"(T, {', '.join(str(d) for d in column.shape) or ''}) "
                f"with width {column.width}"
            )
        if n_steps < 0:
            n_steps = array.shape[0]
        elif array.shape[0] != n_steps:
            raise ValueError(
                f"Column '{column.name}' has {array.shape[0]} rows but earlier "
                f"columns have {n_steps}"
            )
    return n_steps


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED PER-STEP LABELS
# ══════════════════════════════════════════════════════════════════════════════

def contact_bits_from_grf(
    grf_world: np.ndarray,
    contact_force_threshold: float,
) -> np.ndarray:
    """
    Threshold per-foot GRF magnitude into contact binaries.

    ``grf_world`` is ``(T, 4, 3)`` or ``(T, 12)`` in FL FR RL RR x xyz order.

    This is the plain single-threshold rule. Collected episodes use the
    hysteresis + dwell labeller in
    :mod:`mpx.utils.dataset_collection.contact_labeling` instead; this function
    remains for reading v3 data and for analysis.

    Returns ``(T, 4)`` ``uint8``.
    """
    grf = np.asarray(grf_world, dtype=np.float64).reshape(-1, N_FEET, 3)
    magnitude = np.linalg.norm(grf, axis=2)
    return (magnitude > float(contact_force_threshold)).astype(np.uint8)


def dt_since_transition(contact: np.ndarray, dt: float) -> np.ndarray:
    """
    Per-foot seconds since that foot's last touchdown or liftoff.

    The timer resets to ``0`` on every contact-state change of that foot, so the
    value is small exactly at a contact event and grows through steady stance and
    steady swing alike. The first step of an episode counts as ``0`` for every
    foot: nothing before it is observed, so no earlier event can be claimed.

    Feed this the **debounced** contact. Deriving it from a chattering label
    resets the timer on every one-frame glitch and destroys the signal.

    ``contact`` is ``(T, 4)`` binary; returns ``(T, 4)`` ``float32``.
    """
    bits = np.asarray(contact, dtype=np.uint8).reshape(-1, N_FEET)
    n_steps = bits.shape[0]
    elapsed = np.zeros((n_steps, N_FEET), dtype=np.float32)
    if n_steps == 0:
        return elapsed

    step = float(dt)
    for t in range(1, n_steps):
        changed = bits[t] != bits[t - 1]
        elapsed[t] = np.where(changed, 0.0, elapsed[t - 1] + step)
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE-LEVEL METADATA
# ══════════════════════════════════════════════════════════════════════════════

class EpisodeOutcome(str, Enum):
    """How an episode ended."""

    SUCCESS = "success"     # the task goal was reached
    FAILURE = "failure"     # the robot fell or lost the commanded pose
    TRUNCATED = "truncated" # the duration cap or shutdown closed a healthy episode


# Reason strings passed by the simulators, mapped to a coarse outcome.
OUTCOME_BY_REASON: Dict[str, EpisodeOutcome] = {
    "goal_reached":   EpisodeOutcome.SUCCESS,
    "pose_held":      EpisodeOutcome.SUCCESS,
    "crash":          EpisodeOutcome.FAILURE,
    "crashed":        EpisodeOutcome.FAILURE,
    "pose_lost":      EpisodeOutcome.FAILURE,
    "fall":           EpisodeOutcome.FAILURE,
    "duration_cap":   EpisodeOutcome.TRUNCATED,
    "fixed_duration": EpisodeOutcome.TRUNCATED,
    "shutdown":       EpisodeOutcome.TRUNCATED,
    "gate":           EpisodeOutcome.TRUNCATED,
    "event":          EpisodeOutcome.TRUNCATED,
}


def classify_termination(reason: str) -> EpisodeOutcome:
    """Map a simulator reason string to a coarse outcome (unknown → truncated)."""
    return OUTCOME_BY_REASON.get(str(reason), EpisodeOutcome.TRUNCATED)


def episode_timestamp() -> str:
    """ISO-8601 timestamp including the local UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def randomization_group_id(parameters: Mapping[str, Any]) -> str:
    """
    Stable 16-hex-character id for one set of randomization parameters.

    Episodes sharing an id were collected under identical domain randomization,
    so they are near-duplicates of each other and must never be split across
    train and test. The manifest enforces that; this is how it recognises them.
    """
    payload = ";".join(
        f"{key}={float(value):.10g}" if isinstance(value, (int, float))
        and not isinstance(value, bool) else f"{key}={value}"
        for key, value in sorted((parameters or {}).items())
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


@dataclass
class EpisodeMetadata:
    """
    Episode-constant facts, written once per episode.

    ``friction`` and ``payload_kg`` are read from the live model at episode
    start, so they are populated whether or not reset randomization is enabled.

    ``split`` is **deprecated**: train/val/test now comes from
    ``datasets/manifest.json`` at load time, keyed on
    ``randomization_group_id``. The field survives as ``split_assigned``, a
    cached convenience regenerated from the manifest.
    """

    episode_id: str
    run_id: str = ""
    robot: str = ""
    scene: str = ""
    terrain: str = ""
    gait: str = ""
    # Task sub-mode, e.g. "locomotion", "stand_4leg", "stand_3leg_FL". Lets a
    # standing pilot be stratified without parsing run folder names.
    mode: str = ""
    timestamp: str = ""              # episode start, ISO-8601 with UTC offset
    ended_at: str = ""
    control_hz: float = 0.0
    sim_hz: float = 0.0
    substeps_per_control: int = 0
    n_steps: int = 0
    duration_s: float = 0.0
    friction: float | None = None    # foot sliding mu in effect for this episode
    payload_kg: float | None = None  # extra base mass in effect for this episode
    terminate_by: str = EpisodeOutcome.TRUNCATED.value
    terminate_reason: str = ""
    split_assigned: str = "train"
    seed: int | None = None
    randomization_group_id: str = ""
    reset_randomization: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.randomization_group_id:
            self.randomization_group_id = randomization_group_id(
                self.reset_randomization
            )

    def to_row(self) -> Dict[str, Any]:
        """Flat, JSON- and Arrow-friendly representation (one row per episode)."""
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "robot": self.robot,
            "scene": self.scene,
            "terrain": self.terrain,
            "gait": self.gait,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "ended_at": self.ended_at,
            "control_hz": float(self.control_hz),
            "sim_hz": float(self.sim_hz),
            "substeps_per_control": int(self.substeps_per_control),
            "n_steps": int(self.n_steps),
            "duration_s": float(self.duration_s),
            "friction": None if self.friction is None else float(self.friction),
            "payload_kg": None if self.payload_kg is None else float(self.payload_kg),
            "terminate_by": self.terminate_by,
            "terminate_reason": self.terminate_reason,
            "split_assigned": self.split_assigned,
            "seed": None if self.seed is None else int(self.seed),
            "randomization_group_id": self.randomization_group_id,
            "reset_randomization": {
                str(k): float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in (self.reset_randomization or {}).items()
            },
        }


@dataclass
class EpisodeRecord:
    """One finished episode: the per-step table plus its metadata."""

    metadata: EpisodeMetadata
    arrays: Dict[str, np.ndarray]

    def __post_init__(self) -> None:
        n_steps = validate_arrays(self.arrays)
        self.metadata.n_steps = n_steps
        if self.metadata.control_hz > 0:
            self.metadata.duration_s = n_steps / float(self.metadata.control_hz)

    @property
    def n_steps(self) -> int:
        return int(self.metadata.n_steps)


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE-LEVEL TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")


def episode_split_fraction(group_id: str, seed: int = 0) -> float:
    """
    Deterministic value in ``[0, 1)`` derived from a key and ``seed``.

    Hashing rather than counting keeps the split stable when episodes are
    collected in a different order, across runs, or in parallel processes.
    """
    digest = hashlib.blake2b(
        f"{seed}:{group_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign_split(
    group_id: str,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 0,
) -> str:
    """
    Assign a randomization group to "train", "val" or "test".

    Pass the ``randomization_group_id``, not the episode id: episodes sharing
    randomization are near-duplicates, and splitting between them leaks the test
    set into training. The manifest is the authority at load time; this is the
    collection-time cache that fills ``split_assigned``.
    """
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio > 1.0:
        raise ValueError(
            f"Invalid split ratios: val={val_ratio}, test={test_ratio}"
        )
    u = episode_split_fraction(group_id, seed=seed)
    if u < test_ratio:
        return "test"
    if u < test_ratio + val_ratio:
        return "val"
    return "train"


def describe_schema() -> str:
    """Human-readable column table, printed at the start of a collection run."""
    lines = [
        f"episode schema v{EPISODE_SCHEMA_VERSION} — one row per control step",
        f"  {'column':<26} {'w':>3}  {'role':<11} {'unit':<10} {'frame':<9} dtype",
    ]
    for column in EPISODE_COLUMNS:
        lines.append(
            f"  {column.name:<26} {column.width:>3}  {column.role:<11} "
            f"{column.unit:<10} {column.frame:<9} {np.dtype(column.dtype).name}"
        )
    lines.append(
        f"  inputs={len(INPUT_COLUMNS)} (dim {INPUT_DIM})  "
        f"targets={len(TARGET_COLUMNS)}  context={len(CONTEXT_COLUMNS)}  "
        f"privileged={len(PRIVILEGED_COLUMNS)}  deprecated={len(DEPRECATED_COLUMNS)}"
    )
    return "\n".join(lines)


def flatten_inputs(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate the ``input`` columns of an episode into ``(T, INPUT_DIM)`` float32.

    Only ``role == "input"`` columns are included, so privileged and deprecated
    channels cannot reach a model through this path.
    """
    return np.concatenate(
        [
            np.asarray(arrays[c.name], dtype=np.float32).reshape(
                np.asarray(arrays[c.name]).shape[0], -1
            )
            for c in INPUT_COLUMNS
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def input_channel_slices() -> Dict[str, Tuple[int, int]]:
    """Start/end offsets of each input column inside :func:`flatten_inputs`."""
    slices: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for column in INPUT_COLUMNS:
        slices[column.name] = (offset, offset + column.width)
        offset += column.width
    return slices


def iter_column_names(columns: Iterable[Column] | None = None) -> Sequence[str]:
    """Names of ``columns`` (default: every episode column), in table order."""
    return [c.name for c in (columns if columns is not None else EPISODE_COLUMNS)]
