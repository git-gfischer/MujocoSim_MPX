"""
Per-timestep record layout for collected quadruped episodes.

One collected episode is a table with **one row per control step**, not a stack
of pre-cut windows. Windows are assembled at sampling time from
``(episode_id, t)``, so the same timestep is stored once no matter how many
windows overlap it.

Columns carry one of three roles:

``input``
    What a proprioceptive model sees: joint state, torques, IMU, foot kinematics.
``target``
    Ground truth: per-foot contact binaries, GRFs, external base force, base
    linear velocity.
``context``
    Per-step labels used for sampling and error analysis: the step index, the
    rare-contact flag, and the per-foot time since the last contact event.

Episode-constant facts (terrain, gait, friction, payload, outcome, …) live in
:class:`EpisodeMetadata` and are written once per episode, never repeated on
every row.

Frames
------
``imu_gyro``, ``base_lin_vel``   base frame (roll-pitch-yaw aligned with the trunk)
``foot_pos_base``/``foot_vel_base``  yaw-aligned base frame (X forward, Y left, Z up)
``grf_world``, ``external_force``    world frame
``imu_acc``                          world frame — see the note on :data:`EPISODE_COLUMNS`
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
EPISODE_SCHEMA_VERSION = 3

# Foot order used by every per-foot column in this module.
FOOT_ORDER: Tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")

N_FEET = 4
N_JOINTS = 12


@dataclass(frozen=True)
class Column:
    """One column of the per-timestep episode table."""

    name: str
    width: int          # 1 = scalar column, >1 = fixed-size list of that width
    dtype: Any          # NumPy scalar type stored in the column
    role: str           # "input" | "target" | "context"
    unit: str
    frame: str
    doc: str

    @property
    def shape(self) -> Tuple[int, ...]:
        """Per-episode array shape suffix: ``()`` for scalars, ``(width,)`` otherwise."""
        return () if self.width == 1 else (self.width,)


# NOTE on ``imu_acc``: this simulation has no MuJoCo accelerometer sensor, so the
# recorder stores ``data.qacc[:3]`` — the world-frame linear acceleration of the
# floating base, without gravity. It is *not* a proper-acceleration IMU reading in
# the body frame. The frame is declared honestly here so downstream code does not
# assume otherwise.
EPISODE_COLUMNS: Tuple[Column, ...] = (
    # ── context ─────────────────────────────────────────────────────────────
    Column("t", 1, np.int32, "context", "step", "-",
           "Control-step index within the episode; sampling key together with episode_id"),
    Column("time_s", 1, np.float32, "context", "s", "-",
           "Seconds since the first recorded step of this episode"),
    # ── inputs ──────────────────────────────────────────────────────────────
    Column("joint_pos", N_JOINTS, np.float32, "input", "rad", "joint",
           "Joint positions, actuator order"),
    Column("joint_vel", N_JOINTS, np.float32, "input", "rad/s", "joint",
           "Joint velocities, actuator order"),
    Column("joint_torque", N_JOINTS, np.float32, "input", "N*m", "joint",
           "Commanded joint torques, actuator order"),
    Column("imu_acc", 3, np.float32, "input", "m/s^2", "world",
           "Base linear acceleration (see module note: world frame, gravity-free)"),
    Column("imu_gyro", 3, np.float32, "input", "rad/s", "base",
           "Base angular velocity"),
    Column("foot_pos_base", N_FEET * 3, np.float32, "input", "m", "yaw_base",
           "Foot positions from forward kinematics, FL FR RL RR x xyz"),
    Column("foot_vel_base", N_FEET * 3, np.float32, "input", "m/s", "yaw_base",
           "Foot linear velocities from forward kinematics, FL FR RL RR x xyz"),
    # ── targets ─────────────────────────────────────────────────────────────
    Column("contact", N_FEET, np.uint8, "target", "-", "-",
           "Per-foot contact binaries FL FR RL RR, GRF-thresholded"),
    Column("grf_world", N_FEET * 3, np.float32, "target", "N", "world",
           "Ground reaction force per foot, FL FR RL RR x xyz"),
    Column("external_force", 3, np.float32, "target", "N", "world",
           "External perturbation force applied to the base"),
    Column("base_lin_vel", 3, np.float32, "target", "m/s", "base",
           "Base linear velocity expressed in the base frame"),
    # ── context labels ──────────────────────────────────────────────────────
    Column("rare_contact", 1, np.bool_, "context", "-", "-",
           "True when the 4-bit contact pattern is outside the named gait states"),
    Column("dt_since_transition", N_FEET, np.float32, "context", "s", "-",
           "Per foot: seconds since its last touchdown or liftoff"),
)

COLUMNS_BY_NAME: Dict[str, Column] = {c.name: c for c in EPISODE_COLUMNS}

INPUT_COLUMNS: Tuple[Column, ...] = tuple(c for c in EPISODE_COLUMNS if c.role == "input")
TARGET_COLUMNS: Tuple[Column, ...] = tuple(c for c in EPISODE_COLUMNS if c.role == "target")
CONTEXT_COLUMNS: Tuple[Column, ...] = tuple(c for c in EPISODE_COLUMNS if c.role == "context")

# Width of one flattened input vector, for models that want a single feature row.
INPUT_DIM: int = sum(c.width for c in INPUT_COLUMNS)


def columns_by_role(role: str) -> Tuple[Column, ...]:
    """Columns carrying ``role`` ("input", "target" or "context")."""
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
    Thresholding the force magnitude rather than reading MuJoCo's binary contact
    flag avoids solver chatter at touchdown and liftoff.

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


@dataclass
class EpisodeMetadata:
    """
    Episode-constant facts, written once per episode.

    ``friction`` and ``payload_kg`` are read from the live model at episode
    start, so they are populated whether or not reset randomization is enabled.
    """

    episode_id: str
    robot: str = ""
    scene: str = ""
    terrain: str = ""
    gait: str = ""
    timestamp: str = ""              # episode start, ISO-8601 with UTC offset
    ended_at: str = ""
    control_hz: float = 0.0
    n_steps: int = 0
    duration_s: float = 0.0
    friction: float | None = None    # foot sliding mu in effect for this episode
    payload_kg: float | None = None  # extra base mass in effect for this episode
    terminate_by: str = EpisodeOutcome.TRUNCATED.value
    terminate_reason: str = ""
    split: str = "train"
    reset_randomization: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """Flat, JSON- and Arrow-friendly representation (one row per episode)."""
        return {
            "episode_id": self.episode_id,
            "robot": self.robot,
            "scene": self.scene,
            "terrain": self.terrain,
            "gait": self.gait,
            "timestamp": self.timestamp,
            "ended_at": self.ended_at,
            "control_hz": float(self.control_hz),
            "n_steps": int(self.n_steps),
            "duration_s": float(self.duration_s),
            "friction": None if self.friction is None else float(self.friction),
            "payload_kg": None if self.payload_kg is None else float(self.payload_kg),
            "terminate_by": self.terminate_by,
            "terminate_reason": self.terminate_reason,
            "split": self.split,
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


def episode_split_fraction(episode_id: str, seed: int = 0) -> float:
    """
    Deterministic value in ``[0, 1)`` derived from ``episode_id`` and ``seed``.

    Hashing rather than counting keeps the split stable when episodes are
    collected in a different order, across runs, or in parallel processes.
    """
    digest = hashlib.blake2b(
        f"{seed}:{episode_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign_split(
    episode_id: str,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 0,
) -> str:
    """
    Assign a whole episode to "train", "val" or "test".

    Splitting by episode (never by window) keeps overlapping windows of the same
    episode inside one split, so evaluation is not contaminated by near-duplicate
    training samples.
    """
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio > 1.0:
        raise ValueError(
            f"Invalid split ratios: val={val_ratio}, test={test_ratio}"
        )
    u = episode_split_fraction(episode_id, seed=seed)
    if u < test_ratio:
        return "test"
    if u < test_ratio + val_ratio:
        return "val"
    return "train"


def describe_schema() -> str:
    """Human-readable column table, printed at the start of a collection run."""
    lines = [
        f"episode schema v{EPISODE_SCHEMA_VERSION} — one row per control step",
        f"  {'column':<20} {'w':>3}  {'role':<8} {'unit':<8} {'frame':<9} dtype",
    ]
    for column in EPISODE_COLUMNS:
        lines.append(
            f"  {column.name:<20} {column.width:>3}  {column.role:<8} "
            f"{column.unit:<8} {column.frame:<9} {np.dtype(column.dtype).name}"
        )
    return "\n".join(lines)


def flatten_inputs(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate the input columns of an episode into ``(T, INPUT_DIM)`` float32.

    Convenience for models that want one flat feature vector per timestep; the
    stored table keeps the columns separate.
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
