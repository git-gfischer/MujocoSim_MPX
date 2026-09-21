"""
dataset_collection/dataset_bucket_system.py
===========================================
Balanced dataset collection for quadruped proprioceptive signals collected in
MuJoCo. Designed for multi-task learning of:

    - Per-foot contact classification   (4 binaries, FL FR RL RR)
    - Ground reaction force estimation  (regression, 4x3 world-frame [N])
    - External base force estimation    (regression, 3D vector)
    - Base linear velocity estimation   (regression, 3D vector, base frame)

Storage model
-------------
Episodes are stored **once, per timestep**, as parquet tables (see
:mod:`mpx.utils.dataset_collection.dataset_schema` for the column layout). The
bucket system stores no trajectory data at all: a bucket holds lightweight
:class:`SampleRef` entries naming ``(episode_id, t)`` — one *labelled timestep*
— plus the few label statistics needed for balancing and diagnostics.

Collection is deliberately **window-length agnostic**. A sample is a labelled
timestep, and everything the buckets balance (contact state, perturbation,
terrain, gait) is read at that one step. Choosing ``W`` and cutting
``[t - W + 1 … t]`` out of the episode table is the training-time job of the
PyTorch dataset, which just skips index rows with ``t < W - 1``. One collected
run therefore serves any window length.

Bucket key: (contact_state, perturbation_active, terrain, gait_type)

Balancing strategy
------------------
    - Reservoir sampling per bucket (uniform coverage over all seen samples)
    - Perturbation ratio enforcement (hard minimum fraction of perturbed samples)
    - GRF diversity monitored analytically post-collection (std per bucket)
      rather than enforced via binning at collection time
    - Train/val/test assigned per **episode**, so overlapping windows cut from
      one episode never straddle a split boundary

An example can be found at the bottom of the file.
"""

import hashlib
import random
import json
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mpx.config.sim_config.config_dataset_bucket import (
    DatasetBucketConfig,
    dataset_bucket_config,
)
from mpx.utils.dataset_collection.contact_labeling import ContactLabelConfig
from mpx.utils.dataset_collection.dataset_schema import (
    EPISODE_SCHEMA_VERSION,
    FOOT_ORDER,
    EpisodeMetadata,
    EpisodeRecord,
)
from mpx.utils.dataset_collection.episode_storage import (
    INDEX_FILENAME,
    EpisodeStore,
)
from mpx.utils.dataset_collection.operating_regime import (
    REGIME_ORDER,
    OperatingRegimeConfig,
    base_tilt_deg,
    classify as classify_regime,
)

DATASET_SUMMARY_FILENAME = "dataset_summary.json"

# Fallback total weight [N] of a bare Go2. The perturbation axis is binned as a
# fraction of body weight, so a run whose episodes carry a measured
# ``body_weight_n`` uses that instead and this is only the default.
DEFAULT_BODY_WEIGHT_N = 176.0
# Kept in step with EPISODE_SCHEMA_VERSION: they version the same on-disk
# generation, and a run reporting 4 and 3 in the same metadata file only
# looked like a bug.
DATASET_SUMMARY_SCHEMA_VERSION = 4


# ══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class GaitType(Enum):
    TROT       = "trot"
    CRAWL      = "crawl"
    PACE       = "pace"
    BOUND      = "bound"
    BALANCE    = "balance"
    TRANSITION = "transition"   # Gait-to-gait episodes — stored separately


class TerrainType(Enum):
    FLAT   = "flat"
    ROUGH  = "rough"
    STAIRS = "stairs"


# ══════════════════════════════════════════════════════════════════════════════
# CONTACT STATE DEFINITIONS  (FL, FR, RL, RR) — 1 = contact, 0 = swing
# ══════════════════════════════════════════════════════════════════════════════

# The 12 nominal gait states. This is the taxonomy used for bucket keys and
# summaries — not the training label, which is the raw 4-bit vector stored on
# every timestep.
NOMINAL_GAIT_STATES: Dict[str, Tuple[int, int, int, int]] = {
    "FULL":        (1, 1, 1, 1),   # 1111 — standing / crawl / 4-leg balance
    "SWING_RR":    (1, 1, 1, 0),   # 1110 — rear-right  swing
    "SWING_RL":    (1, 1, 0, 1),   # 1101 — rear-left   swing
    "SWING_FR":    (1, 0, 1, 1),   # 1011 — front-right swing
    "SWING_FL":    (0, 1, 1, 1),   # 0111 — front-left  swing
    "HIND_PAIR":   (0, 0, 1, 1),   # 0011 — hind pair stance
    "FRONT_PAIR":  (1, 1, 0, 0),   # 1100 — front pair stance
    "IPSIL_FL_RL": (1, 0, 1, 0),   # 1010 — ipsilateral left  (FL + RL)
    "DIAG_FL_RR":  (1, 0, 0, 1),   # 1001 — diagonal FL + RR (trot phase)
    "DIAG_FR_RL":  (0, 1, 1, 0),   # 0110 — diagonal FR + RL (trot phase)
    "IPSIL_FR_RR": (0, 1, 0, 1),   # 0101 — ipsilateral right (FR + RR)
    "FLIGHT":      (0, 0, 0, 0),   # 0000 — no foot contact (bound flight)
}

# The remaining four patterns are the single-support stances. v3 collapsed all
# four into one "RARE" bucket, so two different bit patterns shared a name and
# the label was ambiguous. Each now gets its own name; the 16 patterns and the 16
# names are in bijection.
SINGLE_SUPPORT_STATES: Dict[str, Tuple[int, int, int, int]] = {
    "SINGLE_FL": (1, 0, 0, 0),
    "SINGLE_FR": (0, 1, 0, 0),
    "SINGLE_RL": (0, 0, 1, 0),
    "SINGLE_RR": (0, 0, 0, 1),
}

VALID_CONTACT_STATES: Dict[str, Tuple[int, int, int, int]] = {
    **NOMINAL_GAIT_STATES,
    **SINGLE_SUPPORT_STATES,
}

# v3 label for the four single-support patterns. Kept only for reading old data.
RARE_CONTACT_STATE = "RARE"

CONTACT_STATE_BITS: Dict[str, str] = {
    name: "".join(str(b) for b in pattern)
    for name, pattern in VALID_CONTACT_STATES.items()
}

BINARY_TO_CONTACT_STATE: Dict[Tuple[int, int, int, int], str] = {
    pattern: name for name, pattern in VALID_CONTACT_STATES.items()
}

# Kept so old class-index datasets and summaries can still be read.
CONTACT_STATE_TO_IDX: Dict[str, int] = {
    k: i for i, k in enumerate(VALID_CONTACT_STATES)
}
IDX_TO_CONTACT_STATE: Dict[int, str] = {
    v: k for k, v in CONTACT_STATE_TO_IDX.items()
}


# Standard leg phase offsets, FL FR RL RR, as fractions of one gait cycle.
# Two legs sharing an offset swing together.
GAIT_PHASE_OFFSETS: Dict[str, Tuple[float, float, float, float]] = {
    "trot":  (0.5, 0.0, 0.0, 0.5),   # diagonal pairs
    "pace":  (0.5, 0.0, 0.5, 0.0),   # lateral pairs
    "bound": (0.5, 0.5, 0.0, 0.0),   # front pair / hind pair
    "crawl": (0.25, 0.75, 0.0, 0.5), # one leg at a time
}


def gait_from_phase_offsets(timer_t, tolerance: float = 0.05) -> GaitType:
    """
    Identify the gait from the controller's per-leg phase offsets.

    The gait label must come from what the controller is actually doing. The
    audited run hardcoded ``GaitType.TROT`` in the simulator while
    ``config_go2.timer_t`` held a crawl pattern, so every episode — and every run
    folder name — claimed a gait the robot never walked.

    Falls back to ``TRANSITION`` for an unrecognised pattern rather than guessing.
    """
    offsets = np.asarray(timer_t, dtype=np.float64).reshape(-1)[:4] % 1.0
    for name, reference in GAIT_PHASE_OFFSETS.items():
        expected = np.asarray(reference, dtype=np.float64) % 1.0
        # The same gait may be written with any leg taken as phase zero, so try
        # every whole-cycle shift. Compare element-wise, never sorted: sorting
        # discards WHICH leg holds which phase, and trot [0.5,0,0,0.5] and pace
        # [0.5,0,0.5,0] sort to the same multiset.
        for anchor in range(4):
            shifted = (offsets - offsets[anchor]) % 1.0
            aligned = (expected - expected[anchor]) % 1.0
            # Phases live on a circle, so 0.98 and 0.00 are 0.02 apart.
            delta = np.abs(shifted - aligned)
            delta = np.minimum(delta, 1.0 - delta)
            if np.all(delta <= tolerance):
                return GaitType(name)
    return GaitType.TRANSITION


def contact_state_name(bits: Sequence[int]) -> str:
    """
    Name for a 4-bit contact pattern.

    All 16 patterns have a name: the 12 nominal gait states plus the four
    ``SINGLE_<foot>`` single-support stances. Nothing falls through to a shared
    catch-all, so a name identifies a pattern uniquely.
    """
    key = tuple(int(b) for b in bits)
    return BINARY_TO_CONTACT_STATE.get(key, RARE_CONTACT_STATE)


def is_rare_contact(bits: Sequence[int]) -> bool:
    """
    True when the pattern is single support, i.e. outside the nominal gait set.

    This is the v3 ``rare_contact`` flag, kept as a deprecated column. Prefer
    testing ``contact_state.startswith("SINGLE_")``, which also says which foot.
    """
    return tuple(int(b) for b in bits) in set(SINGLE_SUPPORT_STATES.values())


def contact_bits_string(bits: Sequence[int]) -> str:
    """``(1, 1, 1, 0)`` → ``"1110"``, in FL FR RL RR order."""
    return "".join(str(int(b)) for b in bits)


def resolve_run_directory(
    *,
    prefix: str,
    robot: str,
    scene: str,
    gait: str,
    terrain: str,
    output_root_dir: str = "datasets",
    run_folder_pattern: str = "{prefix}_{robot}_{scene}_{gait}_{timestamp}",
    use_timestamp: bool = True,
    timestamp: str | None = None,
    collect_out: str | None = None,
) -> Path:
    """
    Build (and create) the run directory for a collection session.

    One run directory holds ``episodes/``, ``episodes.parquet`` and
    ``index.parquet``. If ``collect_out`` is set it names the run directory:
    absolute paths are used as-is, relative ones are placed under
    ``output_root_dir``.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = dict(
        prefix=prefix,
        robot=robot,
        scene=scene,
        gait=gait,
        terrain=terrain,
        timestamp=ts if use_timestamp else "",
    )

    if collect_out:
        run_dir = Path(collect_out)
        # A leftover ".npz" from the previous storage format names a run folder now.
        if run_dir.suffix:
            run_dir = run_dir.with_suffix("")
        if not run_dir.is_absolute():
            run_dir = Path(output_root_dir) / run_dir
    else:
        run_dir = Path(output_root_dir) / run_folder_pattern.format(**fmt).strip("_")

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

# ── bucket axes derived from the row ─────────────────────────────────────────

# Command regimes. Binned on the COMMAND, not the realised velocity: the command
# is what the collection config controls, so a homogeneous bucket points straight
# at the knob that needs widening.
SPEED_BINS: Tuple[str, ...] = (
    "stopped", "slow", "medium", "fast", "reverse", "turning",
)

PERTURBATION_LEVELS: Tuple[str, str, str] = ("none", "small", "large")


def speed_bin(cmd_base_vel: Sequence[float]) -> str:
    """Coarse command regime from ``cmd_base_vel`` (vx, vy, yaw_rate)."""
    values = np.asarray(cmd_base_vel, dtype=np.float64).reshape(-1)
    vx, vy, wz = (float(values[i]) if values.size > i else 0.0 for i in range(3))
    speed = float(np.hypot(vx, vy))
    if speed < 0.05 and abs(wz) < 0.10:
        return "stopped"
    if vx < -0.10:
        return "reverse"
    if abs(wz) >= 0.30 and speed < 0.30:
        return "turning"
    if speed < 0.35:
        return "slow"
    if speed < 0.70:
        return "medium"
    return "fast"


def perturbation_boundaries(
    body_weight_n: float,
    config: DatasetBucketConfig | None = None,
) -> Tuple[float, float]:
    """
    The ``none``/``small`` and ``small``/``large`` boundaries in newtons.

    The floor is a fraction of body weight — a detection threshold has to scale
    with the robot, since payload is randomized. The upper boundary splits the
    **sampler's own configured range**, because a fixed body-weight fraction
    leaves ``large`` empty whenever the sampler maximum falls below it. That is
    exactly what happened: 0.35 x 176 N = 61.6 N against a sampler maximum of
    50.0 N, so the audited run declared a level that could not occur and
    ``BucketKey`` carried a value with zero samples.
    """
    cfg = config if config is not None else dataset_bucket_config
    floor = cfg.perturbation_small_bw_frac * float(body_weight_n)
    sampler_range = getattr(cfg, "perturbation_force_range_n", None)
    if not sampler_range:
        return floor, cfg.perturbation_large_bw_frac * float(body_weight_n)
    low, high = (float(v) for v in sampler_range)
    return floor, 0.5 * (max(low, floor) + high)


def perturbation_level(
    force_n: float,
    body_weight_n: float,
    config: DatasetBucketConfig | None = None,
) -> str:
    """
    ``"none"`` / ``"small"`` / ``"large"`` from ``|F_ext|``.

    Three levels rather than a boolean because the advice "add perturbation
    magnitude range, not just presence" cannot be acted on when a 6 N nudge and a
    60 N shove land in the same bucket. See :func:`perturbation_boundaries` for
    where the two numbers come from.
    """
    cfg = config if config is not None else dataset_bucket_config
    floor, mid = perturbation_boundaries(body_weight_n, cfg)
    magnitude = float(force_n)
    if magnitude < floor:
        return "none"
    if magnitude < mid:
        return "small"
    return "large"


def schedule_mismatch_masks(
    contact: np.ndarray,
    schedule: np.ndarray,
    edge_radius: int = 2,
    sustained_min_run: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Decompose plan-vs-physics disagreement into jitter and real disagreement.

    ``contact`` and ``schedule`` are ``(T, 4)`` uint8.

    Returns
    -------
    per_foot  : (T, 4) bool — the raw disagreement
    edge      : (T, 4) bool — within ``edge_radius`` steps of a contact transition
    sustained : (T, 4) bool — inside a run of >= ``sustained_min_run`` consecutive
                mismatched frames

    Only ``sustained`` is a defensible benchmark subset. Measured on the audited
    run: 88% of mismatched frames sit within two control steps of a contact
    transition, with no systematic lead or lag (the best global shift was 0 in
    all 28 episodes), so they are touchdown/liftoff timing jitter between planner
    and physics rather than slips. A torque-reading model fails on those frames
    for trivial reasons too, so the raw flag does not isolate the subset it
    claims to. ``sustained`` covers 33% of mismatched frames.
    """
    contact = np.asarray(contact, dtype=np.int8).reshape(-1, 4)
    schedule = np.asarray(schedule, dtype=np.int8).reshape(-1, 4)
    n_steps = contact.shape[0]
    per_foot = contact != schedule
    edge = np.zeros_like(per_foot)
    sustained = np.zeros_like(per_foot)
    steps = np.arange(n_steps)

    for foot in range(4):
        transitions = np.flatnonzero(np.diff(contact[:, foot])) + 1
        if transitions.size:
            distance = np.min(
                np.abs(steps[:, None] - transitions[None, :]), axis=1
            )
            edge[:, foot] = per_foot[:, foot] & (distance <= edge_radius)

        series = per_foot[:, foot]
        breaks = np.flatnonzero(np.diff(series.astype(np.int8))) + 1
        start = 0
        for segment in np.split(series, breaks):
            if segment.size and segment[0] and segment.size >= sustained_min_run:
                sustained[start : start + segment.size, foot] = True
            start += segment.size

    return {"per_foot": per_foot, "edge": edge, "sustained": sustained}


@dataclass(slots=True)
class SampleRef:
    """
    One training sample: a labelled timestep, stored as a reference.

    ``t`` is the step whose contact state, GRF and perturbation flag the labels
    describe. A sampler picks its own window length ``W`` and materializes the
    window by slicing ``[t - W + 1 … t]`` out of
    ``episodes/<episode_id>.parquet``; no window length is stored here.

    The few label statistics kept here are what balancing and the summary need;
    everything else is read back from the episode file on demand.

    ``slots=True`` because this is one object per timestep across every gait and
    terrain: a plain dataclass would be several GB of RAM for data already on
    disk.
    """

    episode_id:    str
    t:             int
    contact_state: str                        # one of the 16 named patterns
    contact_bits:  Tuple[int, int, int, int]  # FL FR RL RR
    # Per-foot force magnitude over the control interval, FL FR RL RR [N], taken
    # from the SUBSTEP-AVERAGED grf_base — not the instant the control step
    # happened to sample. The instantaneous grf_world drops to exactly 0 N
    # mid-stance on ~1% of frames (real micro-bounce, aliased at 50 Hz) and
    # poisons every force statistic built on it.
    grf_per_foot_n: Tuple[float, float, float, float]
    # max_i |f_xy,i| / f_z,i over loaded feet: friction utilisation, the
    # slip-relevant axis. 0.0 when no foot is loaded.
    grf_tangential_ratio_max: float
    external_force_n:   float                 # |F_ext| at t
    perturbation_level: str                   # none | small | large
    speed_bin:     str                        # commanded regime at t
    gait_type:     GaitType
    terrain_type:  TerrainType
    # How degraded locomotion was at this step. NOT a bucket-key field: four
    # levels would multiply the key and most combinations are empty. It is a
    # per-bucket statistic (frac_nominal) and an index column instead.
    operating_regime: str = "nominal"
    base_height_terrain: float = 0.0
    base_tilt_deg: float = 0.0
    non_foot_contact_n: float = 0.0
    post_failure:  bool = False               # after the robot became unrecoverable
    valid:         bool = True                # usable for training
    schedule_mismatch: bool = False           # any foot is not where the plan said
    schedule_mismatch_sustained: bool = False # inside a run of >= 3 such frames
    schedule_mismatch_feet: Tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def rare_contact(self) -> bool:
        """True when the pattern is single support (deprecated v3 flag)."""
        return self.contact_state in SINGLE_SUPPORT_STATES

    @property
    def perturbation_active(self) -> bool:
        """Back-compat with the v4 boolean perturbation axis."""
        return self.perturbation_level != "none"

    @property
    def grf_total_n(self) -> float:
        """Sum of per-foot mean force [N]. Kept for the index column."""
        return float(sum(self.grf_per_foot_n))

    @property
    def grf_load_share_max(self) -> float:
        """
        Largest single-foot share of the total load, in [0.25, 1.0].

        0.25 is a perfectly even four-foot stance, 1.0 is single support. This is
        what the GRF head has to discriminate and what the total magnitude —
        pinned near body weight by statics — cannot see.
        """
        total = self.grf_total_n
        return float(max(self.grf_per_foot_n) / total) if total > 1e-6 else 0.0


@dataclass(frozen=True)
class BucketKey:
    """
    Immutable key that uniquely identifies one bucket.
    Frozen so it can be used as a dict key.

    ``speed_bin`` and ``perturbation_level`` are in the key because the diversity
    story depends on them. GRF diversity is enforced upstream by varying speed,
    friction, payload and perturbation magnitude; with none of those in the key,
    a diagnostic can say a bucket is homogeneous but never which axis collapsed.
    The audited run demonstrates it: ``FULL | perturb=false | flat | trot`` held
    9,627 samples and looked healthy, and every one came from a single constant
    command per episode.

    Friction and payload stay **out** of the key — they are continuous and would
    explode it. They are tracked per bucket instead, through
    ``n_randomization_groups`` in :mod:`grf_diversity`: a bucket fed by one
    randomization draw is homogeneous however many samples it holds.

    Maximum theoretical buckets: 16 contact states × 3 perturbation levels
    × 6 speed bins × 3 terrains × 6 gaits — of which only a small subset is
    physically reachable. ``active_buckets`` counts the keys actually hit.
    """
    contact_state:       str
    perturbation_level:  str        # "none" | "small" | "large"
    speed_bin:           str        # stopped|slow|medium|fast|reverse|turning
    terrain:             TerrainType
    gait_type:           GaitType

    @property
    def perturbation_active(self) -> bool:
        """Back-compat with the v4 boolean axis."""
        return self.perturbation_level != "none"

    def label(self) -> str:
        """Stable human-readable key used in the index and JSON summaries."""
        return (
            f"{self.contact_state} | pert={self.perturbation_level} | "
            f"{self.speed_bin} | {self.terrain.value} | {self.gait_type.value}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# BUCKET SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class DatasetBucketSystem:
    """
    Manages balanced dataset collection for quadruped proprioceptive data.

    Core responsibilities
    ─────────────────────
    1. Contact state naming
       Reads the ``contact`` column — the debounced label produced by
       :mod:`mpx.utils.dataset_collection.contact_labeling` — and names its 4-bit
       pattern. It never re-derives the label: there is exactly one definition of
       contact in this codebase and it does not live here. Patterns outside the
       12 named gait states get their own ``SINGLE_<foot>`` names and are kept
       and bucketed, never dropped.

    2. Label enumeration
       Records one reference per labelled timestep of each episode. No window
       length is involved: a sampler cuts windows later, and because a window
       is always taken from a single episode file it can never cross an
       episode boundary.

    3. Bucket assignment (bookkeeping only)
       Each (contact_state, perturbation_level, speed_bin, terrain, gait)
       combination gets
       its own bucket, and ``bucket_key`` travels on every index row so a sampler
       can stratify on it. Buckets no longer **drop** samples: v4.0 equalised the
       three majority contact states and silently discarded 2,069 of 23,994 rows
       (11.2% of `0110`, 10.9% of `1001`, 6.5% of `1111`, 0% of everything else),
       which changed the class prior — accuracy over that index was not accuracy
       over the real distribution, and comparing it to MI-HGNN / ECNN numbers
       computed on unbalanced data would be apples to oranges. Balancing is now a
       training-time **weight**, not a deletion.

    4. Perturbation ratio enforcement
       Hard minimum fraction of stored samples that must be perturbation-active,
       monitored globally AND per bucket: the stated rationale for the axis is
       per-state coverage, so the aggregate is the wrong number. The audited
       run's global 20.9% hid a per-state range of 9.4% to 66.7%.

    5. GRF diversity monitoring  (post-collection, NOT here)
       The bucket system only counts. The diversity statistics live in
       :mod:`mpx.utils.dataset_collection.grf_diversity`, which reads
       ``index.parquet`` after the fact, so the statistic can be changed without
       recollecting.

    6. Persistence
       Each episode is written once as a parquet table when it closes; the
       episode metadata table and the balanced label index are rewritten after
       every episode so an interrupted run still leaves a usable dataset.

    7. Persistent dataset memory
       Loads a small JSON summary of previous collection runs and updates it
       atomically when the current run closes. The memory contains statistics
       only; previous episodes are never loaded into RAM.
    """

    def __init__(
        self,
        config: DatasetBucketConfig | None = None,
        *,
        bucket_capacity: int | None = None,
        body_weight_n: float = DEFAULT_BODY_WEIGHT_N,
        contact_label_config: ContactLabelConfig | None = None,
        regime_config: OperatingRegimeConfig | None = None,
        signal_bounds_version: int = 0,
        dataset_summary_path: str | Path | None = None,
        store: EpisodeStore | None = None,
    ):
        """
        Parameters
        ----------
        config
            Every threshold the system uses (:class:`DatasetBucketConfig`).
            One source: nothing in this class hardcodes a limit that the config
            also defines.

        bucket_capacity
            Override for ``config.bucket_capacity``, which is the reference
            population for the underpopulated-bucket diagnostic. Buckets never
            evict, so this is not a cap on storage.

        body_weight_n
            Total weight of the robot including payload, used to bin the
            perturbation axis as a fraction of body weight. Per-episode values
            from ``EpisodeMetadata.body_weight_n`` take precedence when present.

        contact_label_config, signal_bounds_version
            Recorded in :meth:`aggregate_compat_key`, which decides whether this
            run may be summed into the cross-run ``dataset_summary.json``. A
            change to either changes what ``contact_state`` and the normalised
            signals MEAN, so aggregates spanning the change describe nothing.

        dataset_summary_path
            Optional path to the persistent JSON summary for all collection
            runs. If the file already exists it is read during construction.
            Call :meth:`update_dataset_summary` after saving the current
            dataset, normally from the simulator shutdown hook.

        store
            Parquet writer for this run. When ``None`` the system runs purely
            in memory — used by tests and by the ``__main__`` demo.
        """
        self.config = config if config is not None else dataset_bucket_config
        self.body_weight_n = float(body_weight_n)
        self.contact_label_config = (
            contact_label_config
            if contact_label_config is not None
            else ContactLabelConfig()
        )
        self.signal_bounds_version = int(signal_bounds_version)
        self.regime_config = (
            regime_config if regime_config is not None else OperatingRegimeConfig()
        )
        self.bucket_capacity = int(
            self.config.bucket_capacity
            if bucket_capacity is None
            else bucket_capacity
        )
        # Crash predicate for the per-frame ``post_failure`` flag; mirrors the
        # simulators' own ``_is_crashed`` test.
        self.crash_height_m               = 0.15
        self.crash_tilt_deg               = 60.0
        # Sustained-collapse predicate: below this height for this many control
        # steps means the robot is resting on something other than its feet.
        self.collapse_height_m            = 0.175
        self.collapse_dwell_steps         = 25      # 0.5 s at 50 Hz
        self.min_perturbation_ratio       = self.config.min_perturbation_ratio
        self.store                        = store

        # ── Main storage ─────────────────────────────────────────────────────
        self.buckets: Dict[BucketKey, List[SampleRef]] = defaultdict(list)

        # Total windows ever offered to each bucket (including rejected ones).
        # Required for correct reservoir sampling probability.
        self.bucket_seen_count: Dict[BucketKey, int] = defaultdict(int)

        # ── Global counters ──────────────────────────────────────────────────
        self.total_samples_stored      = 0
        self.total_perturbation_stored = 0
        self.total_rare_stored         = 0
        self.total_samples_seen        = 0

        # ── Episode registry ─────────────────────────────────────────────────
        # Insertion-ordered: episode_id → its metadata (terrain, gait, friction,
        # payload, outcome, split, …).
        self.episodes: Dict[str, EpisodeMetadata] = {}

        # ── Persistent whole-dataset summary ─────────────────────────────────
        self.dataset_summary_path = (
            Path(dataset_summary_path) if dataset_summary_path is not None else None
        )
        self.dataset_memory = self.load_dataset_summary()

    @property
    def episodes_collected(self) -> List[str]:
        """Episode ids in collection order."""
        return list(self.episodes)

    # ══════════════════════════════════════════════════════════════════════════
    # LABEL DERIVATION  (private helpers)
    # ══════════════════════════════════════════════════════════════════════════

    def _perturbation_level(
        self, external_force_norm: float, body_weight_n: float
    ) -> str:
        """Bin ``|F_ext|`` as a fraction of this episode's body weight."""
        return perturbation_level(
            external_force_norm, body_weight_n, self.config
        )

    @staticmethod
    def _episode_grf(record: EpisodeRecord) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-foot force magnitude ``(T, 4)`` and friction utilisation ``(T,)``.

        Reads ``grf_base`` — substep-averaged, body frame. The instantaneous
        ``grf_world`` is aliased: on the audited run 98 index rows carried
        ``grf_total_n == 0`` while ``contact != 0000``, a loaded diagonal stance
        with zero total force, because the control step landed inside a
        micro-bounce. It is only used as a last resort, with a warning, so a
        pre-v4.2 episode still loads but its force statistics are not silently
        mixed with new ones.

        The friction ratio is ``max_i |f_xy,i| / f_z,i`` over feet carrying more
        than 1 N of normal force: how much of the available friction cone the
        contact is using, which is the slip-relevant axis and is measured
        nowhere else in the system. It is 0.0 when no foot is loaded, and it is
        only computable in the body frame, so a fallback episode reports zeros.
        """
        arrays = record.arrays
        if "grf_base" in arrays and np.any(arrays["grf_base"]):
            grf = np.asarray(arrays["grf_base"], dtype=np.float64).reshape(-1, 4, 3)
            magnitude = np.linalg.norm(grf, axis=2)
            normal = grf[:, :, 2]
            tangential = np.linalg.norm(grf[:, :, :2], axis=2)
            loaded = normal > 1.0
            ratio = np.where(loaded, tangential / np.where(loaded, normal, 1.0), 0.0)
            return magnitude, ratio.max(axis=1)

        if "grf_mean_n" in arrays and np.any(arrays["grf_mean_n"]):
            magnitude = np.asarray(
                arrays["grf_mean_n"], dtype=np.float64
            ).reshape(-1, 4)
            return magnitude, np.zeros(magnitude.shape[0], dtype=np.float64)

        warnings.warn(
            f"Episode '{record.metadata.episode_id}' has no substep-averaged GRF "
            f"(grf_base / grf_mean_n); falling back to the aliased instantaneous "
            f"grf_world. Force statistics for this episode are not comparable "
            f"with v4.2 runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        grf = np.asarray(arrays["grf_world"], dtype=np.float64).reshape(-1, 4, 3)
        magnitude = np.linalg.norm(grf, axis=2)
        return magnitude, np.zeros(magnitude.shape[0], dtype=np.float64)

    # ══════════════════════════════════════════════════════════════════════════
    # LABEL ENUMERATION  (public — call once per collected episode)
    # ══════════════════════════════════════════════════════════════════════════

    def add_episode(
        self,
        record: EpisodeRecord,
        stride: int = 1,
    ) -> Dict[str, int]:
        """
        Persist one episode and route each of its labelled timesteps into a bucket.

        The episode's per-timestep table is written to parquet first (so the rows
        a sample references are already on disk), then one :class:`SampleRef` per
        timestep is offered to the matching bucket.

        No window length is involved. Every timestep is indexable, including the
        first: a sampler with window length ``W`` skips rows with ``t < W - 1``
        itself, which lets one collected run serve any ``W``.

        Parameters
        ----------
        record
            The finished episode: per-timestep columns plus
            :class:`~mpx.utils.dataset_collection.dataset_schema.EpisodeMetadata`.
            ``contact``, ``grf_base``, ``external_force`` and ``cmd_base_vel`` are
            read from the table. ``contact`` is the debounced label produced by
            :class:`~mpx.utils.dataset_collection.contact_labeling.ContactDebouncer`
            (substep majority -> Schmitt trigger -> minimum dwell). The bucket
            system never re-derives contact; there is exactly one definition of
            the label and it lives in ``contact_labeling.py``.

        stride
            Step between consecutive indexed timesteps.
            1 → index every step (recommended; subsample at training time)
            n → index every n-th step, for a smaller index

        Returns
        -------
        dict with counts: "added", "rejected", "rare", "total_seen"
        """
        metadata = record.metadata
        episode_id = metadata.episode_id
        n_steps = record.n_steps

        if n_steps <= 0:
            raise ValueError(f"Episode '{episode_id}' has no timesteps")
        if episode_id in self.episodes:
            raise ValueError(f"Episode '{episode_id}' was already added")

        gait_type = GaitType(metadata.gait)
        terrain_type = TerrainType(metadata.terrain)
        operating_regime = self.episode_regime(record)
        post_failure = operating_regime == "failed"
        # `valid` is now "the locomotion was still recognisable", not merely
        # "the robot had not yet hit the ground": severe frames are kept and
        # indexed, but they are not training data by default.
        usable = np.isin(operating_regime, ("nominal", "degraded"))
        heights = np.asarray(
            record.arrays["base_height_terrain"], dtype=np.float64
        ).reshape(-1)
        tilts = (
            np.asarray(record.arrays["base_tilt_deg"], dtype=np.float64).reshape(-1)
            if np.any(record.arrays.get("base_tilt_deg", 0))
            else base_tilt_deg(record.arrays["base_quat"])
        )
        body_forces = np.asarray(
            record.arrays.get("non_foot_contact_n", np.zeros(n_steps)),
            dtype=np.float64,
        ).reshape(-1)

        contact = np.asarray(record.arrays["contact"], dtype=np.uint8).reshape(-1, 4)
        # Where the foot is NOT where the controller planned it to be. Split into
        # jitter and real disagreement: the raw ``.any()`` flag is 88% touchdown
        # timing jitter, so on its own it does not isolate the subset it claims
        # to. ``sustained`` is the benchmark subset.
        schedule = np.asarray(
            record.arrays["contact_schedule"], dtype=np.uint8
        ).reshape(-1, 4)
        mismatch = schedule_mismatch_masks(
            contact,
            schedule,
            edge_radius=self.config.mismatch_edge_radius,
            sustained_min_run=self.config.mismatch_sustained_min_run,
        )
        schedule_mismatch = mismatch["per_foot"].any(axis=1)
        schedule_mismatch_sustained = mismatch["sustained"].any(axis=1)
        schedule_mismatch_feet = mismatch["per_foot"]

        grf_feet, grf_tangential = self._episode_grf(record)
        ext = np.asarray(record.arrays["external_force"], dtype=np.float64).reshape(-1, 3)
        ext_norms = np.linalg.norm(ext, axis=1)
        commands = np.asarray(
            record.arrays["cmd_base_vel"], dtype=np.float64
        ).reshape(-1, 3)
        body_weight = float(metadata.body_weight_n) or self.body_weight_n

        # Persist the rows before any window can point at them.
        self.episodes[episode_id] = metadata
        if self.store is not None:
            self.store.write_episode(record)

        n_added = 0
        n_rejected = 0
        n_rare = 0

        for label_index in range(0, n_steps, stride):
            self.total_samples_seen += 1

            bits = tuple(int(b) for b in contact[label_index])
            state = contact_state_name(bits)
            if is_rare_contact(bits):
                n_rare += 1

            level = self._perturbation_level(ext_norms[label_index], body_weight)
            command_regime = speed_bin(commands[label_index])

            sample = SampleRef(
                episode_id          = episode_id,
                t                   = label_index,
                post_failure        = bool(post_failure[label_index]),
                valid               = bool(usable[label_index]),
                operating_regime    = str(operating_regime[label_index]),
                base_height_terrain = float(heights[label_index]),
                base_tilt_deg       = float(tilts[label_index]),
                non_foot_contact_n  = float(body_forces[label_index]),
                schedule_mismatch   = bool(schedule_mismatch[label_index]),
                schedule_mismatch_sustained = bool(
                    schedule_mismatch_sustained[label_index]
                ),
                schedule_mismatch_feet = tuple(
                    int(b) for b in schedule_mismatch_feet[label_index]
                ),
                contact_state       = state,
                contact_bits        = bits,
                grf_per_foot_n      = tuple(
                    float(v) for v in grf_feet[label_index]
                ),
                grf_tangential_ratio_max = float(grf_tangential[label_index]),
                external_force_n    = float(ext_norms[label_index]),
                perturbation_level  = level,
                speed_bin           = command_regime,
                gait_type           = gait_type,
                terrain_type        = terrain_type,
            )

            key = BucketKey(
                contact_state       = state,
                perturbation_level  = level,
                speed_bin           = command_regime,
                terrain             = terrain_type,
                gait_type           = gait_type,
            )

            if self._bucket_add(key, sample):
                n_added += 1
            else:
                n_rejected += 1

        return {
            "added":      n_added,
            "rejected":   n_rejected,
            "rare":       n_rare,
            "total_seen": n_added + n_rejected,
        }

    def episode_regime(self, record: EpisodeRecord) -> np.ndarray:
        """
        Per-frame ``operating_regime`` for an episode, computing it if absent.

        The recorder writes the column, so this normally just reads it. An
        episode assembled by hand (or collected before the annotation existed)
        gets it derived here from the same four signals, so callers never have to
        branch on whether the column is populated.
        """
        arrays = record.arrays
        regime = arrays.get("operating_regime")
        if regime is not None and np.any(np.asarray(regime).astype(str) != ""):
            return np.asarray(regime).astype(str)

        height = np.asarray(
            arrays["base_height_terrain"], dtype=np.float64
        ).reshape(-1)
        tilt = (
            np.asarray(arrays["base_tilt_deg"], dtype=np.float64).reshape(-1)
            if "base_tilt_deg" in arrays and np.any(arrays["base_tilt_deg"])
            else base_tilt_deg(arrays["base_quat"])
        )
        body = np.asarray(
            arrays.get("non_foot_contact_n", np.zeros(len(height))),
            dtype=np.float64,
        ).reshape(-1)
        tracking = arrays.get("cmd_tracking_error")
        return classify_regime(
            height, tilt, body,
            None if tracking is None else np.asarray(tracking).reshape(-1),
            config=self.regime_config,
        ).astype(str)

    def post_failure_mask(self, record: EpisodeRecord) -> np.ndarray:
        """
        Per-frame flag: the robot had already become unrecoverable at this step.

        Now simply ``operating_regime == "failed"`` — the same predicate expressed
        once, with three graded levels below it instead of a cliff. The old
        version fired on 52 of 32,306 rows (0.16%) while crash episodes
        contributed 3,194 rows (9.9%), so 98.4% of the frames in a crashing
        episode were labelled as normal operation. See
        :mod:`mpx.utils.dataset_collection.operating_regime`.
        """
        return self.episode_regime(record) == "failed"

    def _legacy_post_failure_mask(self, record: EpisodeRecord) -> np.ndarray:
        """
        The v4 crash predicate, kept so the regime thresholds can be compared to it.

        Only failure episodes can contain such frames; the predicate is the
        simulators' own crash test (base too low, or rolled/pitched past 60°), and
        once it first holds every later frame of the episode is flagged.
        """
        n_steps = record.n_steps
        mask = np.zeros(n_steps, dtype=bool)
        if record.metadata.terminate_by != "failure":
            return mask

        height = np.asarray(
            record.arrays["base_height_terrain"], dtype=np.float64
        ).reshape(-1)
        quat = np.asarray(record.arrays["base_quat"], dtype=np.float64).reshape(-1, 4)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))

        # Instantaneous predicates: too low, or rolled/pitched past the limit.
        crashed = (
            (height < self.crash_height_m)
            | (np.abs(roll) > np.deg2rad(self.crash_tilt_deg))
            | (np.abs(pitch) > np.deg2rad(self.crash_tilt_deg))
        )

        # Sustained collapse: a folded robot rests just above the hard floor,
        # level, and stays there. One pilot logged 26 s of exactly that as a
        # clean episode. Flag a low stance only once it has persisted, so a deep
        # squat is not mistaken for a fall.
        low = height < self.collapse_height_m
        if low.any():
            run_length = 0
            for step, is_low in enumerate(low):
                run_length = run_length + 1 if is_low else 0
                if run_length >= self.collapse_dwell_steps:
                    crashed[step] = True

        first = np.flatnonzero(crashed)
        if first.size:
            mask[first[0] :] = True
        return mask

    # ══════════════════════════════════════════════════════════════════════════
    # SAMPLE FILING  (private)
    # ══════════════════════════════════════════════════════════════════════════

    def _bucket_add(self, key: BucketKey, sample: SampleRef) -> bool:
        """
        File a sample under its bucket. **Never drops.**

        The index must contain every row in the episode files: it is a lookup
        table, and dropping rows here silently reweights the classes for anyone
        who counts over it. Class balance is applied at training time as a
        per-sample weight (see :meth:`class_weights`), which is reversible and
        visible, and is applied to the train split only.

        Returns True always; the signature is kept so callers still read as
        "was this stored".
        """
        self.bucket_seen_count[key] += 1
        self.buckets[key].append(sample)
        self._update_global_counters(sample, delta=+1)
        return True

    def _update_global_counters(self, sample: SampleRef, delta: int) -> None:
        """Increment or decrement global counters when a sample is stored/evicted."""
        self.total_samples_stored      += delta
        self.total_perturbation_stored += delta if sample.perturbation_active else 0
        self.total_rare_stored         += delta if sample.rare_contact else 0

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def perturbation_ratio(self) -> float:
        """Fraction of currently stored samples that are perturbation-active."""
        if self.total_samples_stored == 0:
            return 0.0
        return self.total_perturbation_stored / self.total_samples_stored

    @property
    def perturbation_ratio_satisfied(self) -> bool:
        """True if the minimum perturbation ratio constraint is currently met."""
        return self.perturbation_ratio >= self.min_perturbation_ratio

    def perturbation_ratio_by_bucket(self) -> Dict[str, float]:
        """
        Perturbed fraction per bucket, over buckets big enough to mean anything.

        The global ratio is the wrong number to act on: the stated rationale for
        the perturbation axis is that every contact state is seen both pushed and
        unpushed, and an aggregate hides a state that is never pushed. On the
        audited run a global 20.9% covered a per-state range of 9.4% to 66.7%.

        Keyed on the bucket label WITHOUT its perturbation field, since a bucket
        already carries one perturbation level; the ratio is over the group of
        buckets that differ only in that field.
        """
        totals: Dict[str, int] = defaultdict(int)
        perturbed: Dict[str, int] = defaultdict(int)
        for key, samples in self.buckets.items():
            group = (
                f"{key.contact_state} | {key.speed_bin} | "
                f"{key.terrain.value} | {key.gait_type.value}"
            )
            totals[group] += len(samples)
            if key.perturbation_active:
                perturbed[group] += len(samples)
        return {
            group: perturbed[group] / count
            for group, count in sorted(totals.items())
            if count >= self.config.min_bucket_samples_for_ratio_check
        }

    def underpopulated_buckets(
        self, share_of_expected: float | None = None
    ) -> List[Tuple[BucketKey, int]]:
        """
        Buckets holding far less than an even share of the run's samples.

        Relative, not a fraction of ``bucket_capacity``: an absolute denominator
        flags every bucket on a 5-episode pilot and none on a 500-episode run,
        which is the opposite of what the diagnostic is for. ``expected`` is what
        each active bucket would hold under a uniform split.
        """
        threshold = (
            self.config.underpopulated_share_of_expected
            if share_of_expected is None
            else share_of_expected
        )
        if not self.buckets:
            return []
        expected = self.total_samples_stored / len(self.buckets)
        return [
            (key, len(samples))
            for key, samples in self.buckets.items()
            if len(samples) < threshold * expected
        ]

    def undercollected_buckets(
        self, min_bucket_samples: int | None = None
    ) -> List[Tuple[str, int]]:
        """
        Buckets too sparse to weight. Report these; do not train on them.

        Distinct from :meth:`underpopulated_buckets`, which is relative: this one
        is the absolute floor below which a bucket cannot teach anything, and it
        is the same floor :meth:`bucket_weights` zeroes out.
        """
        floor = (
            self.config.min_bucket_samples_for_weighting
            if min_bucket_samples is None
            else min_bucket_samples
        )
        counts: Dict[str, int] = defaultdict(int)
        for key, samples in self.buckets.items():
            counts[key.label()] += len(samples)
        return sorted(
            ((label, count) for label, count in counts.items() if count < floor),
            key=lambda item: item[1],
        )

    def regime_counts(self) -> Dict[str, int]:
        """Stored samples per operating regime, in severity order."""
        counts: Dict[str, int] = {name: 0 for name in REGIME_ORDER}
        for samples in self.buckets.values():
            for sample in samples:
                counts[sample.operating_regime] = (
                    counts.get(sample.operating_regime, 0) + 1
                )
        return counts

    def nominal_fraction_by_bucket(self) -> Dict[str, float]:
        """
        Fraction of each bucket collected under nominal locomotion.

        The regime deliberately stays out of :class:`BucketKey` — four levels
        would multiply the key and most combinations are empty — so it is
        tracked here instead, beside the diversity statistics. A bucket that is
        mostly non-nominal is not the contact state it claims to be; it is a
        falling robot that happened to have those feet down.
        """
        totals: Dict[str, int] = defaultdict(int)
        nominal: Dict[str, int] = defaultdict(int)
        for key, samples in self.buckets.items():
            label = key.label()
            totals[label] += len(samples)
            nominal[label] += sum(
                1 for s in samples if s.operating_regime == "nominal"
            )
        return {
            label: nominal[label] / count
            for label, count in sorted(totals.items())
            if count
        }

    def contact_state_counts(self) -> Dict[str, int]:
        """Total stored samples per contact state, summed across all buckets."""
        counts: Dict[str, int] = defaultdict(int)
        for key, samples in self.buckets.items():
            counts[key.contact_state] += len(samples)
        return dict(counts)

    def split_counts(self) -> Dict[str, int]:
        """Stored samples per train/val/test split, resolved via the episode."""
        counts: Dict[str, int] = defaultdict(int)
        for samples in self.buckets.values():
            for sample in samples:
                metadata = self.episodes.get(sample.episode_id)
                counts[metadata.split_assigned if metadata else "train"] += 1
        return dict(counts)

    def episode_outcome_counts(self) -> Dict[str, int]:
        """Episodes per ``terminate_by`` outcome."""
        counts: Dict[str, int] = defaultdict(int)
        for metadata in self.episodes.values():
            counts[metadata.terminate_by] += 1
        return dict(counts)

    def print_bucket_snapshot(
        self,
        *,
        event: str,
        detail: str = "",
    ) -> None:
        """Compact bucket summary for logging after store/remove events."""
        header = f"[collect] bucket summary — {event}"
        if detail:
            header = f"{header} ({detail})"
        print(header, flush=True)

        ratio_flag = "ok" if self.perturbation_ratio_satisfied else "LOW"
        print(
            f"  stored={self.total_samples_stored:,}  "
            f"seen={self.total_samples_seen:,}  "
            f"episodes={len(self.episodes)}  "
            f"active_buckets={len(self.buckets)}  "
            f"perturb={self.total_perturbation_stored:,} "
            f"({self.perturbation_ratio:.1%}, {ratio_flag})  "
            f"rare={self.total_rare_stored:,}",
            flush=True,
        )

        splits = self.split_counts()
        if splits:
            print(
                "  splits: "
                + "  ".join(f"{name}={count:,}" for name, count in sorted(splits.items())),
                flush=True,
            )
        outcomes = self.episode_outcome_counts()
        if outcomes:
            print(
                "  episode outcomes: "
                + "  ".join(f"{name}={count}" for name, count in sorted(outcomes.items())),
                flush=True,
            )

        counts = self.contact_state_counts()
        if not counts:
            print("  contact states: (empty)", flush=True)
            return

        print("  contact states:", flush=True)
        total = max(self.total_samples_stored, 1)
        for state, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            bits = CONTACT_STATE_BITS.get(state, "mixed")
            pct = 100.0 * count / total
            print(
                f"    {state:<14} {bits:<5}  {count:>6,}  ({pct:5.1f}%)",
                flush=True,
            )

        # Top populated bucket keys (contact × perturb × terrain × gait).
        top = sorted(
            ((key, len(samples)) for key, samples in self.buckets.items()),
            key=lambda x: (-x[1], x[0].contact_state),
        )[:5]
        if top:
            print("  top buckets:", flush=True)
            for key, n in top:
                print(f"    {key.label():<56}  n={n:,}", flush=True)

    def print_summary(self) -> None:
        """Print a human-readable collection summary."""
        sep  = "═" * 64
        thin = "─" * 64

        print(sep)
        print("  DatasetBucketSystem — Collection Summary")
        print(sep)
        print(f"  Samples stored         : {self.total_samples_stored:>8,}")
        print(f"  Samples seen (total)   : {self.total_samples_seen:>8,}")
        print(f"  Perturbation samples   : {self.total_perturbation_stored:>8,}")
        ratio_flag = "✓" if self.perturbation_ratio_satisfied else "✗ UNSATISFIED"
        print(f"  Perturbation ratio     : {self.perturbation_ratio:>8.2%}  "
              f"(min: {self.min_perturbation_ratio:.2%}  {ratio_flag})")
        print(f"  Rare-contact samples   : {self.total_rare_stored:>8,}")
        print(f"  Active buckets         : {len(self.buckets):>8,}")
        print(f"  Episodes collected     : {len(self.episodes):>8,}")

        # ── Episode-level breakdown ──────────────────────────────────────────
        outcomes = self.episode_outcome_counts()
        if outcomes:
            print(thin)
            print("  Episodes per outcome (terminate_by):")
            for name, count in sorted(outcomes.items()):
                print(f"    {name:<12} {count:>6,}")

        splits = self.split_counts()
        if splits:
            print("  Samples per split:")
            total = max(self.total_samples_stored, 1)
            for name, count in sorted(splits.items()):
                print(f"    {name:<12} {count:>7,}  ({100 * count / total:5.1f}%)")

        # ── Contact state breakdown ──────────────────────────────────────────
        print(thin)
        print("  Samples per contact state:")
        total = max(self.total_samples_stored, 1)
        for state, count in sorted(self.contact_state_counts().items()):
            bar = "█" * int(28 * count / total)
            pct = 100 * count / total
            bits = CONTACT_STATE_BITS.get(state, "mixed")
            print(f"    {state:<14} {bits:<5}  {count:>7,}  ({pct:5.1f}%)  {bar}")

        # ── Per-bucket perturbation coverage ─────────────────────────────────
        # The aggregate above says whether enough samples are perturbed; this
        # says whether the perturbed ones are spread over the contact states,
        # which is the actual reason the axis exists.
        ratios = self.perturbation_ratio_by_bucket()
        if ratios:
            print(thin)
            worst = sorted(ratios.items(), key=lambda item: item[1])[:5]
            print(
                f"  Perturbation coverage per bucket group "
                f"(n >= {self.config.min_bucket_samples_for_ratio_check}), worst 5:"
            )
            for group, ratio in worst:
                flag = "⚠" if ratio < self.min_perturbation_ratio else " "
                print(f"   {flag} {group:<52}  {ratio:6.1%}")

        # ── GRF diversity ────────────────────────────────────────────────────
        # Deliberately NOT computed here. The statistics live in
        # grf_diversity.py and read index.parquet after the fact, so they can be
        # changed without recollecting. std of total GRF — what this block used
        # to print — is also the wrong statistic: statics pins the total near
        # body weight whatever the distribution across feet, so it fires hardest
        # on FULL support, where low variance is physics rather than a defect.
        print(thin)
        print(
            "  GRF diversity: run "
            "`python -m mpx.utils.dataset_collection.grf_diversity <run_dir>` "
            "after the run"
        )

        # ── Bucket population warnings ───────────────────────────────────────
        sparse = self.undercollected_buckets()
        if sparse:
            print(thin)
            print(
                f"  ⚠  {len(sparse)} buckets below the "
                f"{self.config.min_bucket_samples_for_weighting}-sample floor — "
                f"NOT COLLECTED, weight 0, rows kept:"
            )
            for label, count in sparse[:10]:
                print(f"     {label:<56}  n={count}")
            if len(sparse) > 10:
                print(f"     ... and {len(sparse) - 10} more")

        under = self.underpopulated_buckets()
        if under:
            expected = self.total_samples_stored / max(len(self.buckets), 1)
            print(thin)
            print(
                f"  ⚠  {len(under)} buckets below "
                f"{self.config.underpopulated_share_of_expected:.0%} of an even "
                f"share ({expected:,.0f} samples):"
            )
            for key, count in sorted(under, key=lambda x: x[1])[:10]:
                print(f"     {key.label():<56}  n={count}")
            if len(under) > 10:
                print(f"     ... and {len(under) - 10} more")

        print(sep)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT — the balanced label index
    # ══════════════════════════════════════════════════════════════════════════

    def index_rows(
        self,
        max_per_bucket: Optional[int] = None,
        shuffle:        bool          = True,
        seed:           Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the balanced label index: one row per exported sample.

        Each row names the ``(episode_id, t)`` a sampler reads plus the bucket it
        was balanced into. No window length appears: the sampler chooses ``W`` and
        skips rows with ``t < W - 1``.

        There is deliberately **no** ``split`` column. Train/val/test comes from
        ``datasets/manifest.json`` at load time, joined on
        ``randomization_group_id`` — episodes sharing randomization are
        near-duplicates and must land in the same split.

        Parameters
        ----------
        max_per_bucket
            Hard cap on samples drawn from each bucket. Use this to enforce
            strict per-bucket balance at training time without dropping the
            stored references.

        shuffle
            Randomly permute the rows before returning.

        seed
            Random seed for reproducible shuffling.
        """
        rows: List[Dict[str, Any]] = []
        for key, samples in self.buckets.items():
            pool = samples if max_per_bucket is None else samples[:max_per_bucket]
            bucket_label = key.label()
            for sample in pool:
                metadata = self.episodes.get(sample.episode_id)
                rows.append(
                    {
                        "episode_id":          sample.episode_id,
                        "t":                   int(sample.t),
                        "run_id":              metadata.run_id if metadata else "",
                        "randomization_group_id": (
                            metadata.randomization_group_id if metadata else ""
                        ),
                        "seed": (
                            int(metadata.seed)
                            if metadata is not None and metadata.seed is not None
                            else -1
                        ),
                        "terminate_reason": (
                            metadata.terminate_reason if metadata else ""
                        ),
                        "post_failure":        bool(sample.post_failure),
                        "valid":               bool(sample.valid),
                        "operating_regime":    sample.operating_regime,
                        "base_height_terrain": float(sample.base_height_terrain),
                        "base_tilt_deg":       float(sample.base_tilt_deg),
                        "non_foot_contact_n":  float(sample.non_foot_contact_n),
                        "schedule_mismatch":   bool(sample.schedule_mismatch),
                        "schedule_mismatch_sustained": bool(
                            sample.schedule_mismatch_sustained
                        ),
                        "schedule_mismatch_bits": contact_bits_string(
                            sample.schedule_mismatch_feet
                        ),
                        # With ``t``, this decides window validity for ANY W.
                        # window_valid_w10 hardcodes one W and contradicts the
                        # window-length agnosticism the whole design rests on;
                        # it survives as a convenience column only.
                        "episode_n_steps":     int(metadata.n_steps) if metadata else 0,
                        "window_valid_w10":    int(sample.t) >= 9,
                        "terrain":             sample.terrain_type.value,
                        "gait":                sample.gait_type.value,
                        "contact_state":       sample.contact_state,
                        "contact_bits":        contact_bits_string(sample.contact_bits),
                        "rare_contact":        bool(sample.rare_contact),
                        "perturbation_active": bool(sample.perturbation_active),
                        "perturbation_level":  sample.perturbation_level,
                        "speed_bin":           sample.speed_bin,
                        "bucket_key":          bucket_label,
                        "grf_total_n":         float(sample.grf_total_n),
                        "grf_load_share_max":  float(sample.grf_load_share_max),
                        "grf_tangential_ratio_max": float(
                            sample.grf_tangential_ratio_max
                        ),
                        "external_force_n":    float(sample.external_force_n),
                    }
                )

        if shuffle:
            random.Random(seed).shuffle(rows)
        return rows

    def _split_of(self, episode_id: str) -> str:
        """Cached collection-time split of an episode (manifest is the authority)."""
        metadata = self.episodes.get(episode_id)
        return metadata.split_assigned if metadata else "train"

    def bucket_weights(
        self,
        alpha: float | None = None,
        splits: Sequence[str] | None = None,
        min_bucket_samples: int | None = None,
        max_weight_ratio: float | None = None,
    ) -> Dict[str, float]:
        """
        Per-bucket sampling weights, ``(1 / count) ** alpha``, keyed on the full
        bucket label.

        The replacement for dropping rows. ``alpha = 0`` is the natural
        distribution, ``alpha = 1`` fully equalises, 0.5 lifts the rare states
        without pretending they are as common as a trot diagonal.

        Three guards the v4.1 version lacked, each fixing a way the old version
        contradicted its own docstring:

        * **Counts come from ``splits``** (train by default). Counting over every
          split let held-out class statistics leak into the training weights.
        * **``min_bucket_samples``** — buckets below the floor get weight 0 and
          are reported as *not collected* rather than rarely collected. One
          sample cannot teach a class, it can only add gradient variance; the
          audited run had a bucket with n = 1 carrying 57x the modal weight.
        * **``max_weight_ratio``** — clipped after weighting, then renormalised
          so the mean weight over weighted samples is 1.0.

        Keyed on the whole bucket, not on ``contact_state`` alone: the key has
        five fields and the perturbation and speed axes exist precisely so their
        coverage can be balanced.
        """
        cfg = self.config
        alpha = cfg.class_balance_alpha if alpha is None else float(alpha)
        splits = tuple(cfg.weighted_splits if splits is None else splits)
        floor = (
            cfg.min_bucket_samples_for_weighting
            if min_bucket_samples is None
            else int(min_bucket_samples)
        )
        ratio_cap = (
            cfg.max_weight_ratio
            if max_weight_ratio is None
            else float(max_weight_ratio)
        )

        in_split = {
            episode_id
            for episode_id, metadata in self.episodes.items()
            if getattr(metadata, "split_assigned", "train") in splits
        }

        counts: Dict[str, int] = defaultdict(int)
        for key, samples in self.buckets.items():
            counts[key.label()] += sum(
                1 for s in samples if s.episode_id in in_split and s.valid
            )

        eligible = {label: n for label, n in counts.items() if n >= floor}
        if not eligible:
            return {label: 0.0 for label in counts}

        weights = {label: (1.0 / n) ** alpha for label, n in eligible.items()}
        lowest = min(weights.values())
        weights = {
            label: min(w, lowest * ratio_cap) for label, w in weights.items()
        }

        total = sum(weights[label] * eligible[label] for label in eligible)
        scale = sum(eligible.values()) / total if total else 1.0
        out = {label: w * scale for label, w in weights.items()}
        out.update({label: 0.0 for label in counts if label not in eligible})
        return out

    def class_weights(self, alpha: float = 0.5) -> Dict[str, float]:
        """
        Deprecated: per-contact-state weights, superseded by :meth:`bucket_weights`.

        Kept for one release for anything outside this package that still calls
        it. It has none of the guards — no split restriction, no population
        floor, no ratio cap — so do not use it for new work.
        """
        warnings.warn(
            "class_weights is deprecated; use bucket_weights, which keys on the "
            "whole bucket and applies the split restriction, the population "
            "floor and the weight-ratio cap.",
            DeprecationWarning,
            stacklevel=2,
        )
        counts = self.contact_state_counts()
        if not counts:
            return {}
        weights = {
            state: (1.0 / max(count, 1)) ** float(alpha)
            for state, count in counts.items()
        }
        total = sum(weights[state] * count for state, count in counts.items())
        scale = sum(counts.values()) / total if total else 1.0
        return {state: w * scale for state, w in weights.items()}

    def balanced_index_rows(
        self,
        alpha: float | None = None,
        max_per_bucket: Optional[int] = None,
        splits: Sequence[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        The index with ``split`` and ``weight`` columns — the balanced view.

        Shipped as ``index_balanced.parquet`` beside the complete index, so the
        balancing is inspectable rather than baked into which rows exist. Rows
        whose weight is 0.0 (a bucket below the population floor) stay in
        ``index.parquet`` — nothing is deleted — they are simply not drawn by a
        weighted sampler.
        """
        weighted_splits = tuple(
            self.config.weighted_splits if splits is None else splits
        )
        weights = self.bucket_weights(alpha=alpha, splits=weighted_splits)
        rows = []
        for row in self.index_rows(max_per_bucket=max_per_bucket, shuffle=False):
            split = self._split_of(row["episode_id"])
            rows.append(
                {
                    "episode_id": row["episode_id"],
                    "t": row["t"],
                    "split": split,
                    # Outside the weighted splits the weight is exactly 1.0: a
                    # reweighted val or test metric describes a distribution
                    # that does not exist.
                    "weight": (
                        float(weights.get(row["bucket_key"], 0.0))
                        if split in weighted_splits
                        else 1.0
                    ),
                }
            )
        return rows

    def split_index_rows(
        self,
        max_per_bucket: Optional[int] = None,
        seed:           Optional[int] = 42,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group the label index by the episode-level train/val/test split.

        The split itself is assigned per episode at collection time (see
        :func:`~mpx.utils.dataset_collection.dataset_schema.assign_split`), so
        every sample of one episode lands in the same bucket of this dict and no
        window cut around them can leak across the boundary.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.index_rows(max_per_bucket=max_per_bucket, shuffle=True, seed=seed):
            metadata = self.episodes.get(row["episode_id"])
            grouped[metadata.split_assigned if metadata else "train"].append(row)
        return dict(grouped)

    def save_dataset(
        self,
        run_dir: str | Path | None = None,
        *,
        metadata: dict | None = None,
        write_metadata: bool = True,
        max_per_bucket: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        balance_alpha: float | None = None,
    ) -> Path:
        """
        Write the episode metadata table and the label index.

        Episode tables themselves were already written by :meth:`add_episode`.
        Returns the path of ``index.parquet``.
        """
        store = self.store
        if store is None:
            if run_dir is None:
                raise ValueError("save_dataset needs a store or a run_dir")
            store = EpisodeStore(Path(run_dir))

        balance_alpha = (
            self.config.class_balance_alpha
            if balance_alpha is None
            else float(balance_alpha)
        )
        store.write_episode_table(list(self.episodes.values()))
        # index.parquet is COMPLETE: one row per sample, nothing dropped.
        index_path = store.write_index(
            self.index_rows(
                max_per_bucket=max_per_bucket, shuffle=shuffle, seed=seed
            )
        )
        # The balanced view lives beside it as weights, never as deletions.
        store.write_balanced_index(self.balanced_index_rows(alpha=balance_alpha))

        if write_metadata and metadata is not None:
            payload = {
                **metadata,
                "run_dir": str(store.run_dir.resolve()),
                "index_path": str(index_path.resolve()),
                "episode_schema_version": EPISODE_SCHEMA_VERSION,
                "dataset_summary_schema_version": DATASET_SUMMARY_SCHEMA_VERSION,
                "bucket_capacity": self.bucket_capacity,
                "aggregate_compat_key": self.aggregate_compat_key(),
                "randomization_groups": sorted(
                    {m.randomization_group_id for m in self.episodes.values()}
                ),
                "bucket_key_fields": list(BucketKey.__dataclass_fields__),
                "class_balance_alpha": float(balance_alpha),
                "max_weight_ratio": float(self.config.max_weight_ratio),
                "weighted_splits": list(self.config.weighted_splits),
                "min_bucket_samples_for_weighting": int(
                    self.config.min_bucket_samples_for_weighting
                ),
                "bucket_weights": self.bucket_weights(alpha=balance_alpha),
                "undercollected_buckets": dict(self.undercollected_buckets()),
                "effective_sample_size": self.effective_sample_size(),
                "foot_order": list(FOOT_ORDER),
                "saved_at": self._summary_timestamp(),
                "n_samples": self.total_samples_stored,
                "total_samples_stored": self.total_samples_stored,
                "total_samples_seen": self.total_samples_seen,
                "episodes_collected": self.episodes_collected,
            }
            (store.run_dir / "run_metadata.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )

        return index_path

    def effective_sample_size(self, window: int = 10) -> Dict[str, Any]:
        """
        How many *independent* windows the index really contains.

        At ``label_stride = 1`` consecutive index rows share ``W - 1`` of their
        ``W`` frames, so a variance estimate computed over stride-1 held-out rows
        is far too tight. Recording the number stops anyone reading 55,782 rows
        as 55,782 independent observations.
        """
        rows = self.total_samples_stored
        return {
            "index_rows": int(rows),
            "window": int(window),
            f"independent_windows_w{int(window)}": int(rows // max(window, 1)),
            "note": (
                f"At label_stride = 1 and W = {int(window)}, consecutive index "
                f"rows share {int(window) - 1} of {int(window)} frames. Variance "
                f"estimates computed over stride-1 val/test rows are far too "
                f"tight; use a stride of >= W on held-out splits."
            ),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PERSISTENT WHOLE-DATASET SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    def aggregate_compat_key(self) -> str:
        """
        Runs may only be summed into one ``dataset_summary.json`` when this matches.

        The cross-run summary accumulates with no compatibility check, and three
        separate changes silently invalidate it: the bucket key fields (which
        rename every bucket), what ``grf_total_n`` measures, and the contact
        labelling config (which changes what ``contact_state`` MEANS). Mixing
        those in one aggregate produces numbers that describe nothing.
        """
        payload = json.dumps(
            {
                "episode_schema_version": EPISODE_SCHEMA_VERSION,
                "summary_schema_version": DATASET_SUMMARY_SCHEMA_VERSION,
                "signal_bounds_version": self.signal_bounds_version,
                # Thresholds only. Body weight and substep count are per-run
                # facts, not definitions of the label, and folding them in would
                # split the aggregate on a payload draw.
                "contact_labeling": {
                    "on_threshold_n": self.contact_label_config.on_threshold_n,
                    "off_threshold_n": self.contact_label_config.off_threshold_n,
                    "min_dwell_steps": self.contact_label_config.min_dwell_steps,
                    "substep_majority": self.contact_label_config.substep_majority,
                },
                "bucket_key_fields": list(BucketKey.__dataclass_fields__),
                "grf_source": "grf_base_substep_mean",
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _summary_timestamp() -> str:
        """Return an ISO-8601 timestamp including the local UTC offset."""
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @classmethod
    def _empty_dataset_summary(cls) -> Dict[str, Any]:
        """Create an empty in-memory representation of the summary file."""
        return {
            "schema_version": DATASET_SUMMARY_SCHEMA_VERSION,
            "compat_key": "",
            "created_at": cls._summary_timestamp(),
            "updated_at": None,
            "summary": {
                "dataset_files": 0,
                "total_size_bytes": 0,
                "samples": 0,
                "perturbation_samples": 0,
                "perturbation_ratio": 0.0,
                "rare_contact_samples": 0,
                "collection_samples_stored": 0,
                "collection_samples_seen": 0,
                "episodes": 0,
                "active_buckets": 0,
                "contact_state_counts": {},
                "terrain_counts": {},
                "gait_counts": {},
                "split_counts": {},
                "terminate_by_counts": {},
                "bucket_counts": {},
                "grf_total_n_by_bucket": {},
                "perturbation_force_n": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "samples": 0,
                },
                "perturbation_force_n_by_bucket": {},
            },
            "datasets": {},
        }

    def load_dataset_summary(self) -> Dict[str, Any]:
        """
        Read the persistent whole-dataset summary.

        A missing file represents an empty dataset. A file written by an older
        schema is moved aside (kept, not deleted) and a fresh memory is started,
        since its per-run records describe a previous storage format.
        Unreadable files raise ``ValueError`` instead of being overwritten.
        """
        if self.dataset_summary_path is None:
            return self._empty_dataset_summary()
        if not self.dataset_summary_path.exists():
            return self._bootstrap_dataset_summary()

        try:
            payload = json.loads(
                self.dataset_summary_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read dataset summary '{self.dataset_summary_path}': {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"Dataset summary '{self.dataset_summary_path}' must contain a JSON object"
            )
        version = payload.get("schema_version")
        if version != DATASET_SUMMARY_SCHEMA_VERSION:
            archived = self.dataset_summary_path.with_suffix(f".v{version}.json")
            if not archived.exists():
                os.replace(self.dataset_summary_path, archived)
            print(
                f"[collect] dataset memory schema v{version} predates the parquet "
                f"format — archived as {archived.name}, starting a new summary",
                flush=True,
            )
            return self._bootstrap_dataset_summary()
        if not isinstance(payload.get("datasets"), dict):
            raise ValueError(
                f"Dataset summary '{self.dataset_summary_path}' has no valid 'datasets' map"
            )

        # Runs may only be summed when they mean the same thing. A changed
        # bucket key, GRF definition or contact-labelling threshold makes the
        # existing per-run records incomparable with this one, so the old file is
        # rotated aside (kept, never deleted) and a fresh aggregate is started.
        stored_key = str(payload.get("compat_key", ""))
        current_key = self.aggregate_compat_key()
        if stored_key and stored_key != current_key:
            archived = self.dataset_summary_path.with_name(
                f"{self.dataset_summary_path.stem}_{stored_key}.json"
            )
            if not archived.exists():
                os.replace(self.dataset_summary_path, archived)
            print(
                f"[collect] dataset memory was written under compat key "
                f"{stored_key}, this run is {current_key} — the bucket key, the "
                f"GRF definition or the contact-labelling thresholds changed, so "
                f"the two cannot be summed. Archived as {archived.name}, "
                f"starting a new aggregate.",
                flush=True,
            )
            return self._bootstrap_dataset_summary()
        return payload

    @staticmethod
    def _json_compatible(value: Any) -> Any:
        """Convert common NumPy/path/enum metadata values to JSON types."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {
                str(key): DatasetBucketSystem._json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [
                DatasetBucketSystem._json_compatible(item)
                for item in value
            ]
        return value

    def _bootstrap_dataset_summary(self) -> Dict[str, Any]:
        """
        Summarize existing parquet runs when no memory file exists yet.

        Keeps datasets collected before (or without) the memory file in the
        whole-dataset totals. Subsequent sessions load the small JSON file
        instead of reopening every index.
        """
        memory = self._empty_dataset_summary()
        if self.dataset_summary_path is None:
            return memory

        datasets = {}
        root = self.dataset_summary_path.parent
        if root.exists():
            for index_path in sorted(root.rglob(INDEX_FILENAME)):
                try:
                    record = self._saved_dataset_summary(index_path)
                except (ImportError, ValueError, OSError) as exc:
                    print(
                        f"[collect] skipping unreadable run '{index_path.parent.name}': {exc}",
                        flush=True,
                    )
                    continue
                datasets[record["dataset_path"]] = record

        memory["datasets"] = datasets
        memory["summary"] = self._aggregate_dataset_runs(datasets)
        return memory

    def _saved_dataset_summary(self, index_path: Path) -> Dict[str, Any]:
        """Build a run record by reading an already-saved run directory."""
        run_dir = index_path.parent
        store = EpisodeStore(run_dir)
        index = store.read_index()
        episodes = store.read_episode_table()

        run_metadata: Dict[str, Any] = {}
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists():
            try:
                run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Cannot read run metadata '{metadata_path}': {exc}"
                ) from exc

        total_size_bytes = sum(
            path.stat().st_size for path in run_dir.rglob("*.parquet")
        )
        completed_at = run_metadata.get("saved_at") or datetime.fromtimestamp(
            index_path.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds")

        return self._build_run_record(
            index_rows=index,
            episode_rows=episodes,
            run_dir=run_dir,
            index_path=index_path,
            completed_at=completed_at,
            total_size_bytes=total_size_bytes,
            metadata=run_metadata,
            collection={
                "samples_stored": int(
                    run_metadata.get("total_samples_stored", len(index))
                ),
                "samples_seen": int(run_metadata.get("total_samples_seen", len(index))),
                "episodes": len(episodes),
                "episode_ids": [row["episode_id"] for row in episodes],
                "bucket_capacity": run_metadata.get("bucket_capacity"),
            },
            max_per_bucket=run_metadata.get("max_per_bucket"),
        )

    def _build_run_record(
        self,
        *,
        index_rows: Sequence[Dict[str, Any]],
        episode_rows: Sequence[Dict[str, Any]],
        run_dir: Path,
        index_path: Path,
        completed_at: str,
        total_size_bytes: int,
        metadata: dict | None,
        collection: Dict[str, Any],
        max_per_bucket: Optional[int],
    ) -> Dict[str, Any]:
        """Aggregate one run's label index into the per-run summary record."""
        contact_counts: Dict[str, int] = defaultdict(int)
        terrain_counts: Dict[str, int] = defaultdict(int)
        gait_counts: Dict[str, int] = defaultdict(int)
        split_counts: Dict[str, int] = defaultdict(int)
        terminate_counts: Dict[str, int] = defaultdict(int)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        perturbation_samples = 0
        rare_samples = 0
        split_by_episode = {
            row["episode_id"]: row.get("split_assigned", "train")
            for row in episode_rows
        }
        for row in index_rows:
            contact_counts[row["contact_state"]] += 1
            terrain_counts[row["terrain"]] += 1
            gait_counts[row["gait"]] += 1
            split_counts[split_by_episode.get(row["episode_id"], "train")] += 1
            perturbation_samples += int(bool(row["perturbation_active"]))
            rare_samples += int(bool(row["rare_contact"]))
            grouped[row["bucket_key"]].append(row)

        for episode in episode_rows:
            terminate_counts[str(episode.get("terminate_by", "unknown"))] += 1

        buckets = []
        for bucket_key, rows in sorted(grouped.items()):
            grf_values = np.asarray([r["grf_total_n"] for r in rows], dtype=np.float64)
            ext_values = np.asarray([r["external_force_n"] for r in rows], dtype=np.float64)
            share_values = np.asarray(
                [r.get("grf_load_share_max", 0.0) for r in rows], dtype=np.float64
            )
            first = rows[0]
            buckets.append(
                {
                    "bucket_key": bucket_key,
                    "contact_state": first["contact_state"],
                    "contact_bits": CONTACT_STATE_BITS.get(first["contact_state"]),
                    "perturbation_active": bool(first["perturbation_active"]),
                    "perturbation_level": first.get("perturbation_level", ""),
                    "speed_bin": first.get("speed_bin", ""),
                    "terrain": first["terrain"],
                    "gait": first["gait"],
                    "samples": len(rows),
                    "samples_seen": len(rows),
                    "randomization_groups": len(
                        {r.get("randomization_group_id", "") for r in rows}
                    ),
                    # The statistic total GRF cannot see: statics pins the total
                    # near body weight whatever the split across feet.
                    "grf_load_share_max": {
                        "median": float(np.median(share_values)),
                        "iqr": float(
                            np.percentile(share_values, 75)
                            - np.percentile(share_values, 25)
                        ),
                    },
                    "grf_total_n": {
                        "mean": float(np.mean(grf_values)),
                        "std": float(np.std(grf_values)),
                        "min": float(np.min(grf_values)),
                        "max": float(np.max(grf_values)),
                    },
                    "external_force_n": {
                        "mean": float(np.mean(ext_values)),
                        "std": float(np.std(ext_values)),
                        "min": float(np.min(ext_values)),
                        "max": float(np.max(ext_values)),
                    },
                }
            )

        n_samples = len(index_rows)
        return {
            "dataset_path": str(index_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "completed_at": completed_at,
            "file_size_bytes": int(total_size_bytes),
            "metadata": self._json_compatible(metadata or {}),
            "collection": collection,
            "export": {
                "samples": n_samples,
                "perturbation_samples": perturbation_samples,
                "perturbation_ratio": (
                    perturbation_samples / n_samples if n_samples else 0.0
                ),
                "rare_contact_samples": rare_samples,
                "active_buckets": len(buckets),
                "max_per_bucket": max_per_bucket,
                "contact_state_counts": dict(sorted(contact_counts.items())),
                "terrain_counts": dict(sorted(terrain_counts.items())),
                "gait_counts": dict(sorted(gait_counts.items())),
                "split_counts": dict(sorted(split_counts.items())),
                "terminate_by_counts": dict(sorted(terminate_counts.items())),
            },
            "episodes": [
                {
                    "episode_id": episode.get("episode_id"),
                    "terrain": episode.get("terrain"),
                    "gait": episode.get("gait"),
                    "timestamp": episode.get("timestamp"),
                    "n_steps": episode.get("n_steps"),
                    "duration_s": episode.get("duration_s"),
                    "friction": episode.get("friction"),
                    "payload_kg": episode.get("payload_kg"),
                    "terminate_by": episode.get("terminate_by"),
                    "terminate_reason": episode.get("terminate_reason"),
                    "split_assigned": episode.get("split_assigned"),
                    "randomization_group_id": episode.get("randomization_group_id"),
                    "seed": episode.get("seed"),
                }
                for episode in episode_rows
            ],
            "buckets": buckets,
        }

    def _current_run_summary(
        self,
        *,
        index_path: Path,
        run_dir: Path,
        metadata: dict | None,
        max_per_bucket: Optional[int],
    ) -> Dict[str, Any]:
        """Build the summary record for the current in-memory collection."""
        total_size_bytes = sum(
            path.stat().st_size for path in run_dir.rglob("*.parquet")
        )
        return self._build_run_record(
            index_rows=self.index_rows(
                max_per_bucket=max_per_bucket, shuffle=False, seed=None
            ),
            episode_rows=[m.to_row() for m in self.episodes.values()],
            run_dir=run_dir,
            index_path=index_path,
            completed_at=self._summary_timestamp(),
            total_size_bytes=total_size_bytes,
            metadata=metadata,
            collection={
                "samples_stored": self.total_samples_stored,
                "samples_seen": self.total_samples_seen,
                "perturbation_samples_stored": self.total_perturbation_stored,
                "rare_contact_samples_stored": self.total_rare_stored,
                "episodes": len(self.episodes),
                "episode_ids": self.episodes_collected,
                "bucket_capacity": self.bucket_capacity,
            },
            max_per_bucket=max_per_bucket,
        )

    @staticmethod
    def _aggregate_dataset_runs(datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Recompute whole-dataset totals from per-run records."""
        contact_counts: Dict[str, int] = defaultdict(int)
        terrain_counts: Dict[str, int] = defaultdict(int)
        gait_counts: Dict[str, int] = defaultdict(int)
        split_counts: Dict[str, int] = defaultdict(int)
        terminate_counts: Dict[str, int] = defaultdict(int)
        bucket_counts: Dict[str, int] = defaultdict(int)
        grf_accumulators: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "n": 0.0,
                "sum": 0.0,
                "sum_squares": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
        )
        perturbation_force_accumulators: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "n": 0.0,
                "sum": 0.0,
                "sum_squares": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
        )

        total_size_bytes = 0
        samples = 0
        perturbation_samples = 0
        rare_samples = 0
        collection_samples_stored = 0
        collection_samples_seen = 0
        episodes = 0

        for run in datasets.values():
            export = run.get("export", {})
            collection = run.get("collection", {})
            total_size_bytes += int(run.get("file_size_bytes") or 0)
            samples += int(export.get("samples", 0))
            perturbation_samples += int(export.get("perturbation_samples", 0))
            rare_samples += int(export.get("rare_contact_samples", 0))
            collection_samples_stored += int(collection.get("samples_stored", 0))
            collection_samples_seen += int(collection.get("samples_seen", 0))
            episodes += int(collection.get("episodes", 0))

            for state, count in export.get("contact_state_counts", {}).items():
                contact_counts[state] += int(count)
            for terrain, count in export.get("terrain_counts", {}).items():
                terrain_counts[terrain] += int(count)
            for gait, count in export.get("gait_counts", {}).items():
                gait_counts[gait] += int(count)
            for split, count in export.get("split_counts", {}).items():
                split_counts[split] += int(count)
            for outcome, count in export.get("terminate_by_counts", {}).items():
                terminate_counts[outcome] += int(count)

            for bucket in run.get("buckets", []):
                bucket_name = bucket.get("bucket_key") or (
                    f"{bucket.get('contact_state')} | "
                    f"pert={bucket.get('perturbation_level', 'none')} | "
                    f"{bucket.get('speed_bin', '')} | "
                    f"{bucket.get('terrain')} | {bucket.get('gait')}"
                )
                bucket_samples = int(bucket.get("samples", 0))
                bucket_counts[bucket_name] += bucket_samples

                grf = bucket.get("grf_total_n", {})
                if bucket_samples > 0 and grf:
                    mean = float(grf["mean"])
                    std = float(grf["std"])
                    acc = grf_accumulators[bucket_name]
                    acc["n"] += bucket_samples
                    acc["sum"] += bucket_samples * mean
                    acc["sum_squares"] += bucket_samples * (
                        std * std + mean * mean
                    )
                    acc["min"] = min(acc["min"], float(grf["min"]))
                    acc["max"] = max(acc["max"], float(grf["max"]))

                external_force = bucket.get("external_force_n", {})
                if (
                    bucket_samples > 0
                    and bucket.get("perturbation_active")
                    and external_force
                ):
                    mean = float(external_force["mean"])
                    std = float(external_force["std"])
                    acc = perturbation_force_accumulators[bucket_name]
                    acc["n"] += bucket_samples
                    acc["sum"] += bucket_samples * mean
                    acc["sum_squares"] += bucket_samples * (
                        std * std + mean * mean
                    )
                    acc["min"] = min(acc["min"], float(external_force["min"]))
                    acc["max"] = max(acc["max"], float(external_force["max"]))

        grf_by_bucket = {}
        for bucket_name, acc in sorted(grf_accumulators.items()):
            n_samples = int(acc["n"])
            mean = acc["sum"] / n_samples
            variance = max(acc["sum_squares"] / n_samples - mean * mean, 0.0)
            grf_by_bucket[bucket_name] = {
                "mean": mean,
                "std": variance ** 0.5,
                "min": acc["min"],
                "max": acc["max"],
                "samples": n_samples,
            }

        perturbation_force_by_bucket = {}
        perturbation_force_samples = 0
        perturbation_force_sum = 0.0
        perturbation_force_sum_squares = 0.0
        perturbation_force_min = float("inf")
        perturbation_force_max = float("-inf")
        for bucket_name, acc in sorted(perturbation_force_accumulators.items()):
            n_samples = int(acc["n"])
            mean = acc["sum"] / n_samples
            variance = max(acc["sum_squares"] / n_samples - mean * mean, 0.0)
            perturbation_force_by_bucket[bucket_name] = {
                "mean": mean,
                "std": variance ** 0.5,
                "min": acc["min"],
                "max": acc["max"],
                "samples": n_samples,
            }
            perturbation_force_samples += n_samples
            perturbation_force_sum += acc["sum"]
            perturbation_force_sum_squares += acc["sum_squares"]
            perturbation_force_min = min(perturbation_force_min, acc["min"])
            perturbation_force_max = max(perturbation_force_max, acc["max"])

        if perturbation_force_samples:
            perturbation_force_mean = perturbation_force_sum / perturbation_force_samples
            perturbation_force_variance = max(
                perturbation_force_sum_squares / perturbation_force_samples
                - perturbation_force_mean * perturbation_force_mean,
                0.0,
            )
            perturbation_force = {
                "mean": perturbation_force_mean,
                "std": perturbation_force_variance ** 0.5,
                "min": perturbation_force_min,
                "max": perturbation_force_max,
                "samples": perturbation_force_samples,
            }
        else:
            perturbation_force = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "samples": 0,
            }

        return {
            "dataset_files": len(datasets),
            "total_size_bytes": total_size_bytes,
            "samples": samples,
            "perturbation_samples": perturbation_samples,
            "perturbation_ratio": (
                perturbation_samples / samples if samples else 0.0
            ),
            "rare_contact_samples": rare_samples,
            "collection_samples_stored": collection_samples_stored,
            "collection_samples_seen": collection_samples_seen,
            "episodes": episodes,
            "active_buckets": len(bucket_counts),
            "contact_state_counts": dict(sorted(contact_counts.items())),
            "terrain_counts": dict(sorted(terrain_counts.items())),
            "gait_counts": dict(sorted(gait_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "terminate_by_counts": dict(sorted(terminate_counts.items())),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "grf_total_n_by_bucket": grf_by_bucket,
            "perturbation_force_n": perturbation_force,
            "perturbation_force_n_by_bucket": perturbation_force_by_bucket,
        }

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        """Write JSON beside its destination and atomically replace the file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def update_dataset_summary(
        self,
        index_path: str | Path,
        *,
        run_dir: str | Path | None = None,
        metadata: dict | None = None,
        max_per_bucket: Optional[int] = None,
    ) -> Path:
        """
        Add or replace this run in the persistent whole-dataset summary.

        The run's ``index.parquet`` absolute path is its stable key, so calling
        this method twice for the same run is idempotent. The summary is re-read
        immediately before updating to pick up runs written since startup.
        """
        if self.dataset_summary_path is None:
            raise ValueError("dataset_summary_path was not configured")

        index_path = Path(index_path)
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Cannot update dataset summary before the index is saved: {index_path}"
            )
        resolved_run_dir = (
            Path(run_dir) if run_dir is not None else index_path.parent
        )
        memory = self.load_dataset_summary()
        datasets = dict(memory["datasets"])
        datasets[str(index_path.resolve())] = self._current_run_summary(
            index_path=index_path,
            run_dir=resolved_run_dir,
            metadata=metadata,
            max_per_bucket=max_per_bucket,
        )

        now = self._summary_timestamp()
        payload = {
            "schema_version": DATASET_SUMMARY_SCHEMA_VERSION,
            "compat_key": self.aggregate_compat_key(),
            "created_at": memory.get("created_at") or now,
            "updated_at": now,
            "summary": self._aggregate_dataset_runs(datasets),
            "datasets": dict(sorted(datasets.items())),
        }
        self._atomic_write_json(self.dataset_summary_path, payload)
        self.dataset_memory = payload
        return self.dataset_summary_path


# ══════════════════════════════════════════════════════════════════════════════
# QUICK USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  DatasetBucketSystem — Minimal Usage Example\n")

    rng = np.random.default_rng(0)

    from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config
    from mpx.utils.dataset_collection.contact_labeling import debounce_sequence
    from mpx.utils.dataset_collection.dataset_schema import (
        EpisodeOutcome,
        assign_split,
        dt_since_transition,
        empty_arrays,
        episode_timestamp,
    )

    bucket_sys = DatasetBucketSystem(config=dataset_collection_config.bucket)

    CONTROL_HZ = 50.0

    def make_fake_episode(
        episode_id:      str,
        T:               int,
        contact_pattern: Tuple[int, int, int, int],
        ext_force_mag:   float,
        gait:            GaitType,
        terrain:         TerrainType,
        outcome:         EpisodeOutcome,
    ) -> EpisodeRecord:
        """Synthetic episode for demonstration only."""
        arrays = empty_arrays(T)
        arrays["t"][:] = np.arange(T, dtype=np.int32)
        arrays["time_s"][:] = np.arange(T, dtype=np.float32) / CONTROL_HZ
        for name in ("joint_pos", "joint_vel", "joint_torque", "imu_acc",
                     "imu_gyro", "foot_pos_base", "foot_vel_base", "base_lin_vel"):
            arrays[name][:] = rng.standard_normal(arrays[name].shape).astype(np.float32)

        grf = np.zeros((T, 4, 3), dtype=np.float32)
        for i, in_contact in enumerate(contact_pattern):
            if in_contact:
                grf[:, i, 2] = rng.normal(60.0, 20.0, T).clip(0)
                grf[:, i, 0] = rng.normal(0.0, 8.0, T)
        arrays["grf_base"][:] = grf.reshape(T, 12)
        arrays["grf_mean_n"][:] = np.linalg.norm(grf, axis=2)
        arrays["grf_world"][:] = grf.reshape(T, 12)

        t0, t1 = T // 3, 2 * T // 3
        arrays["external_force"][t0:t1] = rng.normal(0, ext_force_mag, (t1 - t0, 3))
        arrays["cmd_base_vel"][:, 0] = 0.5

        # Same labelling path as production, so the fixtures cannot drift from it.
        contacts = debounce_sequence(np.linalg.norm(grf, axis=2))
        arrays["contact"][:] = contacts
        arrays["rare_contact"][:] = [is_rare_contact(bits) for bits in contacts]
        arrays["dt_since_transition"][:] = dt_since_transition(contacts, 1.0 / CONTROL_HZ)

        metadata = EpisodeMetadata(
            episode_id=episode_id,
            robot="go2",
            scene=terrain.value,
            terrain=terrain.value,
            gait=gait.value,
            timestamp=episode_timestamp(),
            ended_at=episode_timestamp(),
            control_hz=CONTROL_HZ,
            friction=float(rng.uniform(0.2, 1.0)),
            payload_kg=float(rng.uniform(0.0, 3.0)),
            terminate_by=outcome.value,
            terminate_reason="demo",
            body_weight_n=DEFAULT_BODY_WEIGHT_N,
            split_assigned=assign_split(episode_id),
        )
        return EpisodeRecord(metadata=metadata, arrays=arrays)

    configs = [
        ("trot_flat_001",    GaitType.TROT,       TerrainType.FLAT,   (1, 0, 0, 1), 30.0, EpisodeOutcome.SUCCESS),
        ("trot_rough_001",   GaitType.TROT,       TerrainType.ROUGH,  (0, 1, 1, 0), 50.0, EpisodeOutcome.SUCCESS),
        ("crawl_flat_001",   GaitType.CRAWL,      TerrainType.FLAT,   (0, 1, 1, 1), 10.0, EpisodeOutcome.TRUNCATED),
        ("balance_flat_001", GaitType.BALANCE,    TerrainType.FLAT,   (1, 1, 1, 1), 80.0, EpisodeOutcome.FAILURE),
        ("balance_stair_01", GaitType.BALANCE,    TerrainType.STAIRS, (1, 1, 0, 1), 40.0, EpisodeOutcome.FAILURE),
        ("flight_flat_001",  GaitType.BOUND,      TerrainType.FLAT,   (0, 0, 0, 0), 20.0, EpisodeOutcome.SUCCESS),
        ("single_flat_001",  GaitType.TRANSITION, TerrainType.FLAT,   (1, 0, 0, 0), 20.0, EpisodeOutcome.FAILURE),
    ]

    print("  Adding episodes:")
    for ep_id, gait, terrain, contact, ext_mag, outcome in configs:
        record = make_fake_episode(ep_id, 3_000, contact, ext_mag, gait, terrain, outcome)
        result = bucket_sys.add_episode(record)
        print(f"    {ep_id:<22}  added={result['added']:>5}  "
              f"rare={result['rare']:>5}  split={record.metadata.split_assigned}")

    print()
    bucket_sys.print_summary()

    rows = bucket_sys.index_rows(max_per_bucket=1_000)
    print(f"\n  Label index: {len(rows):,} rows")
    print(f"    first row: {rows[0]}")
    # A sampler picks W at training time and skips rows that start before t=0.
    for window_size in (10, 30, 60):
        usable = sum(1 for row in rows if row["t"] >= window_size - 1)
        print(f"    usable with W={window_size:>3}: {usable:,} rows")
    print()
