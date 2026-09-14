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

import random
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mpx.utils.dataset_collection.dataset_schema import (
    EPISODE_SCHEMA_VERSION,
    FOOT_ORDER,
    EpisodeMetadata,
    EpisodeRecord,
    contact_bits_from_grf,
)
from mpx.utils.dataset_collection.episode_storage import (
    INDEX_FILENAME,
    EpisodeStore,
)

DATASET_SUMMARY_FILENAME = "dataset_summary.json"
DATASET_SUMMARY_SCHEMA_VERSION = 3


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

# The 12 named gait states. They are the taxonomy used for bucket keys and
# summaries — not the training label, which is the raw 4-bit vector stored on
# every timestep.
VALID_CONTACT_STATES: Dict[str, Tuple[int, int, int, int]] = {
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

# The remaining four patterns are the single-foot stances (1000, 0100, 0010,
# 0001). They are outliers of the gait taxonomy, not extra classes: they are
# kept, flagged, and bucketed together instead of being dropped or named.
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


def contact_state_name(bits: Sequence[int]) -> str:
    """Named gait state for a 4-bit pattern, or ``"RARE"`` when it has none."""
    key = tuple(int(b) for b in bits)
    return BINARY_TO_CONTACT_STATE.get(key, RARE_CONTACT_STATE)


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

@dataclass
class SampleRef:
    """
    One training sample: a labelled timestep, stored as a reference.

    ``t`` is the step whose contact state, GRF and perturbation flag the labels
    describe. A sampler picks its own window length ``W`` and materializes the
    window by slicing ``[t - W + 1 … t]`` out of
    ``episodes/<episode_id>.parquet``; no window length is stored here.

    The few label statistics kept here are what balancing and the summary need;
    everything else is read back from the episode file on demand.
    """

    episode_id:    str
    t:             int
    contact_state: str                        # named gait state or "RARE"
    contact_bits:  Tuple[int, int, int, int]  # FL FR RL RR
    grf_total_n:        float                 # sum of per-foot |GRF| at t
    external_force_n:   float                 # |F_ext| at t
    perturbation_active: bool
    gait_type:     GaitType
    terrain_type:  TerrainType

    @property
    def rare_contact(self) -> bool:
        """True when the pattern is outside the 12 named gait states."""
        return self.contact_state == RARE_CONTACT_STATE


@dataclass(frozen=True)
class BucketKey:
    """
    Immutable key that uniquely identifies one bucket.
    Frozen so it can be used as a dict key.

    Maximum theoretical buckets:
        13 contact states (12 named + RARE) × 2 perturbation states
        × 3 terrains × 6 gait types = 468 — of which a subset is physically
        reachable.
    """
    contact_state:       str
    perturbation_active: bool
    terrain:             TerrainType
    gait_type:           GaitType

    def label(self) -> str:
        """Stable human-readable key used in the index and JSON summaries."""
        return (
            f"{self.contact_state} | perturb={str(self.perturbation_active).lower()} | "
            f"{self.terrain.value} | {self.gait_type.value}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# BUCKET SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class DatasetBucketSystem:
    """
    Manages balanced dataset collection for quadruped proprioceptive data.

    Core responsibilities
    ─────────────────────
    1. Contact state derivation
       Converts per-foot GRF magnitudes to 4 contact binaries using a force
       threshold. This avoids MuJoCo contact solver chatter at foot touchdown /
       liftoff. Patterns outside the 12 named gait states are flagged
       ``RARE`` — kept and bucketed, never dropped.

    2. Label enumeration
       Records one reference per labelled timestep of each episode. No window
       length is involved: a sampler cuts windows later, and because a window
       is always taken from a single episode file it can never cross an
       episode boundary.

    3. Bucket assignment and reservoir sampling
       Each (contact_state, perturbation_active, terrain, gait) combination
       gets its own capped bucket. Reservoir sampling ensures that after N
       samples have been seen, the bucket holds a uniform random subset.

    4. Perturbation ratio enforcement
       Hard minimum fraction of stored samples that must be perturbation-active.
       Monitored globally and reported in print_summary().

    5. GRF diversity monitoring  (post-collection diagnostic)
       GRF standard deviation is tracked per bucket. Low-variance buckets
       indicate that more diverse episode configurations are needed.

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
        bucket_capacity:              int   = 5_000, # max samples stored per bucket
        contact_force_threshold:      float = 5.0,   # [N] min per-foot GRF = contact
        perturbation_force_threshold: float = 5.0,   # [N] min |F_ext| = perturbed
        min_perturbation_ratio:       float = 0.25,  # at least 25 % must be perturbed
        dataset_summary_path: str | Path | None = None,
        store: EpisodeStore | None = None,
    ):
        """
        Parameters
        ----------
        bucket_capacity
            Maximum sample references stored per bucket. When full, reservoir
            sampling randomly replaces existing entries so coverage stays uniform.

        contact_force_threshold
            Minimum per-foot GRF [N] to classify a foot as in contact.
            Prefer this over MuJoCo's binary solver flag to avoid high-
            frequency chatter at the contact boundary.

        perturbation_force_threshold
            Minimum magnitude of the external base force vector [N] for a
            sample to be flagged as perturbation-active.

        min_perturbation_ratio
            Minimum fraction of all stored samples that must be perturb-active.
            Collect more perturbation episodes if this constraint is not met.

        dataset_summary_path
            Optional path to the persistent JSON summary for all collection
            runs. If the file already exists it is read during construction.
            Call :meth:`update_dataset_summary` after saving the current
            dataset, normally from the simulator shutdown hook.

        store
            Parquet writer for this run. When ``None`` the system runs purely
            in memory — used by tests and by the ``__main__`` demo.
        """
        self.bucket_capacity              = bucket_capacity
        self.contact_force_threshold      = contact_force_threshold
        self.perturbation_force_threshold = perturbation_force_threshold
        self.min_perturbation_ratio       = min_perturbation_ratio
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

    def derive_contacts(self, grf_world: np.ndarray) -> np.ndarray:
        """
        Per-timestep contact binaries ``(T, 4)`` from world-frame GRF.

        Uses the force **magnitude** per foot against ``contact_force_threshold``
        to avoid MuJoCo contact solver chatter at touchdown / liftoff. The GRF
        must be sampled at the control timestep, not averaged over the
        inter-step interval.
        """
        return contact_bits_from_grf(grf_world, self.contact_force_threshold)

    def _is_perturbation_active(self, external_force_norm: float) -> bool:
        """True if the external base force magnitude exceeds the threshold."""
        return float(external_force_norm) > self.perturbation_force_threshold

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
            ``contact``, ``grf_world`` and ``external_force`` are read from the
            table; ``contact`` is expected to already be the GRF-thresholded
            ground truth (the recorder derives it with :meth:`derive_contacts`).

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

        contact = np.asarray(record.arrays["contact"], dtype=np.uint8).reshape(-1, 4)
        grf = np.asarray(record.arrays["grf_world"], dtype=np.float64).reshape(-1, 4, 3)
        ext = np.asarray(record.arrays["external_force"], dtype=np.float64).reshape(-1, 3)

        grf_totals = np.linalg.norm(grf, axis=2).sum(axis=1)
        ext_norms = np.linalg.norm(ext, axis=1)

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
            if state == RARE_CONTACT_STATE:
                n_rare += 1

            perturb_active = self._is_perturbation_active(ext_norms[label_index])

            sample = SampleRef(
                episode_id          = episode_id,
                t                   = label_index,
                contact_state       = state,
                contact_bits        = bits,
                grf_total_n         = float(grf_totals[label_index]),
                external_force_n    = float(ext_norms[label_index]),
                perturbation_active = perturb_active,
                gait_type           = gait_type,
                terrain_type        = terrain_type,
            )

            key = BucketKey(
                contact_state       = state,
                perturbation_active = perturb_active,
                terrain             = terrain_type,
                gait_type           = gait_type,
            )

            if self._reservoir_add(key, sample):
                n_added += 1
            else:
                n_rejected += 1

        return {
            "added":      n_added,
            "rejected":   n_rejected,
            "rare":       n_rare,
            "total_seen": n_added + n_rejected,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # RESERVOIR SAMPLING  (private)
    # ══════════════════════════════════════════════════════════════════════════

    def _reservoir_add(self, key: BucketKey, sample: SampleRef) -> bool:
        """
        Add a sample to its bucket using reservoir sampling (Vitter's Algorithm R).

        While the bucket has free capacity every sample is stored directly.
        Once full, each new sample replaces a random existing one with
        probability (capacity / n_seen), ensuring that after N total samples
        the bucket holds a uniform random subset of all N seen.

        Returns True if the sample was stored, False if discarded.
        """
        self.bucket_seen_count[key] += 1
        n_seen = self.bucket_seen_count[key]
        bucket = self.buckets[key]

        if len(bucket) < self.bucket_capacity:
            # Free space — store unconditionally
            bucket.append(sample)
            self._update_global_counters(sample, delta=+1)
            return True

        # Bucket full — replace with probability capacity / n_seen
        replace_idx = random.randint(0, n_seen - 1)
        if replace_idx < self.bucket_capacity:
            evicted = bucket[replace_idx]
            self._update_global_counters(evicted, delta=-1)
            bucket[replace_idx] = sample
            self._update_global_counters(sample, delta=+1)
            return True

        return False    # Rejected by reservoir sampling

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

    def underpopulated_buckets(self, threshold: float = 0.5) -> List[Tuple[BucketKey, int]]:
        """
        Return (key, count) pairs for buckets below the fill threshold.
        Use this after each collection batch to decide which episode
        configurations need more data.

        Parameters
        ----------
        threshold
            Fraction of bucket_capacity below which a bucket is flagged.
            0.5 → less than half full.
        """
        return [
            (key, len(samples))
            for key, samples in self.buckets.items()
            if len(samples) / self.bucket_capacity < threshold
        ]

    def grf_diversity_report(self) -> Dict[BucketKey, Dict[str, float]]:
        """
        Compute GRF diversity statistics per bucket.

        Since GRF balance is enforced through episode diversity rather than
        explicit binning, this is the primary diagnostic for checking whether
        the GRF regression head will see sufficient range within each bucket.

        A low std_total_grf for a bucket means the episode configurations
        feeding that bucket are too homogeneous — collect more varied episodes
        (different speeds, terrain roughness, perturbation magnitudes) for
        that (contact_state, gait, terrain) combination.

        Returns
        -------
        Dict mapping each BucketKey to:
            mean_total_grf  — average total GRF [N] across stored samples
            std_total_grf   — standard deviation  (low = poor GRF diversity)
            min_total_grf   — minimum observed total GRF [N]
            max_total_grf   — maximum observed total GRF [N]
            n_samples       — number of samples in this bucket
        """
        report = {}
        for key, samples in self.buckets.items():
            if not samples:
                continue
            arr = np.array([s.grf_total_n for s in samples], dtype=np.float64)
            report[key] = {
                "mean_total_grf": float(np.mean(arr)),
                "std_total_grf":  float(np.std(arr)),
                "min_total_grf":  float(np.min(arr)),
                "max_total_grf":  float(np.max(arr)),
                "n_samples":      len(arr),
            }
        return report

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
                counts[metadata.split if metadata else "train"] += 1
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
                print(f"    {key.label():<48}  n={n:,}", flush=True)

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

        # ── GRF diversity flag ───────────────────────────────────────────────
        print(thin)
        print("  GRF diversity (std of total GRF per bucket):")
        div = self.grf_diversity_report()
        LOW_STD_WARN = 15.0   # [N] — flag buckets with poor GRF spread
        low_var = [
            (k, v) for k, v in div.items()
            if v["std_total_grf"] < LOW_STD_WARN
        ]
        if low_var:
            print(f"  ⚠  {len(low_var)} buckets with low GRF variance "
                  f"(std < {LOW_STD_WARN} N) — consider more diverse episodes:")
            for key, stats in sorted(low_var, key=lambda x: x[1]["std_total_grf"])[:8]:
                print(f"     {key.label():<48}  "
                      f"std={stats['std_total_grf']:.1f} N  "
                      f"n={stats['n_samples']}")
        else:
            print("  ✓  All buckets have sufficient GRF variance.")

        # ── Underpopulated bucket warning ────────────────────────────────────
        under = self.underpopulated_buckets(threshold=0.5)
        if under:
            print(thin)
            print(f"  ⚠  {len(under)} underpopulated buckets (<50% full):")
            for key, count in sorted(under, key=lambda x: x[1])[:10]:
                pct = 100 * count / self.bucket_capacity
                print(f"     {key.label():<48}  "
                      f"{count}/{self.bucket_capacity}  ({pct:.0f}%)")
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

        Each row names the ``(episode_id, t)`` a sampler reads, the split its
        episode belongs to, and the bucket it was balanced into. No window
        length appears: the sampler chooses ``W`` and skips rows with
        ``t < W - 1``.

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
                        "split":               metadata.split if metadata else "train",
                        "terrain":             sample.terrain_type.value,
                        "gait":                sample.gait_type.value,
                        "contact_state":       sample.contact_state,
                        "contact_bits":        contact_bits_string(sample.contact_bits),
                        "rare_contact":        bool(sample.rare_contact),
                        "perturbation_active": bool(sample.perturbation_active),
                        "bucket_key":          bucket_label,
                        "grf_total_n":         float(sample.grf_total_n),
                        "external_force_n":    float(sample.external_force_n),
                    }
                )

        if shuffle:
            random.Random(seed).shuffle(rows)
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
            grouped[row["split"]].append(row)
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
    ) -> Path:
        """
        Write the episode metadata table and the balanced label index.

        Episode tables themselves were already written by :meth:`add_episode`.
        Returns the path of ``index.parquet``.
        """
        store = self.store
        if store is None:
            if run_dir is None:
                raise ValueError("save_dataset needs a store or a run_dir")
            store = EpisodeStore(Path(run_dir))

        store.write_episode_table(list(self.episodes.values()))
        index_path = store.write_index(
            self.index_rows(
                max_per_bucket=max_per_bucket, shuffle=shuffle, seed=seed
            )
        )

        if write_metadata and metadata is not None:
            payload = {
                **metadata,
                "run_dir": str(store.run_dir.resolve()),
                "index_path": str(index_path.resolve()),
                "episode_schema_version": EPISODE_SCHEMA_VERSION,
                "bucket_capacity": self.bucket_capacity,
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

    # ══════════════════════════════════════════════════════════════════════════
    # PERSISTENT WHOLE-DATASET SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _summary_timestamp() -> str:
        """Return an ISO-8601 timestamp including the local UTC offset."""
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @classmethod
    def _empty_dataset_summary(cls) -> Dict[str, Any]:
        """Create an empty in-memory representation of the summary file."""
        return {
            "schema_version": DATASET_SUMMARY_SCHEMA_VERSION,
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
        for row in index_rows:
            contact_counts[row["contact_state"]] += 1
            terrain_counts[row["terrain"]] += 1
            gait_counts[row["gait"]] += 1
            split_counts[row["split"]] += 1
            perturbation_samples += int(bool(row["perturbation_active"]))
            rare_samples += int(bool(row["rare_contact"]))
            grouped[row["bucket_key"]].append(row)

        for episode in episode_rows:
            terminate_counts[str(episode.get("terminate_by", "unknown"))] += 1

        buckets = []
        for bucket_key, rows in sorted(grouped.items()):
            grf_values = np.asarray([r["grf_total_n"] for r in rows], dtype=np.float64)
            ext_values = np.asarray([r["external_force_n"] for r in rows], dtype=np.float64)
            first = rows[0]
            buckets.append(
                {
                    "bucket_key": bucket_key,
                    "contact_state": first["contact_state"],
                    "contact_bits": (
                        CONTACT_STATE_BITS.get(first["contact_state"])
                        if first["contact_state"] != RARE_CONTACT_STATE
                        else None
                    ),
                    "perturbation_active": bool(first["perturbation_active"]),
                    "terrain": first["terrain"],
                    "gait": first["gait"],
                    "samples": len(rows),
                    "samples_seen": len(rows),
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
                    "split": episode.get("split"),
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
                    f"perturb={str(bucket.get('perturbation_active')).lower()} | "
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
    from mpx.utils.dataset_collection.dataset_schema import (
        EpisodeOutcome,
        assign_split,
        dt_since_transition,
        empty_arrays,
        episode_timestamp,
    )

    bucket_cfg = dataset_collection_config.bucket
    bucket_sys = DatasetBucketSystem(
        bucket_capacity=bucket_cfg.bucket_capacity,
        contact_force_threshold=bucket_cfg.contact_force_threshold,
        perturbation_force_threshold=bucket_cfg.perturbation_force_threshold,
        min_perturbation_ratio=bucket_cfg.min_perturbation_ratio,
    )

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
        arrays["grf_world"][:] = grf.reshape(T, 12)

        t0, t1 = T // 3, 2 * T // 3
        arrays["external_force"][t0:t1] = rng.normal(0, ext_force_mag, (t1 - t0, 3))

        contacts = bucket_sys.derive_contacts(grf)
        arrays["contact"][:] = contacts
        arrays["rare_contact"][:] = [
            contact_state_name(bits) == RARE_CONTACT_STATE for bits in contacts
        ]
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
            split=assign_split(episode_id),
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
              f"rare={result['rare']:>5}  split={record.metadata.split}")

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
