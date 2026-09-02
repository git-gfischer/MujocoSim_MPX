"""
dataset_collection/dataset_bucket_system.py
========================
Balanced dataset collection system for quadruped proprioceptive signals
collected in MuJoCo. Designed for multi-task learning of:
    - Contact state classification  (12-class set, FL FR RL RR bit patterns)
    - Ground reaction force estimation per foot  (regression, 4×3 world-frame [N])
    - External base force estimation  (regression, 3D vector)

Bucket key: (contact_state, perturbation_active, terrain, gait_type)

Balancing strategy:
    - Reservoir sampling per bucket (uniform coverage over all seen windows)
    - Perturbation ratio enforcement (hard minimum fraction of perturbed windows)
    - GRF diversity monitored analytically post-collection (std per bucket)
      rather than enforced via binning at collection time

An example can be found at the bottom of the file.
"""

import random
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Proprioceptive vector layout (12-DoF quadruped, 4 feet).
# jp[12] | jv[12] | torque[12] | imu_acc[3] | imu_gyro[3] | foot_pos_base[12] | foot_vel_base[12]
PROPRIO_DIM = 66
PROPRIO_CHANNEL_SLICES: Dict[str, Tuple[int, int]] = {
    "joint_pos": (0, 12),
    "joint_vel": (12, 24),
    "torque": (24, 36),
    "imu_acc": (36, 39),
    "imu_gyro": (39, 42),
    "foot_pos_base": (42, 54),
    "foot_vel_base": (54, 66),
}

DATASET_SUMMARY_FILENAME = "dataset_summary.json"
DATASET_SUMMARY_SCHEMA_VERSION = 1


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

# 12-state set used for bucket keys and contact classification labels.
# Bit order: FL, FR, RL, RR  (e.g. "1110" → three stance feet, RR in swing).
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

CONTACT_STATE_BITS: Dict[str, str] = {
    name: "".join(str(b) for b in pattern)
    for name, pattern in VALID_CONTACT_STATES.items()
}

BINARY_TO_CONTACT_STATE: Dict[Tuple[int, int, int, int], str] = {
    pattern: name for name, pattern in VALID_CONTACT_STATES.items()
}

# Integer index for each state — used as the classification label
CONTACT_STATE_TO_IDX: Dict[str, int] = {
    k: i for i, k in enumerate(VALID_CONTACT_STATES)
}
IDX_TO_CONTACT_STATE: Dict[int, str] = {
    v: k for k, v in CONTACT_STATE_TO_IDX.items()
}


def resolve_dataset_output_path(
    *,
    prefix: str,
    robot: str,
    scene: str,
    gait: str,
    terrain: str,
    output_root_dir: str = "datasets",
    run_folder_pattern: str = "{prefix}_{robot}_{scene}_{gait}_{timestamp}",
    filename_pattern: str = "{prefix}_{robot}_{scene}_{gait}_{terrain}_{timestamp}.npz",
    use_timestamp: bool = True,
    timestamp: str | None = None,
    collect_out: str | None = None,
) -> tuple[Path, Path]:
    """
    Build run directory + ``.npz`` path for a collection session.

    Creates ``output_root_dir / run_folder / filename.npz`` automatically.
    If ``collect_out`` is set:
      - absolute path → used as-is (parent dirs created),
      - relative path → placed inside the run folder.

    Returns ``(run_dir, npz_path)`` as :class:`pathlib.Path` objects.
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

    folder_name = run_folder_pattern.format(**fmt).strip("_")
    run_dir = Path(output_root_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if collect_out:
        npz_path = Path(collect_out)
        if not npz_path.is_absolute():
            npz_path = run_dir / npz_path
    else:
        fname = filename_pattern.format(**fmt).strip("_")
        npz_path = run_dir / fname

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    return run_dir, npz_path


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Window:
    """
    One training sample: a contiguous slice of W timesteps from a single episode.
    Labels are taken at the LAST timestep of the window (the state being predicted).

    proprioceptive_data — shape (W, 66):
        Channels per timestep:
            [0 :12]  joint positions    (12 joints)
            [12:24]  joint velocities   (12 joints)
            [24:36]  joint torques      (12 joints)
            [36:39]  IMU accelerometer  (x, y, z)
            [39:42]  IMU gyroscope      (x, y, z)
            [42:54]  foot positions     (FL FR RL RR × xyz) in yaw-aligned base frame
            [54:66]  foot velocities    (FL FR RL RR × xyz) in yaw-aligned base frame
    """
    # ── Inputs ──────────────────────────────────────────────────────────────
    proprioceptive_data: np.ndarray     # (W, 66)  float32

    # ── Labels ──────────────────────────────────────────────────────────────
    contact_state:  str                 # Key from VALID_CONTACT_STATES
    grf_world:      np.ndarray          # (4, 3) world-frame [N] per foot FL..RR
    external_force: np.ndarray          # (3,)  [fx, fy, fz]      Newtons  float32

    @property
    def grf_magnitude(self) -> np.ndarray:
        """Per-foot force magnitude ``(4,)`` — used for contact-state derivation."""
        return np.linalg.norm(self.grf_world, axis=1).astype(np.float32)

    # ── Metadata (not exported to training arrays; used for analysis) ────────
    episode_id:          str
    gait_type:           GaitType
    terrain_type:        TerrainType
    perturbation_active: bool


@dataclass(frozen=True)
class BucketKey:
    """
    Immutable key that uniquely identifies one bucket.
    Frozen so it can be used as a dict key.

    Maximum theoretical buckets:
        12 contact states × 2 perturbation states × 3 terrains × 6 gait types
        = 432 — of which a subset will be physically reachable.
    """
    contact_state:       str
    perturbation_active: bool
    terrain:             TerrainType
    gait_type:           GaitType


# ══════════════════════════════════════════════════════════════════════════════
# BUCKET SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class DatasetBucketSystem:
    """
    Manages balanced dataset collection for quadruped proprioceptive data.

    Core responsibilities
    ─────────────────────
    1. Contact state derivation
       Converts per-foot GRF magnitudes to a contact state label using a
       force threshold. This avoids MuJoCo contact solver chatter at
       foot touchdown / liftoff boundaries.

    2. Window extraction
       Slides a fixed-size window across each continuous episode.
       Windows never cross episode boundaries.
       Label is always taken at the last timestep of the window.

    3. Bucket assignment and reservoir sampling
       Each (contact_state, perturbation_active, terrain, gait) combination
       gets its own capped bucket. Reservoir sampling ensures that after N
       windows have been seen, the bucket holds a uniform random sample.

    4. Perturbation ratio enforcement
       Hard minimum fraction of stored windows that must be perturbation-active.
       Monitored globally and reported in print_summary().

    5. GRF diversity monitoring  (post-collection diagnostic)
       Rather than controlling GRF diversity via binning, GRF standard
       deviation is tracked per bucket. Low-variance buckets indicate
       that more diverse episode configurations are needed for that state.

    6. Export
       Outputs numpy arrays ready for training, with optional per-bucket
       capping and train/val/test splitting.

    7. Persistent dataset memory
       Loads a small JSON summary of previous collection runs and updates it
       atomically when the current run closes. The memory contains statistics
       only; previous training arrays are never loaded into RAM.
    """

    def __init__(
        self,
        window_size:                  int   = 30,    # timesteps @ 50 Hz = 600 ms
        bucket_capacity:              int   = 5_000, # max windows stored per bucket
        contact_force_threshold:      float = 5.0,   # [N] min per-foot GRF = contact
        perturbation_force_threshold: float = 5.0,   # [N] min |F_ext| = perturbed
        min_perturbation_ratio:       float = 0.25,  # at least 25 % must be perturbed
        dataset_summary_path: str | Path | None = None,
    ):
        """
        Parameters
        ----------
        window_size
            Number of consecutive timesteps per training window.
            30 steps @ 50 Hz = 600 ms ≈ 1.5 trot cycles.

        bucket_capacity
            Maximum windows stored per bucket. When full, reservoir sampling
            randomly replaces existing samples so coverage stays uniform.

        contact_force_threshold
            Minimum per-foot GRF [N] to classify a foot as in contact.
            Prefer this over MuJoCo's binary solver flag to avoid high-
            frequency chatter at the contact boundary.

        perturbation_force_threshold
            Minimum magnitude of the external base force vector [N] for a
            window to be flagged as perturbation-active.

        min_perturbation_ratio
            Minimum fraction of all stored windows that must be perturb-active.
            Collect more perturbation episodes if this constraint is not met.

        dataset_summary_path
            Optional path to the persistent JSON summary for all collection
            runs. If the file already exists it is read during construction.
            Call :meth:`update_dataset_summary` after saving the current
            dataset, normally from the simulator shutdown hook.
        """
        self.window_size                  = window_size
        self.bucket_capacity              = bucket_capacity
        self.contact_force_threshold      = contact_force_threshold
        self.perturbation_force_threshold = perturbation_force_threshold
        self.min_perturbation_ratio       = min_perturbation_ratio

        # ── Main storage ─────────────────────────────────────────────────────
        self.buckets: Dict[BucketKey, List[Window]] = defaultdict(list)

        # Total windows ever offered to each bucket (including rejected ones).
        # Required for correct reservoir sampling probability.
        self.bucket_seen_count: Dict[BucketKey, int] = defaultdict(int)

        # Per-bucket GRF accumulator for post-collection diversity monitoring.
        # Stores total GRF (sum across feet) for every stored window.
        self._bucket_grf_totals: Dict[BucketKey, List[float]] = defaultdict(list)

        # ── Global counters ──────────────────────────────────────────────────
        self.total_windows_stored      = 0
        self.total_perturbation_stored = 0
        self.total_windows_seen        = 0   # includes discarded (invalid contact state)

        # ── Episode registry ─────────────────────────────────────────────────
        self.episodes_collected: List[str] = []

        # ── Persistent whole-dataset summary ─────────────────────────────────
        self.dataset_summary_path = (
            Path(dataset_summary_path) if dataset_summary_path is not None else None
        )
        self.dataset_memory = self.load_dataset_summary()

    # ══════════════════════════════════════════════════════════════════════════
    # LABEL DERIVATION  (private helpers)
    # ══════════════════════════════════════════════════════════════════════════

    def _derive_contact_state(self, grf_world: np.ndarray) -> Optional[str]:
        """
        Convert per-foot world-frame GRF to a contact state string.

        Uses the force **magnitude** per foot against ``contact_force_threshold``
        to avoid MuJoCo contact solver chatter at touchdown / liftoff.

        ``grf_world`` must be sampled at the control timestep (not averaged over
        the inter-step interval).

        Returns None if the binary pattern is not in :data:`VALID_CONTACT_STATES`.
        """
        grf = np.asarray(grf_world, dtype=np.float64).reshape(4, 3)
        magnitudes = np.linalg.norm(grf, axis=1)
        binary = tuple(
            1 if float(magnitudes[i]) > self.contact_force_threshold else 0
            for i in range(4)   # order: FL, FR, RL, RR
        )
        return BINARY_TO_CONTACT_STATE.get(binary)

    def _is_perturbation_active(self, external_force: np.ndarray) -> bool:
        """
        True if the external base force magnitude exceeds the threshold.
        """
        return float(np.linalg.norm(external_force)) > self.perturbation_force_threshold

    # ══════════════════════════════════════════════════════════════════════════
    # WINDOW EXTRACTION  (public — call once per collected episode)
    # ══════════════════════════════════════════════════════════════════════════

    def add_episode(
        self,
        episode_id:          str,
        proprioceptive_data: np.ndarray,    # (T, 66)
        grf_world:           np.ndarray,    # (T, 4, 3) world-frame [N]
        external_force:      np.ndarray,    # (T, 3)
        gait_type:           GaitType,
        terrain_type:        TerrainType,
        stride:              int = 1,
    ) -> Dict[str, int]:
        """
        Slide a window of size W across a continuous episode and route each
        valid window into the appropriate bucket.

        Windows NEVER cross episode boundaries. The label is always taken
        from the LAST timestep of each window (the state being predicted).

        Parameters
        ----------
        episode_id
            Unique string identifier, e.g. ``"trot_flat_042"``.

        proprioceptive_data
            Full-episode sensor array, shape ``(T, 66)``. ``T`` varies between
            episodes, since a simulator closes each episode on a task event.
            Channels: joint_pos[12] | joint_vel[12] | joint_torque[12]
                      | imu_acc[3]  | imu_gyro[3]
                      | foot_pos_base[12] | foot_vel_base[12]

        grf_world
            World-frame GRF per foot ``[N]``, shape ``(T, 4, 3)``, order FL FR RL RR.

        external_force
            External force applied to the robot base ``[N]``, shape ``(T, 3)``, xyz.

        gait_type
            Fixed gait type for this episode.

        terrain_type
            Fixed terrain type for this episode.

        stride
            Step between consecutive window start indices.
            1  → fully overlapping (maximum data, higher temporal correlation)
            W  → non-overlapping   (fewer samples, lower correlation)
            Recommended: stride=1 during collection, subsample at training time.

        Returns
        -------
        dict with counts: "added", "discarded_invalid_state", "total_seen"
        """
        T = proprioceptive_data.shape[0]

        assert T >= self.window_size, (
            f"Episode '{episode_id}' has {T} timesteps but window_size="
            f"{self.window_size}. Minimum episode length = window_size."
        )
        assert grf_world.shape == (T, 4, 3), "grf_world must be shape (T, 4, 3)"
        assert external_force.shape == (T, 3), "external_force must be shape (T, 3)"

        n_added    = 0
        n_discarded = 0

        for start in range(0, T - self.window_size + 1, stride):
            end = start + self.window_size

            self.total_windows_seen += 1

            label_grf   = grf_world[end - 1]         # (4, 3)
            label_ext_f = external_force[end - 1]    # (3,)

            contact_state = self._derive_contact_state(label_grf)

            if contact_state is None:
                n_discarded += 1
                continue

            perturb_active = self._is_perturbation_active(label_ext_f)

            window = Window(
                proprioceptive_data = proprioceptive_data[start:end].astype(np.float32).copy(),
                contact_state       = contact_state,
                grf_world           = label_grf.astype(np.float32).copy(),
                external_force      = label_ext_f.astype(np.float32).copy(),
                episode_id          = episode_id,
                gait_type           = gait_type,
                terrain_type        = terrain_type,
                perturbation_active = perturb_active,
            )

            key = BucketKey(
                contact_state       = contact_state,
                perturbation_active = perturb_active,
                terrain             = terrain_type,
                gait_type           = gait_type,
            )

            if self._reservoir_add(key, window):
                n_added += 1

        if episode_id not in self.episodes_collected:
            self.episodes_collected.append(episode_id)

        return {
            "added":                  n_added,
            "discarded_invalid_state": n_discarded,
            "total_seen":             n_added + n_discarded,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # RESERVOIR SAMPLING  (private)
    # ══════════════════════════════════════════════════════════════════════════

    def _reservoir_add(self, key: BucketKey, window: Window) -> bool:
        """
        Add a window to its bucket using reservoir sampling (Vitter's Algorithm R).

        While the bucket has free capacity every window is stored directly.
        Once full, each new window replaces a random existing one with
        probability (capacity / n_seen), ensuring that after N total windows
        the bucket holds a uniform random sample of all N windows seen.

        Returns True if the window was stored, False if discarded.
        """
        self.bucket_seen_count[key] += 1
        n_seen = self.bucket_seen_count[key]
        bucket = self.buckets[key]

        grf_total = float(np.sum(window.grf_magnitude))

        if len(bucket) < self.bucket_capacity:
            # Free space — store unconditionally
            bucket.append(window)
            self._bucket_grf_totals[key].append(grf_total)
            self._update_global_counters(window, delta=+1)
            return True

        # Bucket full — replace with probability capacity / n_seen
        replace_idx = random.randint(0, n_seen - 1)
        if replace_idx < self.bucket_capacity:
            evicted = bucket[replace_idx]
            self._update_global_counters(evicted, delta=-1)
            self._bucket_grf_totals[key][replace_idx] = grf_total
            bucket[replace_idx] = window
            self._update_global_counters(window, delta=+1)
            return True

        return False    # Rejected by reservoir sampling

    def _update_global_counters(self, window: Window, delta: int) -> None:
        """Increment or decrement global counters when a window is stored/evicted."""
        self.total_windows_stored      += delta
        self.total_perturbation_stored += delta if window.perturbation_active else 0

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def perturbation_ratio(self) -> float:
        """Fraction of currently stored windows that are perturbation-active."""
        if self.total_windows_stored == 0:
            return 0.0
        return self.total_perturbation_stored / self.total_windows_stored

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
            (key, len(windows))
            for key, windows in self.buckets.items()
            if len(windows) / self.bucket_capacity < threshold
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
            mean_total_grf  — average total GRF [N] across stored windows
            std_total_grf   — standard deviation  (low = poor GRF diversity)
            min_total_grf   — minimum observed total GRF [N]
            max_total_grf   — maximum observed total GRF [N]
            n_windows       — number of windows in this bucket
        """
        report = {}
        for key, grf_list in self._bucket_grf_totals.items():
            if not grf_list:
                continue
            arr = np.array(grf_list)
            report[key] = {
                "mean_total_grf": float(np.mean(arr)),
                "std_total_grf":  float(np.std(arr)),
                "min_total_grf":  float(np.min(arr)),
                "max_total_grf":  float(np.max(arr)),
                "n_windows":      len(arr),
            }
        return report

    def contact_state_counts(self) -> Dict[str, int]:
        """Total stored windows per contact state, summed across all buckets."""
        counts: Dict[str, int] = defaultdict(int)
        for key, windows in self.buckets.items():
            counts[key.contact_state] += len(windows)
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
            f"  stored={self.total_windows_stored:,}  "
            f"seen={self.total_windows_seen:,}  "
            f"episodes={len(self.episodes_collected)}  "
            f"active_buckets={len(self.buckets)}  "
            f"perturb={self.total_perturbation_stored:,} "
            f"({self.perturbation_ratio:.1%}, {ratio_flag})",
            flush=True,
        )

        counts = self.contact_state_counts()
        if not counts:
            print("  contact states: (empty)", flush=True)
            return

        print("  contact states:", flush=True)
        total = max(self.total_windows_stored, 1)
        for state, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            bits = CONTACT_STATE_BITS.get(state, "????")
            pct = 100.0 * count / total
            print(
                f"    {state:<14} {bits}  {count:>6,}  ({pct:5.1f}%)",
                flush=True,
            )

        # Top populated bucket keys (contact × perturb × terrain × gait).
        top = sorted(
            ((key, len(windows)) for key, windows in self.buckets.items()),
            key=lambda x: (-x[1], x[0].contact_state),
        )[:5]
        if top:
            print("  top buckets:", flush=True)
            for key, n in top:
                print(
                    f"    {key.contact_state:<14} | "
                    f"perturb={str(key.perturbation_active):<5} | "
                    f"{key.terrain.value:<6} | "
                    f"{key.gait_type.value:<10}  n={n:,}",
                    flush=True,
                )

    def print_summary(self) -> None:
        """Print a human-readable collection summary."""
        sep  = "═" * 64
        thin = "─" * 64

        print(sep)
        print("  DatasetBucketSystem — Collection Summary")
        print(sep)
        print(f"  Windows stored         : {self.total_windows_stored:>8,}")
        print(f"  Windows seen (total)   : {self.total_windows_seen:>8,}")
        print(f"  Perturbation windows   : {self.total_perturbation_stored:>8,}")
        ratio_flag = "✓" if self.perturbation_ratio_satisfied else "✗ UNSATISFIED"
        print(f"  Perturbation ratio     : {self.perturbation_ratio:>8.2%}  "
              f"(min: {self.min_perturbation_ratio:.2%}  {ratio_flag})")
        print(f"  Active buckets         : {len(self.buckets):>8,}")
        print(f"  Episodes collected     : {len(self.episodes_collected):>8,}")

        # ── Contact state breakdown ──────────────────────────────────────────
        print(thin)
        print("  Windows per contact state:")
        total = max(self.total_windows_stored, 1)
        for state, count in sorted(self.contact_state_counts().items()):
            bar = "█" * int(28 * count / total)
            pct = 100 * count / total
            bits = CONTACT_STATE_BITS.get(state, "????")
            print(f"    {state:<14} {bits}  {count:>7,}  ({pct:5.1f}%)  {bar}")

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
                print(f"     {key.contact_state:<14} | "
                      f"perturb={str(key.perturbation_active):<5} | "
                      f"{key.terrain.value:<6} | "
                      f"{key.gait_type.value:<10}  "
                      f"std={stats['std_total_grf']:.1f} N  "
                      f"n={stats['n_windows']}")
        else:
            print("  ✓  All buckets have sufficient GRF variance.")

        # ── Underpopulated bucket warning ────────────────────────────────────
        under = self.underpopulated_buckets(threshold=0.5)
        if under:
            print(thin)
            print(f"  ⚠  {len(under)} underpopulated buckets (<50% full):")
            for key, count in sorted(under, key=lambda x: x[1])[:10]:
                pct = 100 * count / self.bucket_capacity
                print(f"     {key.contact_state:<14} | "
                      f"perturb={str(key.perturbation_active):<5} | "
                      f"{key.terrain.value:<6} | "
                      f"{key.gait_type.value:<10}  "
                      f"{count}/{self.bucket_capacity}  ({pct:.0f}%)")
            if len(under) > 10:
                print(f"     ... and {len(under) - 10} more")

        print(sep)

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
                "windows": 0,
                "perturbation_windows": 0,
                "perturbation_ratio": 0.0,
                "collection_windows_stored": 0,
                "collection_windows_seen": 0,
                "episodes": 0,
                "active_buckets": 0,
                "contact_state_counts": {},
                "terrain_counts": {},
                "gait_counts": {},
                "bucket_counts": {},
                "grf_total_n_by_bucket": {},
                "perturbation_force_n": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "windows": 0,
                },
                "perturbation_force_n_by_bucket": {},
            },
            "datasets": {},
        }

    def load_dataset_summary(self) -> Dict[str, Any]:
        """
        Read the persistent whole-dataset summary.

        A missing file represents an empty dataset and is created only after a
        successfully saved collection run. Invalid files raise ``ValueError``
        instead of being silently overwritten.
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
        if payload.get("schema_version") != DATASET_SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported dataset summary schema in '{self.dataset_summary_path}': "
                f"{payload.get('schema_version')!r}"
            )
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
        Summarize existing ``.npz`` files when no memory file exists yet.

        This one-time migration keeps datasets collected before the persistent
        summary feature in the whole-dataset totals. Subsequent sessions load
        the small JSON file instead of reopening every dataset.
        """
        memory = self._empty_dataset_summary()
        if self.dataset_summary_path is None:
            return memory

        datasets = {}
        root = self.dataset_summary_path.parent
        if root.exists():
            for npz_path in sorted(root.rglob("*.npz")):
                record = self._saved_dataset_summary(npz_path)
                datasets[record["dataset_path"]] = record

        memory["datasets"] = datasets
        memory["summary"] = self._aggregate_dataset_runs(datasets)
        return memory

    def _saved_dataset_summary(self, npz_path: Path) -> Dict[str, Any]:
        """Build a run record directly from an existing saved dataset."""
        metadata = {}
        metadata_path = npz_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Cannot read dataset metadata '{metadata_path}': {exc}"
                ) from exc

        try:
            with np.load(npz_path, allow_pickle=False) as saved:
                y_contact = np.asarray(saved["y_contact"], dtype=np.int64).reshape(-1)
                y_grf = np.asarray(saved["y_grf"], dtype=np.float64)
                y_ext = np.asarray(saved["y_ext_force"], dtype=np.float64)
                if "contact_state_names" in saved:
                    state_names = [
                        str(name) for name in saved["contact_state_names"].tolist()
                    ]
                else:
                    state_names = list(CONTACT_STATE_TO_IDX)
        except (OSError, EOFError, KeyError, ValueError) as exc:
            raise ValueError(f"Cannot summarize dataset '{npz_path}': {exc}") from exc

        n_windows = len(y_contact)
        if y_grf.shape != (n_windows, 4, 3):
            raise ValueError(
                f"Dataset '{npz_path}' has invalid y_grf shape {y_grf.shape}"
            )
        if y_ext.shape != (n_windows, 3):
            raise ValueError(
                f"Dataset '{npz_path}' has invalid y_ext_force shape {y_ext.shape}"
            )

        terrain = str(metadata.get("terrain", "unknown"))
        gait = str(metadata.get("gait", "unknown"))
        perturbation = (
            np.linalg.norm(y_ext, axis=1) > self.perturbation_force_threshold
        )
        external_force_magnitudes = np.linalg.norm(y_ext, axis=1)
        grf_totals = np.linalg.norm(y_grf, axis=2).sum(axis=1)
        grouped_indices: Dict[Tuple[str, bool], List[int]] = defaultdict(list)
        for index, state_index in enumerate(y_contact):
            state = (
                state_names[int(state_index)]
                if 0 <= int(state_index) < len(state_names)
                else f"UNKNOWN_{int(state_index)}"
            )
            grouped_indices[(state, bool(perturbation[index]))].append(index)

        buckets = []
        contact_counts: Dict[str, int] = defaultdict(int)
        perturbation_windows = 0
        for (state, is_perturbed), indices in grouped_indices.items():
            grf_values = grf_totals[indices]
            external_force_values = external_force_magnitudes[indices]
            count = len(indices)
            contact_counts[state] += count
            if is_perturbed:
                perturbation_windows += count
            buckets.append(
                {
                    "contact_state": state,
                    "contact_bits": CONTACT_STATE_BITS.get(state),
                    "perturbation_active": is_perturbed,
                    "terrain": terrain,
                    "gait": gait,
                    "windows": count,
                    "windows_seen": count,
                    "grf_total_n": {
                        "mean": float(np.mean(grf_values)),
                        "std": float(np.std(grf_values)),
                        "min": float(np.min(grf_values)),
                        "max": float(np.max(grf_values)),
                    },
                    "external_force_n": {
                        "mean": float(np.mean(external_force_values)),
                        "std": float(np.std(external_force_values)),
                        "min": float(np.min(external_force_values)),
                        "max": float(np.max(external_force_values)),
                    },
                }
            )
        buckets.sort(
            key=lambda item: (
                item["contact_state"],
                item["perturbation_active"],
                item["terrain"],
                item["gait"],
            )
        )

        episode_ids = list(metadata.get("episodes_collected", []))
        episodes = int(metadata.get("episodes_stored", len(episode_ids)))
        x_shape = metadata.get("X_shape", [])
        window_size = x_shape[1] if len(x_shape) >= 2 else None
        completed_at = metadata.get("saved_at")
        if not completed_at:
            completed_at = datetime.fromtimestamp(
                npz_path.stat().st_mtime
            ).astimezone().isoformat(timespec="seconds")

        return {
            "dataset_path": str(npz_path.resolve()),
            "run_dir": str(npz_path.parent.resolve()),
            "completed_at": completed_at,
            "file_size_bytes": npz_path.stat().st_size,
            "metadata": self._json_compatible(metadata),
            "collection": {
                "windows_stored": int(
                    metadata.get("total_windows_stored", n_windows)
                ),
                "windows_seen": int(metadata.get("total_windows_seen", n_windows)),
                "perturbation_windows_stored": perturbation_windows,
                "episodes": episodes,
                "episode_ids": episode_ids,
                "window_size": window_size,
                "bucket_capacity": metadata.get("bucket_capacity"),
            },
            "export": {
                "windows": n_windows,
                "perturbation_windows": perturbation_windows,
                "perturbation_ratio": (
                    perturbation_windows / n_windows if n_windows else 0.0
                ),
                "active_buckets": len(buckets),
                "max_per_bucket": metadata.get("max_per_bucket"),
                "contact_state_counts": dict(sorted(contact_counts.items())),
                "terrain_counts": {terrain: n_windows},
                "gait_counts": {gait: n_windows},
            },
            "buckets": buckets,
        }

    def _current_run_summary(
        self,
        *,
        dataset_path: Path,
        run_dir: Path,
        metadata: dict | None,
        max_per_bucket: Optional[int],
    ) -> Dict[str, Any]:
        """Build the summary record for the current in-memory collection."""
        buckets = []
        contact_counts: Dict[str, int] = defaultdict(int)
        terrain_counts: Dict[str, int] = defaultdict(int)
        gait_counts: Dict[str, int] = defaultdict(int)
        perturbation_windows = 0
        exported_windows = 0

        for key, windows in self.buckets.items():
            pool = windows if max_per_bucket is None else windows[:max_per_bucket]
            n_windows = len(pool)
            if n_windows == 0:
                continue

            grf_totals = np.asarray(
                [float(np.sum(window.grf_magnitude)) for window in pool],
                dtype=np.float64,
            )
            external_force_magnitudes = np.asarray(
                [float(np.linalg.norm(window.external_force)) for window in pool],
                dtype=np.float64,
            )
            contact_counts[key.contact_state] += n_windows
            terrain_counts[key.terrain.value] += n_windows
            gait_counts[key.gait_type.value] += n_windows
            exported_windows += n_windows
            if key.perturbation_active:
                perturbation_windows += n_windows

            buckets.append(
                {
                    "contact_state": key.contact_state,
                    "contact_bits": CONTACT_STATE_BITS.get(key.contact_state),
                    "perturbation_active": key.perturbation_active,
                    "terrain": key.terrain.value,
                    "gait": key.gait_type.value,
                    "windows": n_windows,
                    "windows_seen": int(self.bucket_seen_count[key]),
                    "grf_total_n": {
                        "mean": float(np.mean(grf_totals)),
                        "std": float(np.std(grf_totals)),
                        "min": float(np.min(grf_totals)),
                        "max": float(np.max(grf_totals)),
                    },
                    "external_force_n": {
                        "mean": float(np.mean(external_force_magnitudes)),
                        "std": float(np.std(external_force_magnitudes)),
                        "min": float(np.min(external_force_magnitudes)),
                        "max": float(np.max(external_force_magnitudes)),
                    },
                }
            )

        buckets.sort(
            key=lambda item: (
                item["contact_state"],
                item["perturbation_active"],
                item["terrain"],
                item["gait"],
            )
        )
        perturbation_ratio = (
            perturbation_windows / exported_windows if exported_windows else 0.0
        )

        return {
            "dataset_path": str(dataset_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "completed_at": self._summary_timestamp(),
            "file_size_bytes": (
                dataset_path.stat().st_size if dataset_path.exists() else None
            ),
            "metadata": self._json_compatible(metadata or {}),
            "collection": {
                "windows_stored": self.total_windows_stored,
                "windows_seen": self.total_windows_seen,
                "perturbation_windows_stored": self.total_perturbation_stored,
                "episodes": len(self.episodes_collected),
                "episode_ids": list(self.episodes_collected),
                "window_size": self.window_size,
                "bucket_capacity": self.bucket_capacity,
            },
            "export": {
                "windows": exported_windows,
                "perturbation_windows": perturbation_windows,
                "perturbation_ratio": perturbation_ratio,
                "active_buckets": len(buckets),
                "max_per_bucket": max_per_bucket,
                "contact_state_counts": dict(sorted(contact_counts.items())),
                "terrain_counts": dict(sorted(terrain_counts.items())),
                "gait_counts": dict(sorted(gait_counts.items())),
            },
            "buckets": buckets,
        }

    @staticmethod
    def _aggregate_dataset_runs(datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Recompute whole-dataset totals from per-run records."""
        contact_counts: Dict[str, int] = defaultdict(int)
        terrain_counts: Dict[str, int] = defaultdict(int)
        gait_counts: Dict[str, int] = defaultdict(int)
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
        windows = 0
        perturbation_windows = 0
        collection_windows_stored = 0
        collection_windows_seen = 0
        episodes = 0

        for run in datasets.values():
            export = run.get("export", {})
            collection = run.get("collection", {})
            total_size_bytes += int(run.get("file_size_bytes") or 0)
            windows += int(export.get("windows", 0))
            perturbation_windows += int(export.get("perturbation_windows", 0))
            collection_windows_stored += int(collection.get("windows_stored", 0))
            collection_windows_seen += int(collection.get("windows_seen", 0))
            episodes += int(collection.get("episodes", 0))

            for state, count in export.get("contact_state_counts", {}).items():
                contact_counts[state] += int(count)
            for terrain, count in export.get("terrain_counts", {}).items():
                terrain_counts[terrain] += int(count)
            for gait, count in export.get("gait_counts", {}).items():
                gait_counts[gait] += int(count)
            for bucket in run.get("buckets", []):
                bucket_name = (
                    f"{bucket.get('contact_state')} | "
                    f"perturb={str(bucket.get('perturbation_active')).lower()} | "
                    f"{bucket.get('terrain')} | {bucket.get('gait')}"
                )
                bucket_windows = int(bucket.get("windows", 0))
                bucket_counts[bucket_name] += bucket_windows

                grf = bucket.get("grf_total_n", {})
                if bucket_windows > 0 and grf:
                    mean = float(grf["mean"])
                    std = float(grf["std"])
                    acc = grf_accumulators[bucket_name]
                    acc["n"] += bucket_windows
                    acc["sum"] += bucket_windows * mean
                    acc["sum_squares"] += bucket_windows * (
                        std * std + mean * mean
                    )
                    acc["min"] = min(acc["min"], float(grf["min"]))
                    acc["max"] = max(acc["max"], float(grf["max"]))

                external_force = bucket.get("external_force_n", {})
                if (
                    bucket_windows > 0
                    and bucket.get("perturbation_active")
                    and external_force
                ):
                    mean = float(external_force["mean"])
                    std = float(external_force["std"])
                    acc = perturbation_force_accumulators[bucket_name]
                    acc["n"] += bucket_windows
                    acc["sum"] += bucket_windows * mean
                    acc["sum_squares"] += bucket_windows * (
                        std * std + mean * mean
                    )
                    acc["min"] = min(acc["min"], float(external_force["min"]))
                    acc["max"] = max(acc["max"], float(external_force["max"]))

        grf_by_bucket = {}
        for bucket_name, acc in sorted(grf_accumulators.items()):
            n_windows = int(acc["n"])
            mean = acc["sum"] / n_windows
            variance = max(acc["sum_squares"] / n_windows - mean * mean, 0.0)
            grf_by_bucket[bucket_name] = {
                "mean": mean,
                "std": variance ** 0.5,
                "min": acc["min"],
                "max": acc["max"],
                "windows": n_windows,
            }

        perturbation_force_by_bucket = {}
        perturbation_force_windows = 0
        perturbation_force_sum = 0.0
        perturbation_force_sum_squares = 0.0
        perturbation_force_min = float("inf")
        perturbation_force_max = float("-inf")
        for bucket_name, acc in sorted(perturbation_force_accumulators.items()):
            n_windows = int(acc["n"])
            mean = acc["sum"] / n_windows
            variance = max(acc["sum_squares"] / n_windows - mean * mean, 0.0)
            perturbation_force_by_bucket[bucket_name] = {
                "mean": mean,
                "std": variance ** 0.5,
                "min": acc["min"],
                "max": acc["max"],
                "windows": n_windows,
            }
            perturbation_force_windows += n_windows
            perturbation_force_sum += acc["sum"]
            perturbation_force_sum_squares += acc["sum_squares"]
            perturbation_force_min = min(perturbation_force_min, acc["min"])
            perturbation_force_max = max(perturbation_force_max, acc["max"])

        if perturbation_force_windows:
            perturbation_force_mean = perturbation_force_sum / perturbation_force_windows
            perturbation_force_variance = max(
                perturbation_force_sum_squares / perturbation_force_windows
                - perturbation_force_mean * perturbation_force_mean,
                0.0,
            )
            perturbation_force = {
                "mean": perturbation_force_mean,
                "std": perturbation_force_variance ** 0.5,
                "min": perturbation_force_min,
                "max": perturbation_force_max,
                "windows": perturbation_force_windows,
            }
        else:
            perturbation_force = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "windows": 0,
            }

        return {
            "dataset_files": len(datasets),
            "total_size_bytes": total_size_bytes,
            "windows": windows,
            "perturbation_windows": perturbation_windows,
            "perturbation_ratio": (
                perturbation_windows / windows if windows else 0.0
            ),
            "collection_windows_stored": collection_windows_stored,
            "collection_windows_seen": collection_windows_seen,
            "episodes": episodes,
            "active_buckets": len(bucket_counts),
            "contact_state_counts": dict(sorted(contact_counts.items())),
            "terrain_counts": dict(sorted(terrain_counts.items())),
            "gait_counts": dict(sorted(gait_counts.items())),
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
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def update_dataset_summary(
        self,
        dataset_path: str | Path,
        *,
        run_dir: str | Path | None = None,
        metadata: dict | None = None,
        max_per_bucket: Optional[int] = None,
    ) -> Path:
        """
        Add or replace this run in the persistent whole-dataset summary.

        The saved dataset's absolute path is its stable key, so calling this
        method twice for the same file is idempotent. The summary is re-read
        immediately before updating to pick up runs written since startup.
        """
        if self.dataset_summary_path is None:
            raise ValueError("dataset_summary_path was not configured")

        npz_path = Path(dataset_path)
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"Cannot update dataset summary before the dataset is saved: {npz_path}"
            )
        resolved_run_dir = (
            Path(run_dir) if run_dir is not None else npz_path.parent
        )
        memory = self.load_dataset_summary()
        datasets = dict(memory["datasets"])
        dataset_key = str(npz_path.resolve())
        datasets[dataset_key] = self._current_run_summary(
            dataset_path=npz_path,
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

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════════════

    def export(
        self,
        max_per_bucket: Optional[int] = None,
        shuffle:        bool          = True,
        seed:           Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Export all stored windows as numpy arrays ready for training.

        Parameters
        ----------
        max_per_bucket
            Hard cap on windows drawn from each bucket. Use this to enforce
            strict per-bucket balance at training time without modifying
            the stored data.

        shuffle
            Randomly permute the output arrays before returning.

        seed
            Random seed for reproducible shuffling.

        Returns
        -------
        X            : (N, W, 66)  proprioceptive windows  float32
        y_contact    : (N,)        contact state index      int64
        y_grf        : (N, 4, 3)    world-frame GRF per foot [N]  float32
        y_ext_force  : (N, 3)      external base force [N]  float32
        """
        X, y_contact, y_grf, y_ext_force = [], [], [], []

        for key, windows in self.buckets.items():
            pool = windows if max_per_bucket is None else windows[:max_per_bucket]
            for w in pool:
                X.append(w.proprioceptive_data)
                y_contact.append(CONTACT_STATE_TO_IDX[w.contact_state])
                y_grf.append(w.grf_world)
                y_ext_force.append(w.external_force)

        X           = np.array(X,           dtype=np.float32)
        y_contact   = np.array(y_contact,   dtype=np.int64)
        y_grf       = np.array(y_grf,       dtype=np.float32)
        y_ext_force = np.array(y_ext_force, dtype=np.float32)

        if shuffle:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(X))
            X, y_contact, y_grf, y_ext_force = (
                X[idx], y_contact[idx], y_grf[idx], y_ext_force[idx]
            )

        return X, y_contact, y_grf, y_ext_force

    def export_split(
        self,
        val_ratio:      float         = 0.15,
        test_ratio:     float         = 0.10,
        max_per_bucket: Optional[int] = None,
        seed:           Optional[int] = 42,
    ) -> Tuple[
        Tuple[np.ndarray, ...],
        Tuple[np.ndarray, ...],
        Tuple[np.ndarray, ...],
    ]:
        """
        Export pre-split train / val / test arrays.

        NOTE: This splits at the window level after shuffling. For strict
        evaluation, split at the episode level before calling add_episode()
        and use separate DatasetBucketSystem instances for train and eval.

        Returns
        -------
        (X_train, y_contact_train, y_grf_train, y_ext_train),
        (X_val,   y_contact_val,   y_grf_val,   y_ext_val),
        (X_test,  y_contact_test,  y_grf_test,  y_ext_test)
        """
        X, y_c, y_g, y_e = self.export(
            max_per_bucket=max_per_bucket, shuffle=True, seed=seed
        )

        N       = len(X)
        n_test  = int(N * test_ratio)
        n_val   = int(N * val_ratio)
        n_train = N - n_val - n_test

        s = {
            "train": slice(0,               n_train),
            "val":   slice(n_train,         n_train + n_val),
            "test":  slice(n_train + n_val, N),
        }

        def _s(sl):
            return X[sl], y_c[sl], y_g[sl], y_e[sl]

        return _s(s["train"]), _s(s["val"]), _s(s["test"])

    def save_npz(
        self,
        path: str | Path,
        *,
        max_per_bucket: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        metadata: dict | None = None,
        write_metadata: bool = False,
    ) -> Path:
        """Export arrays and write them to a compressed ``.npz`` file."""
        npz_path = Path(path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        X, y_contact, y_grf, y_ext = self.export(
            max_per_bucket=max_per_bucket,
            shuffle=shuffle,
            seed=seed if seed is not None else None,
        )
        np.savez_compressed(
            str(npz_path),
            X=X,
            y_contact=y_contact,
            y_grf=y_grf,
            y_ext_force=y_ext,
            contact_state_names=np.array(list(CONTACT_STATE_TO_IDX.keys())),
            contact_state_bits=np.array(list(CONTACT_STATE_BITS.values())),
            proprio_dim=np.int32(PROPRIO_DIM),
            proprio_channel_names=np.array(list(PROPRIO_CHANNEL_SLICES.keys())),
            proprio_channel_starts=np.array([s[0] for s in PROPRIO_CHANNEL_SLICES.values()]),
            proprio_channel_ends=np.array([s[1] for s in PROPRIO_CHANNEL_SLICES.values()]),
        )

        if write_metadata and metadata is not None:
            meta_path = npz_path.with_suffix(".json")
            payload = {
                **metadata,
                "npz_path": str(npz_path.resolve()),
                "n_windows": int(len(X)),
                "X_shape": list(X.shape),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "total_windows_stored": self.total_windows_stored,
                "episodes_collected": list(self.episodes_collected),
            }
            meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return npz_path

    def save_dataset(
        self,
        run_dir: str | Path,
        npz_path: str | Path,
        *,
        metadata: dict | None = None,
        write_metadata: bool = True,
        max_per_bucket: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Path:
        """Save ``.npz`` (and optional metadata) into a prepared run directory."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return self.save_npz(
            npz_path,
            max_per_bucket=max_per_bucket,
            shuffle=shuffle,
            seed=seed,
            metadata={**(metadata or {}), "run_dir": str(run_dir.resolve())},
            write_metadata=write_metadata,
        )


# ══════════════════════════════════════════════════════════════════════════════
# QUICK USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  DatasetBucketSystem — Minimal Usage Example\n")

    rng = np.random.default_rng(0)

    from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config

    bucket_cfg = dataset_collection_config.bucket
    bucket_sys = DatasetBucketSystem(
        window_size=bucket_cfg.window_size,
        bucket_capacity=bucket_cfg.bucket_capacity,
        contact_force_threshold=bucket_cfg.contact_force_threshold,
        perturbation_force_threshold=bucket_cfg.perturbation_force_threshold,
        min_perturbation_ratio=bucket_cfg.min_perturbation_ratio,
    )

    # 2. Simulate episodes and add them
    def make_fake_episode(
        T:               int,
        contact_pattern: Tuple[int, int, int, int],
        ext_force_mag:   float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synthetic episode data for demonstration only."""
        prop_data = rng.standard_normal((T, PROPRIO_DIM)).astype(np.float32)
        grf_world = np.zeros((T, 4, 3), dtype=np.float32)
        for i, c in enumerate(contact_pattern):
            if c:
                fz = rng.normal(60.0, 20.0, T).clip(0)
                grf_world[:, i, 2] = fz

        ext_force = np.zeros((T, 3), dtype=np.float32)
        t0, t1 = T // 3, 2 * T // 3
        ext_force[t0:t1] = rng.normal(0, ext_force_mag, (t1 - t0, 3))
        return prop_data, grf_world, ext_force

    configs = [
        ("trot_flat_001",    GaitType.TROT,       TerrainType.FLAT,   (1, 0, 0, 1), 30.0),
        ("trot_rough_001",   GaitType.TROT,       TerrainType.ROUGH,  (0, 1, 1, 0), 50.0),
        ("crawl_flat_001",   GaitType.CRAWL,      TerrainType.FLAT,   (0, 1, 1, 1), 10.0),
        ("balance_flat_001", GaitType.BALANCE,    TerrainType.FLAT,   (1, 1, 1, 1), 80.0),
        ("balance_stair_01", GaitType.BALANCE,    TerrainType.STAIRS, (1, 1, 0, 1), 40.0),
        ("flight_flat_001",  GaitType.BOUND,      TerrainType.FLAT,   (0, 0, 0, 0), 20.0),
        ("trans_flat_001",   GaitType.TRANSITION, TerrainType.FLAT,   (1, 1, 1, 0), 20.0),
    ]

    print("  Adding episodes:")
    for ep_id, gait, terrain, contact, ext_mag in configs:
        prop, grf, ext = make_fake_episode(T=3_000, contact_pattern=contact, ext_force_mag=ext_mag)
        result = bucket_sys.add_episode(
            episode_id          = ep_id,
            proprioceptive_data = prop,
            grf_world           = grf,
            external_force      = ext,
            gait_type           = gait,
            terrain_type        = terrain,
            stride              = 1,
        )
        print(f"    {ep_id:<22}  added={result['added']:>5}  "
              f"discarded={result['discarded_invalid_state']:>4}")

    # 3. Summary + GRF diversity report
    print()
    bucket_sys.print_summary()

    # 4. Export
    X, y_contact, y_grf, y_ext = bucket_sys.export(max_per_bucket=1_000)
    print(f"\n  Exported shapes:")
    print(f"    X            : {X.shape}")
    print(f"    y_contact    : {y_contact.shape}")
    print(f"    y_grf        : {y_grf.shape}")
    print(f"    y_ext_force  : {y_ext.shape}\n")
