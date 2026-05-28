"""
dataset_bucket_system.py
========================
Balanced dataset collection system for quadruped proprioceptive signals
collected in MuJoCo. Designed for multi-task learning of:
    - Contact state classification  (9-class reduced set)
    - Ground reaction force estimation per foot  (regression, 4 values)
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
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple


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
# CONTACT STATE DEFINITIONS  (FL, FR, HL, HR) — 1 = contact, 0 = swing
# ══════════════════════════════════════════════════════════════════════════════

# Reduced 9-state set. Windows whose foot GRF pattern maps to a state outside
# this dict are discarded during collection.
#
# Excluded states and why:
#   Single-leg  (1000, 0100, 0010, 0001) — too brief in normal locomotion,
#               only relevant for explicit fall-recovery collection
#   No contact  (0000) — flight phase, only add if collecting bound explicitly
#   Lateral     (1010, 0101) — lateral balance; add back if you collect it
VALID_CONTACT_STATES: Dict[str, Tuple[int, int, int, int]] = {
    # ── Full support ────────────────────────────────────────────────────────
    "FULL":       (1, 1, 1, 1),   # Standing, slow crawl, 4-leg balance

    # ── Three-leg support (one leg in swing) ────────────────────────────────
    "SWING_FL":   (0, 1, 1, 1),   # Front-left  swing
    "SWING_FR":   (1, 0, 1, 1),   # Front-right swing
    "SWING_HL":   (1, 1, 0, 1),   # Hind-left   swing
    "SWING_HR":   (1, 1, 1, 0),   # Hind-right  swing

    # ── Two-leg support, diagonal pairs ─────────────────────────────────────
    "DIAG_FL_HR": (1, 0, 0, 1),   # Trot phase 1 — most common trot state
    "DIAG_FR_HL": (0, 1, 1, 0),   # Trot phase 2 — most common trot state

    # ── Two-leg support, sagittal pairs ─────────────────────────────────────
    "FRONT_PAIR": (1, 1, 0, 0),   # Pace (front legs), bound front landing
    "HIND_PAIR":  (0, 0, 1, 1),   # Pace (hind legs),  bound rear  landing
}

# Integer index for each state — used as the classification label
CONTACT_STATE_TO_IDX: Dict[str, int] = {
    k: i for i, k in enumerate(VALID_CONTACT_STATES)
}
IDX_TO_CONTACT_STATE: Dict[int, str] = {
    v: k for k, v in CONTACT_STATE_TO_IDX.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Window:
    """
    One training sample: a contiguous slice of W timesteps from a single episode.
    Labels are taken at the LAST timestep of the window (the state being predicted).

    proprioceptive_data — shape (W, 42):
        Channels per timestep:
            [0 :12]  joint positions    (12 joints)
            [12:24]  joint velocities   (12 joints)
            [24:36]  joint torques      (12 joints)
            [36:39]  IMU accelerometer  (x, y, z)
            [39:42]  IMU gyroscope      (x, y, z)
    """
    # ── Inputs ──────────────────────────────────────────────────────────────
    proprioceptive_data: np.ndarray     # (W, 42)  float32

    # ── Labels ──────────────────────────────────────────────────────────────
    contact_state:  str                 # Key from VALID_CONTACT_STATES
    grf_per_foot:   np.ndarray          # (4,)  [FL, FR, HL, HR]  Newtons  float32
    external_force: np.ndarray          # (3,)  [fx, fy, fz]      Newtons  float32

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
        9 contact states × 2 perturbation states × 3 terrains × 6 gait types
        = 324 — of which ~80–100 will be physically reachable.
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
    """

    def __init__(
        self,
        window_size:                  int   = 30,    # timesteps @ 50 Hz = 600 ms
        bucket_capacity:              int   = 5_000, # max windows stored per bucket
        contact_force_threshold:      float = 5.0,   # [N] min per-foot GRF = contact
        perturbation_force_threshold: float = 5.0,   # [N] min |F_ext| = perturbed
        min_perturbation_ratio:       float = 0.25,  # at least 25 % must be perturbed
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

    # ══════════════════════════════════════════════════════════════════════════
    # LABEL DERIVATION  (private helpers)
    # ══════════════════════════════════════════════════════════════════════════

    def _derive_contact_state(self, grf_per_foot: np.ndarray) -> Optional[str]:
        """
        Convert per-foot GRF magnitudes to a contact state string.

        Uses a force threshold rather than MuJoCo's binary solver flag to
        avoid contact chatter at touchdown / liftoff boundaries.

        IMPORTANT: grf_per_foot must be decimated cleanly from the MuJoCo
        internal step — take the value AT the 50 Hz control timestep,
        do not average over the inter-step interval.

        Returns None if the resulting binary pattern is not in the reduced
        set — the caller discards those windows.
        """
        binary = tuple(
            1 if float(grf_per_foot[i]) > self.contact_force_threshold else 0
            for i in range(4)   # order: FL, FR, HL, HR
        )
        for state_name, state_tuple in VALID_CONTACT_STATES.items():
            if state_tuple == binary:
                return state_name
        return None     # Not in reduced set

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
        proprioceptive_data: np.ndarray,    # (T, 42)
        grf_per_foot:        np.ndarray,    # (T, 4)
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
            Unique string identifier, e.g. "trot_flat_042".

        proprioceptive_data
            Full-episode sensor array, shape (T, 42).
            Channels: joint_pos[12] | joint_vel[12] | joint_torque[12]
                      | imu_acc[3]  | imu_gyro[3]

        grf_per_foot
            Per-foot GRF magnitudes [N], shape (T, 4), order: FL FR HL HR.
            Must be at the 50 Hz control rate — same rate as proprioceptive_data.

        external_force
            External force applied to the robot base [N], shape (T, 3), xyz.

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
        assert grf_per_foot.shape   == (T, 4), "grf_per_foot must be shape (T, 4)"
        assert external_force.shape == (T, 3), "external_force must be shape (T, 3)"

        n_added    = 0
        n_discarded = 0

        for start in range(0, T - self.window_size + 1, stride):
            end = start + self.window_size

            self.total_windows_seen += 1

            # Labels are derived at the LAST timestep of the window
            label_grf   = grf_per_foot[end - 1]    # (4,)
            label_ext_f = external_force[end - 1]   # (3,)

            contact_state = self._derive_contact_state(label_grf)

            # Discard windows whose contact pattern is outside the reduced set
            if contact_state is None:
                n_discarded += 1
                continue

            perturb_active = self._is_perturbation_active(label_ext_f)

            window = Window(
                proprioceptive_data = proprioceptive_data[start:end].astype(np.float32).copy(),
                contact_state       = contact_state,
                grf_per_foot        = label_grf.astype(np.float32).copy(),
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

        grf_total = float(np.sum(window.grf_per_foot))

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
            print(f"    {state:<14}  {count:>7,}  ({pct:5.1f}%)  {bar}")

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
        X            : (N, W, 42)  proprioceptive windows  float32
        y_contact    : (N,)        contact state index      int64
        y_grf        : (N, 4)      per-foot GRF [N]         float32
        y_ext_force  : (N, 3)      external base force [N]  float32
        """
        X, y_contact, y_grf, y_ext_force = [], [], [], []

        for key, windows in self.buckets.items():
            pool = windows if max_per_bucket is None else windows[:max_per_bucket]
            for w in pool:
                X.append(w.proprioceptive_data)
                y_contact.append(CONTACT_STATE_TO_IDX[w.contact_state])
                y_grf.append(w.grf_per_foot)
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


# ══════════════════════════════════════════════════════════════════════════════
# QUICK USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  DatasetBucketSystem — Minimal Usage Example\n")

    rng = np.random.default_rng(0)

    # 1. Instantiate — no pilot collection needed, no GRF thresholds to set
    bucket_sys = DatasetBucketSystem(
        window_size                  = 30,      # 30 steps @ 50 Hz = 600 ms
        bucket_capacity              = 5_000,
        contact_force_threshold      = 5.0,     # N — below this = swing
        perturbation_force_threshold = 5.0,     # N — above this = perturbed
        min_perturbation_ratio       = 0.25,    # at least 25 % must be perturbed
    )

    # 2. Simulate episodes and add them
    def make_fake_episode(
        T:               int,
        contact_pattern: Tuple[int, int, int, int],
        ext_force_mag:   float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synthetic episode data for demonstration only."""
        prop_data    = rng.standard_normal((T, 42)).astype(np.float32)
        grf_per_foot = np.zeros((T, 4), dtype=np.float32)
        for i, c in enumerate(contact_pattern):
            if c:
                grf_per_foot[:, i] = rng.normal(60.0, 20.0, T).clip(0)

        ext_force = np.zeros((T, 3), dtype=np.float32)
        # Perturbation event in the middle third of the episode
        t0, t1 = T // 3, 2 * T // 3
        ext_force[t0:t1] = rng.normal(0, ext_force_mag, (t1 - t0, 3))
        return prop_data, grf_per_foot, ext_force

    configs = [
        ("trot_flat_001",    GaitType.TROT,       TerrainType.FLAT,   (1, 0, 0, 1), 30.0),
        ("trot_rough_001",   GaitType.TROT,       TerrainType.ROUGH,  (0, 1, 1, 0), 50.0),
        ("crawl_flat_001",   GaitType.CRAWL,      TerrainType.FLAT,   (0, 1, 1, 1), 10.0),
        ("balance_flat_001", GaitType.BALANCE,    TerrainType.FLAT,   (1, 1, 1, 1), 80.0),
        ("balance_stair_01", GaitType.BALANCE,    TerrainType.STAIRS, (1, 1, 0, 1), 40.0),
        ("trans_flat_001",   GaitType.TRANSITION, TerrainType.FLAT,   (1, 1, 1, 0), 20.0),
    ]

    print("  Adding episodes:")
    for ep_id, gait, terrain, contact, ext_mag in configs:
        prop, grf, ext = make_fake_episode(T=3_000, contact_pattern=contact, ext_force_mag=ext_mag)
        result = bucket_sys.add_episode(
            episode_id          = ep_id,
            proprioceptive_data = prop,
            grf_per_foot        = grf,
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
