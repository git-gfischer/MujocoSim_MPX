"""
Dataset collection settings for proprioceptive bucket storage.

Collected data lands in one run directory per session::

    <output_root_dir>/<run_folder>/episodes/<episode_id>.parquet   per-timestep rows
    <output_root_dir>/<run_folder>/episodes.parquet                per-episode metadata
    <output_root_dir>/<run_folder>/index.parquet                   balanced label-timestep index

Typical use::

    from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config
    from mpx.utils.dataset_collection.episode_recorder import setup_sim_collection

    hooks = setup_sim_collection(
        True,
        gait_type=GaitType.TROT,
        scene="flat",
        sim_hz=200.0,
        robot="go2",
        cfg=dataset_collection_config,
    )

Tune ``DatasetCollectionConfig`` fields or construct a custom profile::

    custom = DatasetCollectionConfig(
        bucket=DatasetBucketConfig(bucket_capacity=10_000),
        episode=EpisodeCollectionConfig(episode_duration_s=90.0),
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config
from mpx.utils.dataset_collection.contact_labeling import ContactLabelConfig
from mpx.utils.dataset_collection.operating_regime import OperatingRegimeConfig


def default_dataset_output_root() -> str:
    """Return ``<repo_root>/datasets``, sibling to the ``mpx`` package folder."""
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "datasets")


@dataclass
class DatasetOutputConfig:
    """Filesystem layout for saved datasets."""

    # Root folder where run directories are created (default: ``<repo>/datasets``).
    output_root_dir: str = field(default_factory=default_dataset_output_root)

    # Subfolder name for each collection run (created automatically).
    # Placeholders: {prefix} {robot} {scene} {gait} {terrain} {timestamp}
    run_folder_pattern: str = "{prefix}_{robot}_{scene}_{gait}_{timestamp}"

    # Append a timestamp so consecutive runs do not overwrite each other.
    use_timestamp: bool = True

    # Write ``run_metadata.json`` in the run folder alongside the parquet files.
    write_metadata_json: bool = True

    # Refresh ``episodes.parquet`` + ``index.parquet`` after each completed episode
    # (episode tables are always written once, when the episode closes).
    save_after_each_episode: bool = True

    # Parquet codec for the per-timestep episode tables. "zstd" roughly halves
    # the size of float sensor traces; "snappy" is faster, "none" disables it.
    parquet_compression: str = "zstd"

    # Run ``tools/validate_run.py`` when the run closes. The audited run declared
    # ``clipping_audit.limit = 0.001``, reported three channels over it, and
    # shipped: the check was right and nothing ran it.
    validate_after_run: bool = True

    # Checks skipped by that automatic pass. Empty: the difficulty probe is the
    # gate that decides whether a folder is worth training on, and leaving it
    # SKIPPED — which is what every validation_report.json said — means the
    # decision is never made. It costs a few minutes at the end of a run, which
    # is nothing against the collection itself.
    validate_skip: Tuple[str, ...] = ()

    # Move a failing run to ``<output_root_dir>/_quarantine/<run>`` so nothing
    # downstream picks it up by globbing ``datasets/*/``. Moved, not deleted: a
    # failing folder is still evidence of what went wrong.
    quarantine_failed_runs: bool = True


@dataclass
class DatasetBucketConfig:
    """
    Every threshold :class:`DatasetBucketSystem` uses. One source, no duplicates.

    There is deliberately no window length here. Collection stores whole
    trajectories and indexes *labelled timesteps*; the window a model sees is a
    training-time choice, made by the PyTorch dataset.

    There is also deliberately **no contact threshold**. Contact labelling lives
    in :class:`~mpx.utils.dataset_collection.contact_labeling.ContactLabelConfig`
    and nowhere else; the bucket system reads the ``contact`` column and never
    re-derives it. A second threshold here was a third definition of the label
    that disagreed with the shipped one on 8.6% of rows.
    """

    # Reference population per bucket, used as the denominator of the
    # underpopulated-bucket diagnostic. NOT an eviction limit: buckets never
    # drop samples (see DatasetBucketSystem._bucket_add).
    bucket_capacity: int = 5_000

    # ── perturbation axis ────────────────────────────────────────────────────
    # The DETECTION FLOOR is a fraction of body weight, not an absolute, because
    # payload is randomized 0.5-5.0 kg: the old 5 N threshold was 2.8% of the
    # Go2's 176 N, inside the noise of the external-force log itself.
    perturbation_small_bw_frac: float = 0.10     # 17.6 N on a bare Go2

    # The small/large boundary comes from the SAMPLER'S OWN RANGE, not from a
    # second body-weight fraction. A fixed 0.35 (61.6 N) sat above the sampler's
    # 50 N maximum, so `large` was structurally empty and BucketKey carried a
    # value that could never occur. Splitting the sampler's range puts the
    # boundary at 33.8 N against a measured perturbed p50 of 32.9 N, so both
    # bins populate — and they keep populating if the magnitude is retuned.
    perturbation_force_range_n: Tuple[float, float] = field(
        default_factory=lambda: tuple(ext_base_force_config.force_magnitude_range)
    )

    # Fallback used only when the sampler range is unknown (range set to None).
    perturbation_large_bw_frac: float = 0.35

    # Target minimum fraction of stored samples with active perturbation.
    min_perturbation_ratio: float = 0.25

    # Buckets smaller than this are excluded from the per-bucket perturbation
    # ratio report: a 6-sample bucket at 0% says nothing.
    min_bucket_samples_for_ratio_check: int = 100

    # ── class balancing (weights, never deletions) ───────────────────────────
    # 0 = natural distribution, 1 = fully equalised, 0.5 lifts the rare states
    # without pretending they are as common as a trot diagonal.
    class_balance_alpha: float = 0.5

    # Below this a bucket is reported as NOT COLLECTED and gets weight 0. One
    # sample cannot teach a class, it can only add gradient variance; the
    # audited run had a bucket with n = 1 carrying 57x the modal weight.
    min_bucket_samples_for_weighting: int = 50

    # Hard cap on max/min weight after weighting, before renormalisation.
    max_weight_ratio: float = 10.0

    # Splits the weights apply to. Reweighting val or test would make the
    # reported metric describe a distribution that does not exist.
    weighted_splits: Tuple[str, ...] = ("train",)

    # ── diversity diagnostics (see grf_diversity.py) ─────────────────────────
    # Spread of the per-foot LOAD SHARE, which is what total GRF cannot see:
    # statics pins the total near body weight whatever the distribution.
    load_share_iqr_warn: float = 0.02

    # Robust total-GRF spread, as a fraction of body weight so the threshold
    # does not move with the payload draw.
    total_grf_iqr_warn_bw_frac: float = 0.08

    # A bucket fed by fewer randomization draws than this is homogeneous however
    # many samples it holds.
    min_randomization_groups_per_bucket: int = 3

    # p99/median above this means the bucket's only spread is touchdown impacts.
    impact_dominated_p99_ratio: float = 2.5

    # Buckets below this are too small for any of the diversity statistics.
    diversity_min_bucket_samples: int = 100

    # ── population diagnostics ───────────────────────────────────────────────
    # Relative, not absolute: bucket_capacity as a denominator flags every
    # bucket on a 5-episode pilot and none on a 500-episode run. A bucket is
    # underpopulated when it holds less than this share of what an even split
    # over the active buckets would give it.
    underpopulated_share_of_expected: float = 0.25

    # ── schedule_mismatch decomposition ──────────────────────────────────────
    # Control steps from a contact transition within which a mismatch counts as
    # touchdown/liftoff jitter rather than a real disagreement.
    mismatch_edge_radius: int = 2

    # Consecutive mismatched frames that make a run "sustained".
    mismatch_sustained_min_run: int = 3


@dataclass
class SimRateConfig:
    """Physics and control rates for a collection run."""

    # Physics rate [Hz]. 500 Hz (2 ms) keeps the contact solver's 10 ms time
    # constant at 5 substeps, which is what stops the one-frame contact dropouts;
    # at 200 Hz the same time constant is only 2 substeps and the foot bounces.
    sim_hz: float = 500.0

    # Control and logging rate [Hz]. Unchanged — the labels are reduced from the
    # substeps between two control steps, not sampled at one instant.
    control_hz: float = 50.0

    @property
    def substeps_per_control(self) -> int:
        return max(1, int(round(self.sim_hz / self.control_hz)))


@dataclass
class EpisodeCollectionConfig:
    """On-the-fly episode buffering before routing into buckets."""

    # Control / label sample rate [Hz] (decimated from sim rate, e.g. 500 Hz → 50 Hz).
    control_hz: float = 50.0

    # How episode boundaries are decided:
    #   "event"          — the simulator closes each episode on a task event
    #                      (locomotion: goal reached, balance: desired pose lost).
    #                      ``episode_duration_s`` then acts only as a safety cap.
    #   "fixed_duration" — close every episode after ``episode_duration_s``.
    episode_mode: str = "event"

    # Fixed-duration mode: exact episode length.
    # Event mode: hard cap that force-closes an episode that never fires an event.
    episode_duration_s: float = 60.0

    # Shortest episode accepted into the buckets. Shorter ones are dropped as
    # too short to be worth a trajectory. This is the only length floor: it must
    # be at least as long as the longest window any downstream dataset will cut.
    min_episode_duration_s: float = 1.0

    # Stride between indexed label timesteps inside
    # :meth:`DatasetBucketSystem.add_episode`. 1 indexes every step.
    label_stride: int = 1

    # Keep episodes that ended in a fall or a lost pose (``terminate_by="failure"``).
    # The steps leading into a failure are where contact/GRF estimates break down,
    # so they are collected by default. A manual respawn is never stored.
    store_failed_episodes: bool = True

    # Close an episode when the navigator reaches its goal, and resample the
    # domain randomization for the next one.
    #
    # A goal is the natural episode boundary under --nav: it is a task success,
    # so it puts real `terminate_by="success"` episodes in the data, and closing
    # there is what lets the randomization advance. Without it a robot that keeps
    # reaching goals holds one friction and payload for the whole episode, so a
    # long survival produces a lot of rows under a single parameter set.
    end_episode_on_goal: bool = True

    # ...but not at EVERY goal. v3 closed on each one and left a 9 s median
    # episode, far short of the steady state a temporal representation needs.
    # Goals reached before this are ignored for episode purposes — the navigator
    # just gets a new goal — and the first goal after it closes the episode.
    # That buys randomization diversity without collapsing episode length.
    goal_closes_episode_after_s: float = 30.0

    # Resample every domain-randomization knob at each episode boundary, not only
    # at a respawn. v3 reused one parameter set across up to 12 consecutive
    # episodes, which made near-duplicate episodes land in different splits.
    randomize_per_episode: bool = True

    # Drive collection from a segmented velocity command instead of goal
    # following. Goal following never commands a yaw rate and never reverses:
    # the audited v4 run had commanded yaw identically zero for all 23,994 rows,
    # so yaw-invariance could not be tested at all. See
    # mpx/utils/simulation_utils/velocity_command.py.
    segmented_commands: bool = True


@dataclass
class DatasetExportConfig:
    """
    Defaults for the balanced sample index and the episode-level split.

    Diagnostic thresholds are NOT here — they live once, in
    :class:`DatasetBucketConfig`. Two of them used to be defined both here and
    as literals inside ``print_summary``; they agreed until someone tuned one.
    """

    max_per_bucket: int | None = None

    # Train/val/test is assigned per episode, so overlapping windows cut from one
    # episode never straddle a split boundary.
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    split_seed: int = 0

    shuffle_seed: int = 42


@dataclass
class DatasetCollectionConfig:
    """Combined profile for bucket storage + sim episode recording."""

    # Set True to collect without passing ``--collect`` on the CLI.
    enabled: bool = False

    rates: SimRateConfig = field(default_factory=SimRateConfig)
    contact_labeling: ContactLabelConfig = field(default_factory=ContactLabelConfig)
    operating_regime: OperatingRegimeConfig = field(
        default_factory=OperatingRegimeConfig
    )
    bucket: DatasetBucketConfig = field(default_factory=DatasetBucketConfig)
    episode: EpisodeCollectionConfig = field(default_factory=EpisodeCollectionConfig)
    export: DatasetExportConfig = field(default_factory=DatasetExportConfig)
    output: DatasetOutputConfig = field(default_factory=DatasetOutputConfig)


# Default profile used by simulators and examples.
dataset_collection_config = DatasetCollectionConfig()

# The bucket thresholds on their own, for the bucket system and the post-hoc
# diagnostics, neither of which needs the rest of the collection profile.
dataset_bucket_config = dataset_collection_config.bucket
