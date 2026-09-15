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

from mpx.utils.dataset_collection.contact_labeling import ContactLabelConfig


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


@dataclass
class DatasetBucketConfig:
    """Parameters for :class:`DatasetBucketSystem` (thresholds, capacity).

    There is deliberately no window length here. Collection stores whole
    trajectories and indexes *labelled timesteps*; the window a model sees is a
    training-time choice, made by the PyTorch dataset.
    """

    # Max stored samples per (contact, perturbation, terrain, gait) bucket.
    bucket_capacity: int = 5_000

    # Per-foot GRF magnitude [N] above which a foot counts as in contact.
    contact_force_threshold: float = 5.0

    # External base-force norm [N] above which a sample is perturbation-active.
    perturbation_force_threshold: float = 5.0

    # Target minimum fraction of stored samples with active perturbation.
    min_perturbation_ratio: float = 0.25


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

    # Close an episode when the navigator reaches its goal. Off during data
    # collection: goal_reached truncated v3 episodes to a 9 s median, far short
    # of the 30 s of steady state a temporal representation needs. With this
    # False the navigator resamples a goal and the episode continues.
    end_episode_on_goal: bool = False

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
    """Defaults for the balanced sample index and the episode-level split."""

    max_per_bucket: int | None = None

    # Train/val/test is assigned per episode, so overlapping windows cut from one
    # episode never straddle a split boundary.
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    split_seed: int = 0

    shuffle_seed: int = 42

    # Diagnostics in :meth:`DatasetBucketSystem.print_summary`.
    grf_diversity_warn_std_n: float = 15.0
    underpopulated_bucket_fraction: float = 0.5


@dataclass
class DatasetCollectionConfig:
    """Combined profile for bucket storage + sim episode recording."""

    # Set True to collect without passing ``--collect`` on the CLI.
    enabled: bool = False

    rates: SimRateConfig = field(default_factory=SimRateConfig)
    contact_labeling: ContactLabelConfig = field(default_factory=ContactLabelConfig)
    bucket: DatasetBucketConfig = field(default_factory=DatasetBucketConfig)
    episode: EpisodeCollectionConfig = field(default_factory=EpisodeCollectionConfig)
    export: DatasetExportConfig = field(default_factory=DatasetExportConfig)
    output: DatasetOutputConfig = field(default_factory=DatasetOutputConfig)


# Default profile used by simulators and examples.
dataset_collection_config = DatasetCollectionConfig()
