"""
Dataset collection settings for proprioceptive bucket storage.

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

    # Default ``.npz`` filename inside the run folder.
    filename_pattern: str = "{prefix}_{robot}_{scene}_{gait}_{terrain}_{timestamp}.npz"

    # Append a timestamp so consecutive runs do not overwrite each other.
    use_timestamp: bool = True

    # Write a small ``.json`` metadata file next to the ``.npz``.
    write_metadata_json: bool = True

    # Rewrite the ``.npz`` on disk after each completed episode (safe if sim is killed).
    save_after_each_episode: bool = True


@dataclass
class DatasetBucketConfig:
    """Parameters for :class:`DatasetBucketSystem` (windowing, thresholds, capacity)."""

    # Sliding-window length [control steps]. 30 @ 50 Hz ≈ 600 ms.
    window_size: int = 30

    # Max stored windows per (contact, perturbation, terrain, gait) bucket.
    bucket_capacity: int = 5_000

    # Per-foot GRF magnitude [N] above which a foot counts as in contact.
    contact_force_threshold: float = 5.0

    # External base-force norm [N] above which a window is perturbation-active.
    perturbation_force_threshold: float = 5.0

    # Target minimum fraction of stored windows with active perturbation.
    min_perturbation_ratio: float = 0.25


@dataclass
class EpisodeCollectionConfig:
    """On-the-fly episode buffering before routing into buckets."""

    # Control / label sample rate [Hz] (decimated from sim rate, e.g. 200 Hz → 50 Hz).
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

    # Shortest episode accepted into the buckets. Shorter ones are dropped, since
    # an episode must span at least one full window to yield a training sample.
    min_episode_duration_s: float = 1.0

    # Stride between sliding windows inside :meth:`DatasetBucketSystem.add_episode`.
    window_stride: int = 1


@dataclass
class DatasetExportConfig:
    """Defaults for ``export`` / ``export_split`` / ``save_npz``."""

    max_per_bucket: int | None = None
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    shuffle_seed: int = 42

    # Diagnostics in :meth:`DatasetBucketSystem.print_summary`.
    grf_diversity_warn_std_n: float = 15.0
    underpopulated_bucket_fraction: float = 0.5


@dataclass
class DatasetCollectionConfig:
    """Combined profile for bucket storage + sim episode recording."""

    # Set True to collect without passing ``--collect`` on the CLI.
    enabled: bool = False

    bucket: DatasetBucketConfig = field(default_factory=DatasetBucketConfig)
    episode: EpisodeCollectionConfig = field(default_factory=EpisodeCollectionConfig)
    export: DatasetExportConfig = field(default_factory=DatasetExportConfig)
    output: DatasetOutputConfig = field(default_factory=DatasetOutputConfig)

    # Deprecated: prefer ``output.filename_pattern`` + :func:`resolve_dataset_output_path`.
    default_output_pattern: str = "dataset_{prefix}_{robot}_{scene}.npz"


# Default profile used by simulators and examples.
dataset_collection_config = DatasetCollectionConfig()
