"""
Parquet storage for collected episodes.

A collection run writes three things into its run directory::

    <run_dir>/episodes/<episode_id>.parquet   one row per control step
    <run_dir>/episodes.parquet                one row per episode (metadata + split)
    <run_dir>/index.parquet                   the bucket-balanced label index

Episode files are written once, when the episode closes, and never rewritten.
The two small tables are rewritten after each episode so a run that is killed
mid-collection still leaves a consistent dataset behind.

The label index is what a training sampler reads: each row names an
``(episode_id, t)`` pair — one *labelled timestep* — plus the bucket it was
balanced into. It carries no window length: the sampler picks ``W`` and
materializes a window by slicing ``[t - W + 1 … t]`` out of the episode file,
skipping rows with ``t < W - 1``. The same collected run therefore serves any
window length.

``pyarrow`` is imported lazily so the rest of the collection stack — schema,
bucketing, tests — still imports in an environment without it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from mpx.utils.dataset_collection.dataset_schema import (
    EPISODE_COLUMNS,
    EPISODE_SCHEMA_VERSION,
    Column,
    EpisodeMetadata,
    EpisodeRecord,
)

EPISODES_DIRNAME = "episodes"
EPISODE_TABLE_FILENAME = "episodes.parquet"
INDEX_FILENAME = "index.parquet"
BALANCED_INDEX_FILENAME = "index_balanced.parquet"

_PYARROW_HINT = (
    "Parquet storage needs pyarrow. Install it into the project environment "
    "with `pixi add pyarrow`."
)


def _pyarrow():
    """Import pyarrow, raising a message that says how to install it."""
    try:
        import pyarrow as pa  # noqa: PLC0415  (deliberately lazy)
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(f"{_PYARROW_HINT} ({exc})") from exc
    return pa, pq


def pyarrow_available() -> bool:
    """True when parquet storage can be used in this environment."""
    try:
        _pyarrow()
    except ImportError:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

def _arrow_field(pa, column: Column):
    """Arrow field for one episode column (fixed-size list when width > 1)."""
    value_type = pa.from_numpy_dtype(np.dtype(column.dtype))
    arrow_type = (
        value_type if column.width == 1 else pa.list_(value_type, column.width)
    )
    return pa.field(
        column.name,
        arrow_type,
        nullable=False,
        metadata={
            b"role": column.role.encode(),
            b"unit": column.unit.encode(),
            b"frame": column.frame.encode(),
            b"doc": column.doc.encode(),
        },
    )


def episode_arrow_schema(metadata: Mapping[str, Any] | None = None):
    """Arrow schema of the per-timestep episode table."""
    pa, _ = _pyarrow()
    key_value = {
        b"mpx_schema_version": str(EPISODE_SCHEMA_VERSION).encode(),
    }
    if metadata:
        key_value[b"mpx_episode"] = json.dumps(metadata, default=str).encode()
    return pa.schema(
        [_arrow_field(pa, column) for column in EPISODE_COLUMNS],
        metadata=key_value,
    )


def _episode_arrays_to_arrow(pa, arrays: Mapping[str, np.ndarray]) -> List[Any]:
    """Convert the column dict of one episode into Arrow arrays."""
    columns: List[Any] = []
    for column in EPISODE_COLUMNS:
        values = np.ascontiguousarray(
            np.asarray(arrays[column.name], dtype=column.dtype)
        )
        if column.width == 1:
            columns.append(pa.array(values.reshape(-1)))
        else:
            flat = pa.array(values.reshape(-1))
            columns.append(pa.FixedSizeListArray.from_arrays(flat, column.width))
    return columns


def index_arrow_schema():
    """
    Arrow schema of the bucket-balanced label index.

    ``t`` is the labelled timestep: the step the row's contact state, GRF and
    perturbation flag describe. It is window-length agnostic on purpose.

    There is no ``split`` column: train/val/test lives in ``datasets/manifest.json``
    and is joined on ``randomization_group_id``, so a group cannot be split.
    """
    pa, _ = _pyarrow()
    return pa.schema(
        [
            pa.field("episode_id", pa.string(), nullable=False),
            pa.field("t", pa.int32(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("randomization_group_id", pa.string(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("terminate_reason", pa.string(), nullable=False),
            # True once the robot became unrecoverable in a failure episode.
            pa.field("post_failure", pa.bool_(), nullable=False),
            # ANY foot disagrees with the plan. Dominated by touchdown/liftoff
            # timing jitter: 88% of mismatched frames sit within two control
            # steps of a contact transition, so this alone is not a slip subset.
            pa.field("schedule_mismatch", pa.bool_(), nullable=False),
            # Inside a run of >= 3 consecutive mismatched frames — the part that
            # is a real disagreement rather than jitter. This is the benchmark
            # subset; 33% of mismatched frames on the audited run.
            pa.field("schedule_mismatch_sustained", pa.bool_(), nullable=False),
            # Which feet disagreed, "0100"-style in FL FR RL RR order.
            pa.field("schedule_mismatch_bits", pa.string(), nullable=False),
            # Usable for training: operating_regime in (nominal, degraded).
            pa.field("valid", pa.bool_(), nullable=False),
            # How degraded locomotion was: nominal | degraded | severe | failed.
            # Copied from the episode file so a sampler can filter or stratify
            # on it without opening one.
            pa.field("operating_regime", pa.string(), nullable=False),
            pa.field("base_height_terrain", pa.float32(), nullable=False),
            pa.field("base_tilt_deg", pa.float32(), nullable=False),
            pa.field("non_foot_contact_n", pa.float32(), nullable=False),
            # Rows in the episode this sample belongs to. With ``t`` it decides
            # window validity for ANY W, which window_valid_w10 cannot.
            pa.field("episode_n_steps", pa.int32(), nullable=False),
            # A 10-step causal window ending here lies inside this episode.
            pa.field("window_valid_w10", pa.bool_(), nullable=False),
            pa.field("terrain", pa.string(), nullable=False),
            pa.field("gait", pa.string(), nullable=False),
            pa.field("contact_state", pa.string(), nullable=False),
            pa.field("contact_bits", pa.string(), nullable=False),
            pa.field("rare_contact", pa.bool_(), nullable=False),
            pa.field("perturbation_active", pa.bool_(), nullable=False),
            # "none" | "small" | "large", binned on |F_ext| / body weight.
            pa.field("perturbation_level", pa.string(), nullable=False),
            # Command regime: stopped | slow | medium | fast | reverse | turning.
            pa.field("speed_bin", pa.string(), nullable=False),
            pa.field("bucket_key", pa.string(), nullable=False),
            pa.field("grf_total_n", pa.float32(), nullable=False),
            # Largest single-foot share of the total load, in [0.25, 1.0].
            # Total load is pinned near body weight by statics; the SHARE is what
            # varies and what the GRF head has to discriminate.
            pa.field("grf_load_share_max", pa.float32(), nullable=False),
            # max_i |f_xy,i| / f_z,i over loaded feet: friction utilisation.
            pa.field("grf_tangential_ratio_max", pa.float32(), nullable=False),
            pa.field("external_force_n", pa.float32(), nullable=False),
        ],
        metadata={b"mpx_schema_version": str(EPISODE_SCHEMA_VERSION).encode()},
    )


def episode_table_arrow_schema():
    """Arrow schema of the one-row-per-episode metadata table."""
    pa, _ = _pyarrow()
    return pa.schema(
        [
            pa.field("episode_id", pa.string(), nullable=False),
            pa.field("run_id", pa.string()),
            pa.field("robot", pa.string()),
            pa.field("scene", pa.string()),
            pa.field("terrain", pa.string()),
            pa.field("gait", pa.string()),
            pa.field("mode", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("ended_at", pa.string()),
            pa.field("control_hz", pa.float32()),
            pa.field("sim_hz", pa.float32()),
            pa.field("substeps_per_control", pa.int32()),
            pa.field("n_steps", pa.int32()),
            pa.field("duration_s", pa.float32()),
            pa.field("friction", pa.float32()),
            pa.field("payload_kg", pa.float32()),
            pa.field("body_weight_n", pa.float32()),
            pa.field("frac_nominal", pa.float32()),
            pa.field("terminate_by", pa.string()),
            pa.field("terminate_reason", pa.string()),
            # Cached from the manifest; the manifest is the authority.
            pa.field("split_assigned", pa.string()),
            pa.field("seed", pa.int64()),
            pa.field("randomization_group_id", pa.string()),
            # Sampled reset knobs vary per profile, so they travel as JSON text
            # rather than forcing a fixed column per knob.
            pa.field("reset_randomization", pa.string()),
        ],
        metadata={b"mpx_schema_version": str(EPISODE_SCHEMA_VERSION).encode()},
    )


# ══════════════════════════════════════════════════════════════════════════════
# WRITER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeStore:
    """
    Writes and reads one run directory's parquet files.

    Parameters
    ----------
    run_dir
        Directory that holds ``episodes/``, ``episodes.parquet`` and
        ``index.parquet``. Created on construction.
    compression
        Parquet codec for episode files. ``zstd`` roughly halves the file size
        of float sensor traces at negligible write cost.
    """

    run_dir: Path
    compression: str = "zstd"

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── paths ────────────────────────────────────────────────────────────────

    @property
    def episodes_dir(self) -> Path:
        return self.run_dir / EPISODES_DIRNAME

    @property
    def episode_table_path(self) -> Path:
        return self.run_dir / EPISODE_TABLE_FILENAME

    @property
    def index_path(self) -> Path:
        return self.run_dir / INDEX_FILENAME

    @property
    def balanced_index_path(self) -> Path:
        return self.run_dir / BALANCED_INDEX_FILENAME

    def episode_path(self, episode_id: str) -> Path:
        return self.episodes_dir / f"{episode_id}.parquet"

    # ── writing ──────────────────────────────────────────────────────────────

    def write_episode(self, record: EpisodeRecord) -> Path:
        """Write one episode's per-timestep table. Returns the file path."""
        pa, pq = _pyarrow()
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        path = self.episode_path(record.metadata.episode_id)

        schema = episode_arrow_schema(record.metadata.to_row())
        table = pa.Table.from_arrays(
            _episode_arrays_to_arrow(pa, record.arrays), schema=schema
        )
        self._write_table_atomic(pq, table, path)
        return path

    def write_episode_table(self, episodes: Sequence[EpisodeMetadata]) -> Path:
        """Rewrite the one-row-per-episode metadata table."""
        pa, pq = _pyarrow()
        schema = episode_table_arrow_schema()
        rows = []
        for metadata in episodes:
            row = metadata.to_row()
            row["reset_randomization"] = json.dumps(
                row.get("reset_randomization") or {}, sort_keys=True
            )
            rows.append(row)
        table = pa.Table.from_pylist(rows, schema=schema)
        self._write_table_atomic(pq, table, self.episode_table_path)
        return self.episode_table_path

    def write_index(self, rows: Iterable[Mapping[str, Any]]) -> Path:
        """Rewrite the bucket-balanced label index."""
        pa, pq = _pyarrow()
        schema = index_arrow_schema()
        table = pa.Table.from_pylist(list(rows), schema=schema)
        self._write_table_atomic(pq, table, self.index_path)
        return self.index_path

    def write_balanced_index(self, rows: Iterable[Mapping[str, Any]]) -> Path:
        """
        Write ``index_balanced.parquet``: ``(episode_id, t, split, weight)``.

        The materialised class-balanced view. It carries weights, not a subset,
        so the complete index stays the ground truth for "what rows exist".

        ``split`` travels with the weight because the weight is only meaningful
        inside the weighted splits: outside them it is exactly 1.0, and a reader
        that cannot see the split cannot tell a real weight from that default.
        """
        pa, pq = _pyarrow()
        schema = pa.schema(
            [
                pa.field("episode_id", pa.string(), nullable=False),
                pa.field("t", pa.int32(), nullable=False),
                pa.field("split", pa.string(), nullable=False),
                pa.field("weight", pa.float32(), nullable=False),
            ],
            metadata={b"mpx_schema_version": str(EPISODE_SCHEMA_VERSION).encode()},
        )
        table = pa.Table.from_pylist(list(rows), schema=schema)
        self._write_table_atomic(pq, table, self.balanced_index_path)
        return self.balanced_index_path

    def _write_table_atomic(self, pq, table, path: Path) -> None:
        """Write a parquet file beside its destination, then rename over it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
            pq.write_table(table, temp_path, compression=self.compression)
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    # ── reading ──────────────────────────────────────────────────────────────

    @staticmethod
    def read_episode(path: str | Path) -> Dict[str, np.ndarray]:
        """
        Read one episode file back into the column dict used by the schema.

        Fixed-size list columns come back as ``(T, width)`` arrays.
        """
        _, pq = _pyarrow()
        table = pq.read_table(str(path))
        arrays: Dict[str, np.ndarray] = {}
        for column in EPISODE_COLUMNS:
            chunked = table.column(column.name)
            if column.width == 1:
                arrays[column.name] = np.asarray(
                    chunked.to_numpy(zero_copy_only=False), dtype=column.dtype
                )
            else:
                flat = np.asarray(
                    chunked.combine_chunks().flatten().to_numpy(zero_copy_only=False),
                    dtype=column.dtype,
                )
                arrays[column.name] = flat.reshape(-1, column.width)
        return arrays

    @staticmethod
    def read_episode_metadata(path: str | Path) -> Dict[str, Any]:
        """Read the episode metadata stored in a file's parquet key/value block."""
        _, pq = _pyarrow()
        key_value = pq.read_schema(str(path)).metadata or {}
        raw = key_value.get(b"mpx_episode")
        return json.loads(raw.decode("utf-8")) if raw else {}

    def read_index(self) -> List[Dict[str, Any]]:
        """Read the label index as a list of row dicts (empty when absent)."""
        _, pq = _pyarrow()
        if not self.index_path.exists():
            return []
        return pq.read_table(str(self.index_path)).to_pylist()

    def read_episode_table(self) -> List[Dict[str, Any]]:
        """Read the per-episode metadata table (empty when absent)."""
        _, pq = _pyarrow()
        if not self.episode_table_path.exists():
            return []
        rows = pq.read_table(str(self.episode_table_path)).to_pylist()
        for row in rows:
            raw = row.get("reset_randomization")
            row["reset_randomization"] = json.loads(raw) if raw else {}
        return rows

    def window(
        self,
        episode_id: str,
        t: int,
        window_size: int,
        *,
        columns: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Slice the window ending at label timestep ``t``: ``[t - W + 1 … t]``.

        ``window_size`` is the caller's choice — it is not stored with the
        dataset. This is the reference implementation of the sampler's read
        path: it loads the episode file and slices it. A sampler that touches
        the same episode repeatedly should cache the episode arrays rather than
        call this per window.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        start = int(t) - int(window_size) + 1
        if start < 0:
            raise ValueError(
                f"Window of size {window_size} ending at t={t} starts before "
                f"the episode begins"
            )
        arrays = self.read_episode(self.episode_path(episode_id))
        wanted = columns if columns is not None else list(arrays)
        return {name: arrays[name][start : int(t) + 1] for name in wanted}
