"""
Post-hoc diversity diagnostics over a written run.

Deliberately reads ``index.parquet`` rather than the in-memory buckets, so the
statistic can be changed without recollecting. The bucket system only counts.

Why these statistics and not the old one
----------------------------------------
The v4.1 diagnostic was ``std`` of total GRF per bucket, computed at collection
time. It measured the wrong thing and could not be fixed after the fact.

*Total magnitude is nearly invariant by construction.* In support, ``sum |f_i|``
equals body weight plus payload whatever the distribution across feet: a 90/10
front-rear split and a 50/50 split give the same number. So the diagnostic fired
hardest on ``FULL`` support — exactly where low total variance is physics rather
than a defect — and was blind to the quantity the GRF head actually has to
learn. ``load_share_iqr`` measures that quantity directly.

*``std`` is not robust here.* The distribution has ``p99/p50 = 1.87`` and a
732 N maximum from touchdown transients, so ``std`` mostly reports impacts.
``total_grf_iqr_bw`` is an interquartile range, and it is normalised by body
weight so the threshold does not move with the payload draw — the old 15.0 N was
an absolute number on a robot whose payload is randomized 0.52-4.98 kg.

*Sample count hides homogeneity.* A bucket of 9,627 samples drawn from one
command and one friction draw is homogeneous however full it looks.
``n_randomization_groups`` catches that; nothing else does.

Usage::

    python -m mpx.utils.dataset_collection.grf_diversity datasets/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from mpx.config.sim_config.config_dataset_bucket import (
    DatasetBucketConfig,
    dataset_bucket_config,
)

# Columns the report needs. Reading only these keeps a 60 MB index cheap.
INDEX_COLUMNS: Sequence[str] = (
    "bucket_key",
    "episode_id",
    "randomization_group_id",
    "valid",
    "perturbation_active",
    "grf_total_n",
    "grf_load_share_max",
    "grf_tangential_ratio_max",
)


def _iqr(values: np.ndarray) -> float:
    """Interquartile range; 0.0 when there are too few samples to have one."""
    if values.size < 4:
        return 0.0
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def _read_index(run_dir: Path) -> Dict[str, np.ndarray]:
    """Read the label index into a column dict, valid rows only."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    path = run_dir / "index.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No index.parquet in {run_dir}")

    available = set(pq.read_schema(str(path)).names)
    missing = [c for c in INDEX_COLUMNS if c not in available]
    if missing:
        raise ValueError(
            f"{path} predates the diversity diagnostics: missing {missing}. "
            f"Re-collect, or read it with the version of the code that wrote it."
        )

    table = pq.read_table(str(path), columns=list(INDEX_COLUMNS))
    columns = {
        name: np.asarray(table.column(name).to_numpy(zero_copy_only=False))
        for name in INDEX_COLUMNS
    }
    keep = columns["valid"].astype(bool)
    return {name: values[keep] for name, values in columns.items()}


def _body_weight(run_dir: Path, fallback: float = 176.0) -> float:
    """Body weight [N] for this run, from the episode table or the metadata."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    episodes = run_dir / "episodes.parquet"
    if episodes.is_file() and "body_weight_n" in set(
        pq.read_schema(str(episodes)).names
    ):
        values = np.asarray(
            pq.read_table(str(episodes), columns=["body_weight_n"])
            .column("body_weight_n")
            .to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        values = values[values > 0]
        if values.size:
            return float(np.median(values))
    return float(fallback)


def bucket_diversity(
    run_dir: str | Path,
    body_weight_n: float | None = None,
) -> List[Dict[str, Any]]:
    """
    One row per bucket with the statistics that actually discriminate.

    Returns a list of dicts sorted by sample count, descending. Plain dicts
    rather than a DataFrame: pandas is not a dependency of this project, and
    every consumer here either prints the rows or filters them.
    """
    run_dir = Path(run_dir)
    index = _read_index(run_dir)
    weight = (
        _body_weight(run_dir) if body_weight_n is None else float(body_weight_n)
    )

    keys = index["bucket_key"].astype(str)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1

    rows: List[Dict[str, Any]] = []
    for part in np.split(order, boundaries):
        if part.size == 0:
            continue
        total = index["grf_total_n"][part].astype(np.float64)
        share = index["grf_load_share_max"][part].astype(np.float64)
        tangential = index["grf_tangential_ratio_max"][part].astype(np.float64)
        median_total = float(np.median(total))
        rows.append(
            {
                "bucket_key": str(index["bucket_key"][part[0]]),
                "n_samples": int(part.size),
                "n_episodes": int(np.unique(index["episode_id"][part]).size),
                "n_randomization_groups": int(
                    np.unique(index["randomization_group_id"][part]).size
                ),
                "total_grf_median": median_total,
                "total_grf_iqr_bw": _iqr(total) / max(weight, 1e-6),
                "total_grf_p99_over_median": (
                    float(np.percentile(total, 99) / median_total)
                    if median_total > 1e-6 else 0.0
                ),
                "load_share_median": float(np.median(share)),
                "load_share_iqr": _iqr(share),
                "tangential_ratio_p95": float(np.percentile(tangential, 95)),
                "perturbation_ratio": float(
                    index["perturbation_active"][part].astype(bool).mean()
                ),
            }
        )
    rows.sort(key=lambda row: -row["n_samples"])
    return rows


def diversity_warnings(
    report: Sequence[Dict[str, Any]],
    config: DatasetBucketConfig | None = None,
) -> Dict[str, Any]:
    """Buckets that will not teach the GRF head anything, and why."""
    cfg = config if config is not None else dataset_bucket_config
    big = [r for r in report if r["n_samples"] >= cfg.diversity_min_bucket_samples]
    if not big:
        return {
            "buckets_examined": 0,
            "note": (
                f"no bucket reached {cfg.diversity_min_bucket_samples} samples — "
                f"collect more before reading a diversity number"
            ),
        }

    worst = min(big, key=lambda r: r["perturbation_ratio"])
    return {
        "buckets_examined": len(big),
        # The one that matters. Total load is pinned near body weight by
        # statics, so it cannot see whether the DISTRIBUTION across feet varies.
        "low_load_share_spread": [
            r["bucket_key"] for r in big
            if r["load_share_iqr"] < cfg.load_share_iqr_warn
        ],
        "low_total_grf_spread": [
            r["bucket_key"] for r in big
            if r["total_grf_iqr_bw"] < cfg.total_grf_iqr_warn_bw_frac
        ],
        # A bucket fed by one randomization draw is homogeneous however full it
        # looks; the sample count cannot show this.
        "single_condition": [
            r["bucket_key"] for r in big
            if r["n_randomization_groups"] < cfg.min_randomization_groups_per_bucket
        ],
        "spread_is_only_impacts": [
            r["bucket_key"] for r in big
            if r["total_grf_p99_over_median"] > cfg.impact_dominated_p99_ratio
        ],
        # The stated rationale for the perturbation axis is per-state coverage,
        # so check the worst bucket, not the aggregate. The audited run's global
        # 20.9% hid a per-state range of 9.4% to 66.7%.
        "worst_perturbation_ratio": float(worst["perturbation_ratio"]),
        "worst_perturbation_bucket": str(worst["bucket_key"]),
        "perturbation_ratio_target": float(cfg.min_perturbation_ratio),
    }


def format_report(
    report: Sequence[Dict[str, Any]],
    warnings: Dict[str, Any],
    limit: int = 20,
) -> str:
    """Human-readable table plus the warning lists."""
    lines = [
        f"{'bucket':<52} {'n':>7} {'grp':>4} {'shareIQR':>9} "
        f"{'iqr/bw':>7} {'p99/med':>8} {'pert':>6}",
    ]
    for row in list(report)[:limit]:
        lines.append(
            f"{row['bucket_key']:<52} {row['n_samples']:>7,} "
            f"{row['n_randomization_groups']:>4} {row['load_share_iqr']:>9.3f} "
            f"{row['total_grf_iqr_bw']:>7.3f} "
            f"{row['total_grf_p99_over_median']:>8.2f} "
            f"{row['perturbation_ratio']:>6.1%}"
        )
    if len(report) > limit:
        lines.append(f"... and {len(report) - limit} more buckets")

    lines.append("")
    for name in (
        "low_load_share_spread",
        "low_total_grf_spread",
        "single_condition",
        "spread_is_only_impacts",
    ):
        flagged = warnings.get(name) or []
        lines.append(f"{name}: {len(flagged)}")
        for key in flagged[:5]:
            lines.append(f"    {key}")
    if "worst_perturbation_ratio" in warnings:
        lines.append(
            f"worst perturbation coverage: "
            f"{warnings['worst_perturbation_ratio']:.1%} "
            f"(target {warnings['perturbation_ratio_target']:.0%}) in "
            f"{warnings['worst_perturbation_bucket']}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run folder to analyse")
    parser.add_argument("--body-weight-n", type=float, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = bucket_diversity(args.run, body_weight_n=args.body_weight_n)
    warnings = diversity_warnings(report)
    print(format_report(report, warnings, limit=args.limit))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"buckets": report, "warnings": warnings}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
