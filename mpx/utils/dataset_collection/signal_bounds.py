"""
Global signal bounds and stance values for Proprioceptive Image normalisation.

The PI encoder normalises every signal against ``s_min``/``s_max`` and centres it
on a ``stance_value``. If those bounds are derived per run folder from the data
in it, the same joint angle maps to a different pixel value in ``flat/trot`` than
in ``stairs/crawl``, and any cross-terrain comparison measures the normalisation
rather than the model. So the bounds live in **one** datasheet-derived file,
``datasets/signal_bounds.json``, committed to the repository and identical for
every run.

Each run embeds the file's SHA-256 and an inline copy in its
``run_metadata.json``. A loader that is handed two folders with different hashes
must refuse to mix them — :func:`assert_same_bounds` does that check.

Because the bounds are fixed rather than fitted, a signal *can* exceed them.
:func:`clipping_audit` measures how often that happens per channel; a run where
any channel clips more than 0.1% of the time is flagged, because clipped samples
saturate to ±1 and lose all internal structure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from mpx.utils.dataset_collection.dataset_schema import (
    FOOT_ORDER,
    JOINT_ORDER,
    N_FEET,
)

SIGNAL_BOUNDS_FILENAME = "signal_bounds.json"

# Axis names used by the 3-vector signals in the bounds file.
_AXES = ("x", "y", "z")


def default_signal_bounds_path() -> Path:
    """``<repo_root>/datasets/signal_bounds.json``."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "datasets" / SIGNAL_BOUNDS_FILENAME


def load_signal_bounds(path: str | Path | None = None) -> Dict[str, Any]:
    """Read the bounds file. Missing or malformed is an error, never a default."""
    bounds_path = Path(path) if path is not None else default_signal_bounds_path()
    if not bounds_path.is_file():
        raise FileNotFoundError(
            f"Global signal bounds not found at '{bounds_path}'. This file is "
            f"required: PI normalisation must not be fitted per run folder."
        )
    try:
        payload = json.loads(bounds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read signal bounds '{bounds_path}': {exc}") from exc
    if not isinstance(payload.get("signals"), dict):
        raise ValueError(f"Signal bounds '{bounds_path}' has no 'signals' object")
    return payload


def signal_bounds_sha256(path: str | Path | None = None) -> str:
    """SHA-256 of the bounds file exactly as it sits on disk."""
    bounds_path = Path(path) if path is not None else default_signal_bounds_path()
    return hashlib.sha256(bounds_path.read_bytes()).hexdigest()


def signal_bounds_version(bounds: Mapping[str, Any] | None = None) -> int:
    """
    Integer version of the bounds file.

    The hash alone tells you two runs disagree but not which is newer; the
    version makes a mixed-version dataset detectable rather than merely
    mismatched.
    """
    payload = bounds if bounds is not None else load_signal_bounds()
    return int(payload.get("signal_bounds_version", 1))


def assert_same_bounds(run_metadata: Sequence[Mapping[str, Any]]) -> str:
    """
    Check that every run was normalised against the same bounds file.

    Returns the shared hash; raises when two runs disagree, since mixing them
    would compare images built on different scales.
    """
    hashes = {str(m.get("signal_bounds_sha256", "")) for m in run_metadata}
    if len(hashes) != 1 or "" in hashes:
        raise ValueError(
            f"Runs were normalised against different signal bounds: {sorted(hashes)}. "
            f"Re-collect or re-normalise before mixing them."
        )
    return hashes.pop()


def _per_element_bounds(signal: Mapping[str, Any], width: int) -> np.ndarray:
    """
    Expand a signal's bounds dict into a ``(width, 2)`` low/high array.

    Per-joint bounds (``HAA``/``HFE``/``KFE``) tile across the four legs; per-axis
    bounds (``x``/``y``/``z``) tile across the four feet for the 12-vectors and
    apply once for the 3-vectors.
    """
    bounds = signal["bounds"]
    keys = list(bounds)

    if set(keys) >= set(JOINT_ORDER):
        per_leg = np.asarray([bounds[j] for j in JOINT_ORDER], dtype=np.float64)
        return np.tile(per_leg, (width // len(JOINT_ORDER), 1))

    if set(keys) >= set(_AXES):
        per_axis = np.asarray([bounds[a] for a in _AXES], dtype=np.float64)
        repeats = width // len(_AXES)
        return np.tile(per_axis, (repeats, 1))

    raise ValueError(f"Unrecognised bounds keys {keys}")


def _per_element_stance(signal: Mapping[str, Any], width: int) -> np.ndarray:
    """Expand a signal's ``stance_value`` the same way as its bounds."""
    stance = signal.get("stance_value")
    if not stance:
        return np.zeros(width, dtype=np.float64)
    keys = list(stance)
    if set(keys) >= set(JOINT_ORDER):
        per_leg = np.asarray([stance[j] for j in JOINT_ORDER], dtype=np.float64)
        return np.tile(per_leg, width // len(JOINT_ORDER))
    if set(keys) >= set(_AXES):
        per_axis = np.asarray([stance[a] for a in _AXES], dtype=np.float64)
        return np.tile(per_axis, width // len(_AXES))
    raise ValueError(f"Unrecognised stance keys {keys}")


def encoder_constraints(
    signal_name: str,
    width: int,
    bounds: Mapping[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    """
    ``constraints=[max, min]`` and ``stance_value`` for one PI sub-encoder.

    ``ImageEncoder`` takes them in that order; passing them explicitly is what
    keeps the encoder off its ``stance_value=0.0`` default, which is wrong for
    every joint-position channel and for ``imu_acc_body.z``.
    """
    payload = bounds if bounds is not None else load_signal_bounds()
    signal = payload["signals"][signal_name]
    limits = _per_element_bounds(signal, width)
    return {
        "constraints": np.stack([limits[:, 1], limits[:, 0]], axis=0),
        "stance_value": _per_element_stance(signal, width),
        "unit": signal.get("unit", ""),
    }


# Human-readable statement of what the audit counts, written into the metadata
# so nobody has to guess which denominator produced a number.
CLIPPING_DEFINITION = (
    "per_element: fraction of (row, element) pairs at or outside the bound; "
    "per_row_any: fraction of rows with at least one element outside"
)

# Gate applied to ``per_element``. Above this, enough samples saturate to +-1
# after normalisation that the channel loses internal structure.
CLIPPING_LIMIT = 0.001


def clipping_audit(
    arrays: Mapping[str, np.ndarray],
    bounds: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Per-channel clipping, counted two ways over **every** row given.

    The v4.0 audit reported a single fraction whose denominator was ambiguous —
    ``joint_pos`` came out at 2.05% and reconciled with neither definition,
    because it was computed on small per-channel subsamples. Both definitions are
    now reported explicitly, over all rows:

    ``per_element``
        fraction of (row, element) pairs outside the bound. This is the gated
        number: it is what decides how much of the encoded image saturates.
    ``per_row_any``
        fraction of rows with at least one element outside. Useful for deciding
        whether to drop rows, and always the larger of the two.

    Counts, not just fractions, travel with the result so episode-level audits
    can be merged into a run-level one without re-reading the data.
    """
    payload = bounds if bounds is not None else load_signal_bounds()
    per_element: Dict[str, float] = {}
    per_row_any: Dict[str, float] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for name, signal in payload["signals"].items():
        if name not in arrays:
            continue
        values = np.asarray(arrays[name], dtype=np.float64)
        values = values.reshape(values.shape[0], -1)
        if values.size == 0:
            per_element[name] = 0.0
            per_row_any[name] = 0.0
            counts[name] = {"rows": 0, "elements": 0, "out_elements": 0, "out_rows": 0}
            continue
        limits = _per_element_bounds(signal, values.shape[1])
        outside = (values < limits[:, 0]) | (values > limits[:, 1])
        n_rows, n_cols = values.shape
        counts[name] = {
            "rows": int(n_rows),
            "elements": int(n_rows * n_cols),
            "out_elements": int(outside.sum()),
            "out_rows": int(outside.any(axis=1).sum()),
        }
        per_element[name] = float(outside.mean())
        per_row_any[name] = float(outside.any(axis=1).mean())

    return {
        "definition": CLIPPING_DEFINITION,
        "limit": CLIPPING_LIMIT,
        "per_element": per_element,
        "per_row_any": per_row_any,
        "counts": counts,
    }


def merge_clipping_audits(audits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Combine per-episode audits into one run-level audit.

    Sums the counts and recomputes the fractions from them, rather than averaging
    or taking a worst case: a 61-step episode and a 3,000-step one must not carry
    equal weight in a number that is meant to describe the whole run.
    """
    totals: Dict[str, Dict[str, int]] = {}
    for audit in audits:
        for name, count in (audit.get("counts") or {}).items():
            entry = totals.setdefault(
                name, {"rows": 0, "elements": 0, "out_elements": 0, "out_rows": 0}
            )
            for key in entry:
                entry[key] += int(count.get(key, 0))

    per_element = {
        name: (c["out_elements"] / c["elements"]) if c["elements"] else 0.0
        for name, c in totals.items()
    }
    per_row_any = {
        name: (c["out_rows"] / c["rows"]) if c["rows"] else 0.0
        for name, c in totals.items()
    }
    return {
        "definition": CLIPPING_DEFINITION,
        "limit": CLIPPING_LIMIT,
        "per_element": per_element,
        "per_row_any": per_row_any,
        "counts": totals,
    }


def worst_clipping(audit: Mapping[str, Any]) -> tuple[str, float]:
    """Channel with the highest ``per_element`` fraction, and that fraction."""
    per_element = audit.get("per_element") or {}
    if not per_element:
        return ("", 0.0)
    name = max(per_element, key=lambda k: per_element[k])
    return (name, float(per_element[name]))


def describe_bounds(bounds: Mapping[str, Any] | None = None) -> str:
    """One line per signal, printed at the start of a collection run."""
    payload = bounds if bounds is not None else load_signal_bounds()
    lines = [
        f"signal bounds v{payload.get('schema_version')} "
        f"({payload.get('robot')}) — global, not fitted per run"
    ]
    for name, signal in payload["signals"].items():
        pairs = "  ".join(
            f"{key}=[{low:g},{high:g}]" for key, (low, high) in signal["bounds"].items()
        )
        lines.append(f"  {name:<22} {signal.get('unit',''):<8} {pairs}")
    lines.append(f"  feet={list(FOOT_ORDER)} joints={list(JOINT_ORDER)} n_feet={N_FEET}")
    return "\n".join(lines)
