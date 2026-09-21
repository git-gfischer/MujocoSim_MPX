"""
Operating regime: how degraded the robot's locomotion is, frame by frame.

Why this exists
---------------
``post_failure`` fired on **52 of 32,306 rows (0.16%)** of the audited run, while
crash episodes contributed **3,194 rows (9.9%)**. So 98.4% of the frames inside a
crashing episode were labelled as normal operation — including the belly-scraping
stumble where the robot is still trying to execute the gait.

The crash predicate was never meant to carry that load. ``crash_height_m = 0.15``
and ``crash_tilt_deg = 60`` are *terminate* conditions, and the robot is visibly
wrong for more than a second before they trip:

    steps before end   base height   tilt median   tilt p95
        250-1000          0.202 m        2.9         15.9
        120-250           0.220 m        3.4         22.2
         60-120           0.212 m        5.6         18.7
          30-60           0.194 m       12.9         24.9
          15-30           0.175 m       17.3         34.8
           5-15           0.164 m       17.7         44.4
            0-5           0.152 m       29.6         59.4

And it is not only crash episodes. Episode 00023 of that run terminates
``goal_reached`` with **76.2% of frames below 0.20 m** and a median height of
0.190 m — a full minute of walking in a collapsed posture, indistinguishable from
a healthy run by anything the dataset recorded.

Those frames are valuable: dragging feet, unplanned shin and belly contact, GRFs
the gait schedule never intended. That is exactly the regime a contact estimator
should be tested on. **They are kept, and labelled.**

The four signals
----------------
1. **Non-foot contact force.** The most direct of the four and the only one that
   needs no threshold tuning: a healthy quadruped has exactly zero force on any
   geom that is not a foot. Above the floor it forces at least ``severe``,
   because if the body is on the ground the posture thresholds are moot.
2. **Terrain-relative base height.**
3. **Tilt**, the angle between the base z-axis and gravity.
4. **Command tracking error**, ``‖realised - commanded‖`` planar velocity.

The regime is the **worst** of the four, then a dwell filter so a single noisy
frame cannot flip the annotation. The filter is causal, like the contact
debouncer: the annotation stays reproducible online and no future information
leaks backwards into a context column.

Thresholds are grounded in the measured non-crash distribution of the audited
run, not guessed:

    non-crash:  height p1 0.183  p5 0.195  median 0.249 | tilt p95  6.4  p99  9.2  max 17.3
    crash    :  height p1 0.149  p5 0.165  median 0.208 | tilt p95 24.5  p99 35.3  max 59.7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

# Worst-first ordering. Index into it to compare severity.
REGIME_ORDER: tuple[str, ...] = ("nominal", "degraded", "severe", "failed")
REGIME_INDEX: Dict[str, int] = {name: i for i, name in enumerate(REGIME_ORDER)}

# Width of the fixed-size unicode column the regime is stored in.
REGIME_DTYPE = np.dtype("U8")


@dataclass
class OperatingRegimeConfig:
    """
    Thresholds grounded in the measured non-crash distribution, not guessed.

    Height rails sit just below the non-crash p5 (0.195 m) and p1 (0.183 m);
    tilt rails sit above the non-crash p99 (9.2 deg) and max (17.3 deg). A
    nominal episode should therefore almost never leave ``nominal``.
    """

    # Terrain-relative base height [m]
    height_degraded_m: float = 0.195
    height_severe_m: float = 0.175
    height_failed_m: float = 0.150

    # Tilt = angle between the base z-axis and gravity [deg]
    tilt_degraded_deg: float = 12.0
    tilt_severe_deg: float = 25.0
    tilt_failed_deg: float = 45.0

    # Total normal force on any geom that is not one of the four feet [N].
    # A healthy quadruped reads exactly 0, so this needs no tuning.
    non_foot_contact_force_n: float = 5.0

    # Command tracking: |realised - commanded| planar velocity [m/s]
    tracking_error_degraded_mps: float = 0.35

    # A regime must hold this long before it is entered, so a single noisy frame
    # does not flip the annotation.
    dwell_steps: int = 3

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "height_degraded_m": float(self.height_degraded_m),
            "height_severe_m": float(self.height_severe_m),
            "height_failed_m": float(self.height_failed_m),
            "tilt_degraded_deg": float(self.tilt_degraded_deg),
            "tilt_severe_deg": float(self.tilt_severe_deg),
            "tilt_failed_deg": float(self.tilt_failed_deg),
            "non_foot_contact_force_n": float(self.non_foot_contact_force_n),
            "tracking_error_degraded_mps": float(self.tracking_error_degraded_mps),
            "dwell_steps": int(self.dwell_steps),
        }

    @classmethod
    def from_constraints(
        cls, constraints: Mapping[str, Any] | None
    ) -> "OperatingRegimeConfig":
        """
        Read the rails out of the ``orientation`` block of ``go2_constrains.yaml``.

        The thresholds live next to the constraints they were reconciled against,
        rather than in a second config that can drift away from them. Anything the
        YAML does not define keeps the default.
        """
        block = dict((constraints or {}).get("orientation") or {})
        defaults = cls()
        return cls(
            height_degraded_m=float(
                block.get("height_degraded_m", defaults.height_degraded_m)
            ),
            height_severe_m=float(
                block.get("height_severe_m", defaults.height_severe_m)
            ),
            height_failed_m=float(
                block.get("height_failed_m", defaults.height_failed_m)
            ),
            tilt_degraded_deg=float(
                block.get("tilt_degraded_deg", defaults.tilt_degraded_deg)
            ),
            tilt_severe_deg=float(
                block.get("tilt_severe_deg", defaults.tilt_severe_deg)
            ),
            tilt_failed_deg=float(
                block.get("tilt_failed_deg", defaults.tilt_failed_deg)
            ),
        )


def base_tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Angle between the base z-axis and gravity [deg], from a ``(T, 4)`` wxyz trace.

    One number instead of separate roll and pitch: a robot 12 deg out in roll and
    12 deg out in pitch is more tilted than either figure suggests, and it is the
    total that decides whether it is still walking.
    """
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    x, y = quat[:, 1], quat[:, 2]
    # World-frame z component of the base z-axis = R[2, 2] = 1 - 2(x^2 + y^2).
    cosine = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def posture_level(
    base_height_m: np.ndarray,
    tilt_deg: np.ndarray,
    tracking_error_mps: np.ndarray | None = None,
    config: OperatingRegimeConfig | None = None,
) -> np.ndarray:
    """
    Per-frame severity from the three *posture* signals: height, tilt, tracking.

    These are the ones that need a dwell filter — each is a continuous quantity
    crossing a threshold, so a noisy frame can straddle it. Body contact is
    handled separately in :func:`classify`, because it is not that kind of signal.
    """
    cfg = config if config is not None else OperatingRegimeConfig()
    height = np.asarray(base_height_m, dtype=np.float64).reshape(-1)
    tilt = np.asarray(tilt_deg, dtype=np.float64).reshape(-1)

    level = np.zeros(height.shape[0], dtype=np.int8)

    level = np.maximum(level, np.where(height < cfg.height_degraded_m, 1, 0))
    level = np.maximum(level, np.where(height < cfg.height_severe_m, 2, 0))
    level = np.maximum(level, np.where(height < cfg.height_failed_m, 3, 0))

    level = np.maximum(level, np.where(tilt > cfg.tilt_degraded_deg, 1, 0))
    level = np.maximum(level, np.where(tilt > cfg.tilt_severe_deg, 2, 0))
    level = np.maximum(level, np.where(tilt > cfg.tilt_failed_deg, 3, 0))

    if tracking_error_mps is not None:
        error = np.asarray(tracking_error_mps, dtype=np.float64).reshape(-1)
        level = np.maximum(
            level, np.where(error > cfg.tracking_error_degraded_mps, 1, 0)
        )

    return level


def body_contact_level(
    non_foot_contact_n: np.ndarray,
    config: OperatingRegimeConfig | None = None,
) -> np.ndarray:
    """
    ``severe`` wherever a non-foot geom carries load. **Not** dwell-filtered.

    A healthy quadruped reads exactly 0 N here, so there is no threshold to
    straddle and nothing for a dwell filter to clean up: any reading above the
    floor is a real event, and the substep averaging upstream has already
    discarded a one-substep graze. Suppressing a two-frame belly strike would
    mean labelling a frame ``nominal`` while the body is on the ground, which is
    the exact failure this annotation exists to prevent.
    """
    cfg = config if config is not None else OperatingRegimeConfig()
    body = np.asarray(non_foot_contact_n, dtype=np.float64).reshape(-1)
    return np.where(
        body > cfg.non_foot_contact_force_n, REGIME_INDEX["severe"], 0
    ).astype(np.int8)


def apply_dwell(level: np.ndarray, dwell_steps: int) -> np.ndarray:
    """
    Causal dwell filter: a new level must persist ``dwell_steps`` before it is entered.

    Same shape as the contact debouncer's minimum dwell, and for the same reason:
    a single noisy frame must not flip the annotation, and the filter must use
    only current and past samples so the column stays reproducible online.

    ``failed`` is absorbing. The robot does not get back up, and latching it keeps
    the v4 ``post_failure`` semantics — once unrecoverable, every later frame of
    the episode is flagged.
    """
    values = np.asarray(level, dtype=np.int8).reshape(-1)
    n_steps = values.shape[0]
    out = np.zeros(n_steps, dtype=np.int8)
    if n_steps == 0:
        return out

    dwell = max(1, int(dwell_steps))
    state = int(values[0])
    pending = -1
    count = 0
    failed = REGIME_INDEX["failed"]

    for t in range(n_steps):
        candidate = int(values[t])
        if state == failed:
            out[t] = failed
            continue
        if candidate == state:
            pending, count = -1, 0
        else:
            if candidate == pending:
                count += 1
            else:
                pending, count = candidate, 1
            if count >= dwell:
                state = pending
                pending, count = -1, 0
        out[t] = state
    return out


def classify(
    base_height_m: np.ndarray,
    tilt_deg: np.ndarray,
    non_foot_contact_n: np.ndarray,
    tracking_error_mps: np.ndarray | None = None,
    config: OperatingRegimeConfig | None = None,
) -> np.ndarray:
    """
    Regime names ``(T,)`` for a whole episode.

    Posture signals are dwell-filtered, then body contact is taken as a floor on
    top — so ``non_foot_contact_n`` above its threshold always reads at least
    ``severe``, and a dwell filter can never label a frame ``nominal`` while the
    robot's body is on the ground.
    """
    cfg = config if config is not None else OperatingRegimeConfig()
    held = apply_dwell(
        posture_level(base_height_m, tilt_deg, tracking_error_mps, cfg),
        cfg.dwell_steps,
    )
    level = np.maximum(held, body_contact_level(non_foot_contact_n, cfg))
    # `failed` stays absorbing after the floor is applied, too.
    failed = REGIME_INDEX["failed"]
    if (level == failed).any():
        first = int(np.flatnonzero(level == failed)[0])
        level[first:] = failed
    return np.asarray(REGIME_ORDER, dtype=REGIME_DTYPE)[level]


def regime_counts(regimes: Sequence[str]) -> Dict[str, int]:
    """Counts per regime, in severity order, including the empty ones."""
    values = np.asarray(regimes, dtype=REGIME_DTYPE).reshape(-1)
    return {
        name: int((values == name).sum()) for name in REGIME_ORDER
    }
