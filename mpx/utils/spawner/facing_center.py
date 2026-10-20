"""Spawn outside a centered obstacle, heading toward its center.

The ramp pyramid sits on the origin. The robot is placed on the square
``max(|x|, |y|) == reach`` just past the square base, with yaw aimed at the
center, and the navigation goal goes on that same square on the far side. A
square keeps every spawn and goal the same distance from the base edge, so the
out-of-bounds region can sit close around it.
"""

from __future__ import annotations

import numpy as np


def yaw_toward(xy, target=(0.0, 0.0)) -> float:
    """World yaw [rad] that points +x from ``xy`` toward ``target``."""
    dx = float(target[0]) - float(xy[0])
    dy = float(target[1]) - float(xy[1])
    return float(np.arctan2(dy, dx))


def _on_square(direction, reach: float) -> np.ndarray:
    """``direction`` scaled so its larger component is ``reach``."""
    v = np.asarray(direction, dtype=np.float64).reshape(2)
    if float(np.max(np.abs(v))) < 1e-6:
        v = np.array([1.0, 0.0])
    return v * float(reach) / float(np.max(np.abs(v)))


def sample_facing_center(
    rng: np.random.Generator,
    reach: float,
    height: float,
    target=(0.0, 0.0),
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return ``p0`` (z only) and ``(x, y, yaw)`` facing ``target``."""
    ang = float(rng.uniform(-np.pi, np.pi))
    x, y = np.asarray(target, dtype=np.float64) + _on_square((np.cos(ang), np.sin(ang)), reach)
    yaw = yaw_toward((x, y), target)
    p0 = np.array([0.0, 0.0, float(height)], dtype=np.float64)
    return p0, (float(x), float(y), yaw)


def opposite_side_xy(xy, reach: float, target=(0.0, 0.0)) -> np.ndarray:
    """Point on the square of ``reach`` straight across ``target`` from ``xy``."""
    origin = np.asarray(target, dtype=np.float64).reshape(2)
    v = np.asarray(xy, dtype=np.float64).reshape(-1)[:2] - origin
    return origin + _on_square(-v, reach)
