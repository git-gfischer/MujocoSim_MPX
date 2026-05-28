"""
Shared MuJoCo simulation helpers for spawn validation, orientation, and viewer overlays.

Functions are grouped below by role. Primary consumer: :mod:`mpx.utils.spawner`.

MuJoCo naming
    :func:`geom_label`, :func:`body_label` — readable IDs in collision error strings.

Quadruped body tree
    :func:`leg_prefix_from_body_name` — map a body name to FL/FR/RL/RR (or ``None``).
    :func:`bodies_direct_parent_child` — True if two bodies are immediate parent/child.
    :func:`geom_belongs_to_robot_under_root` — True if a geom is on the robot subtree.

Quaternion math (MuJoCo ``qpos`` order ``[w, x, y, z]``)
    :func:`quat_normalize_wxyz`, :func:`quat_mul_wxyz`, :func:`quat_yaw_wxyz`.

Viewer overlay
    :func:`alloc_decor_geom` — reserve a decoration geom in ``viewer.user_scn``.

Model lookup
    :func:`resolve_foot_geom_ids` — foot geom names → MuJoCo geom indices.
"""

from __future__ import annotations

from typing import Any, Sequence

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# MuJoCo naming helpers (human-readable collision messages)
# ---------------------------------------------------------------------------


def geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    """Return the geom name, or ``geom{id}`` if unnamed."""
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return nm if nm else f"geom{geom_id}"


def body_label(model: mujoco.MjModel, body_id: int) -> str:
    """Return the body name, or ``body{id}`` if unnamed."""
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return nm if nm else f"body{body_id}"


# Backward-compatible aliases (spawner imports the underscored names).
_geom_label = geom_label
_body_label = body_label


# ---------------------------------------------------------------------------
# Quadruped body-tree helpers (spawn collision filtering / tolerances)
# ---------------------------------------------------------------------------


def leg_prefix_from_body_name(body_name: str | None) -> str | None:
    """
    Return ``FL`` / ``FR`` / ``RL`` / ``RR`` if ``body_name`` is that leg or starts with ``LEG_``.

    Returns ``None`` for non-leg bodies (e.g. ``base``).
    """
    if not body_name:
        return None
    for leg in ("FL", "FR", "RL", "RR"):
        if body_name == leg or body_name.startswith(f"{leg}_"):
            return leg
    return None


_leg_prefix_from_body_name = leg_prefix_from_body_name


def bodies_direct_parent_child(model: mujoco.MjModel, body_a: int, body_b: int) -> bool:
    """True if ``body_a`` and ``body_b`` are immediate parent and child (either direction)."""
    pa = int(model.body_parentid[body_a])
    pb = int(model.body_parentid[body_b])
    return pa == body_b or pb == body_a


_bodies_direct_parent_child = bodies_direct_parent_child


def geom_belongs_to_robot_under_root(
    model: mujoco.MjModel,
    geom_id: int,
    robot_root_body_id: int,
) -> bool:
    """
    True if ``geom_id`` attaches to ``robot_root_body_id`` or a descendant body.

    MuJoCo detail: body id 0 is ``world`` and ``body_parentid[0] == 0``. A naïve
    ``while bid >= 0`` walk never terminates for scene geoms — this stops at world
    and guards against accidental parent cycles.
    """
    bid = int(model.geom_bodyid[geom_id])
    visited: set[int] = set()

    while True:
        if bid == robot_root_body_id:
            return True

        if bid <= 0:
            return False

        if bid in visited:
            return False
        visited.add(bid)

        parent = int(model.body_parentid[bid])
        if parent == bid:
            return False
        bid = parent


# ---------------------------------------------------------------------------
# Quaternion math (spawn orientation; scalar-first wxyz)
# ---------------------------------------------------------------------------


def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    """Unit-length quaternion; returns input unchanged if norm is tiny."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    return q / (n if n > 1e-12 else 1.0)


_quat_normalize_wxyz = quat_normalize_wxyz


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product (compose rotations), MuJoCo order ``[w, x, y, z]``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


_quat_mul_wxyz = quat_mul_wxyz


def quat_yaw_wxyz(yaw_rad: float) -> np.ndarray:
    """Pure yaw about world +Z by ``yaw_rad`` radians (wxyz)."""
    h = 0.5 * float(yaw_rad)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_from_rpy_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )

# ---------------------------------------------------------------------------
# Viewer overlay (passive MuJoCo viewer decoration scene)
# ---------------------------------------------------------------------------


def alloc_decor_geom(viewer: Any) -> int:
    """Increment ``viewer.user_scn.ngeom`` and return the new geom index."""
    viewer.user_scn.ngeom += 1
    return int(viewer.user_scn.ngeom - 1)


_alloc_decor_geom = alloc_decor_geom


# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------


def resolve_foot_geom_ids(model: mujoco.MjModel, foot_geom_names: Sequence[str]) -> list[int]:
    """Resolve foot geom names to IDs; raises ``ValueError`` if any name is missing."""
    gids = []
    for nm in foot_geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if gid < 0:
            raise ValueError(
                f"Foot geom name {nm!r} not found in model "
                "(use mujoco-visible collision geom names, e.g. Go2 FL/FR/RL/RR)."
            )
        gids.append(gid)
    return gids
