"""
Shared MuJoCo simulation helpers for spawn validation, orientation, and viewer overlays.

Functions are grouped below by role. Primary consumer: :mod:`mpx.utils.spawner`.

MuJoCo naming
    :func:`geom_label`, :func:`body_label` — readable IDs in collision error strings.

Quadruped body tree
    :func:`leg_prefix_from_body_name` — map a body name to FL/FR/RL/RR (or ``None``).
    :func:`bodies_direct_parent_child` — True if two bodies are immediate parent/child.
    :func:`geom_belongs_to_robot_under_root` — True if a geom is on the robot subtree.

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
# Viewer overlay (passive MuJoCo viewer decoration scene)
# ---------------------------------------------------------------------------


def alloc_decor_geom(viewer: Any) -> int:
    """Increment ``viewer.user_scn.ngeom`` and return the new geom index."""
    viewer.user_scn.ngeom += 1
    return int(viewer.user_scn.ngeom - 1)


_alloc_decor_geom = alloc_decor_geom


#----------------------------------------------------------------------------
def timer_run(duty_factor,step_freq, leg_time, dt):
    import jax.numpy as jnp
    # Extract relevant fields
    # Update timer
    leg_time = leg_time + dt * step_freq
    leg_time = jnp.where(leg_time > 1, leg_time - 1, leg_time)
    contact = jnp.where(leg_time < duty_factor, 1, 0)

    return contact, leg_time

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


#----------------------------------------------------------------------------

def geom_ids(model: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    """Return the MuJoCo geom ids for the provided geom names."""

    return np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names],
        dtype=np.int32,
    )


def geom_positions(data: mujoco.MjData, geom_ids: Sequence[int], flatten: bool = True) -> np.ndarray:
    """Return the geom positions for the selected geoms."""

    positions = np.asarray([data.geom_xpos[int(geom_id)] for geom_id in geom_ids], dtype=np.float64)
    return positions.reshape(-1) if flatten else positions

def _reserve_user_geom(viewer) -> int:
    if viewer is None:
        return -1
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        raise ValueError(
            f"Viewer user scene is full ({viewer.user_scn.ngeom}/{viewer.user_scn.maxgeom})."
        )
    viewer.user_scn.ngeom += 1
    return viewer.user_scn.ngeom - 1

#---------------------------Render Helpers---------------------------
def render_vector(
    viewer,
    vector: np.ndarray,
    pos: np.ndarray,
    scale: float,
    color: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0]),
    geom_id: int = -1,
) -> int:
    """Render a decorative arrow aligned with the provided vector."""

    if viewer is None:
        return -1

    if geom_id < 0:
        geom_id = _reserve_user_geom(viewer)

    geom = viewer.user_scn.geoms[geom_id]
    direction = np.asarray(vector, dtype=np.float64).reshape(3)
    start = np.asarray(pos, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0])
        norm = 1.0

    end = start + (scale * direction / norm)
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.ones(3) * 1e-3,
        pos=np.zeros(3),
        mat=np.eye(3).reshape(9),
        rgba=np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, start, end)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1
    return geom_id


def render_sphere(
    viewer,
    position: np.ndarray,
    diameter: float,
    color: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0]),
    geom_id: int = -1,
) -> int:
    """Render a decorative sphere at the provided position."""

    if viewer is None:
        return -1

    if geom_id < 0:
        geom_id = _reserve_user_geom(viewer)

    geom = viewer.user_scn.geoms[geom_id]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.5 * diameter, 0.0, 0.0]),
        pos=np.asarray(position, dtype=np.float64).reshape(3),
        mat=np.eye(3).reshape(9),
        rgba=np.asarray(color, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1
    return geom_id


def render_sphere_trajectory(
    viewer,
    positions: np.ndarray,
    alphas: np.ndarray,
    diameter: float,
    color: np.ndarray = np.array([1.0, 0.45, 0.0, 1.0]),
    geom_ids: list[int] | None = None,
) -> list[int]:
    """Render or update a sequence of decorative spheres."""

    if viewer is None:
        return []

    positions = np.asarray(positions, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if positions.shape[0] == 0:
        return []

    if geom_ids is None or len(geom_ids) != positions.shape[0]:
        geom_ids = [-1] * positions.shape[0]

    base_color = np.asarray(color, dtype=np.float32)
    for idx, (position, alpha) in enumerate(zip(positions, alphas, strict=False)):
        rgba = np.array(base_color, copy=True)
        rgba[3] = float(alpha)
        geom_ids[idx] = render_sphere(
            viewer,
            position,
            diameter,
            color=rgba,
            geom_id=geom_ids[idx],
        )

    return geom_ids

#-------------------------Ghost Geom Helpers-------------------------
def _build_ghost_geoms(
    viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
) -> dict[int, dict[str, Any]]:
    """Cache the static visual spec for each rendered ghost geom."""

    scene = mujoco.MjvScene(mj_model, maxgeom=max(2 * mj_model.ngeom, 200))
    mujoco.mjv_updateScene(
        mj_model,
        mj_data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    ghost_geoms = {}
    ignored_names = {"floor", "plane", "world", "ground"}
    for geom in scene.geoms[: scene.ngeom]:
        if geom.segid == -1:
            continue

        geom_model_id = int(geom.objid)
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_model_id)
        body_id = mj_model.geom_bodyid[geom_model_id]
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_rgba = mj_model.geom_rgba[geom_model_id]
        if geom_name in ignored_names or body_name in ignored_names or geom_rgba[3] == 0:
            continue

        ghost_geoms[_reserve_user_geom(viewer)] = {
            "model_id": geom_model_id,
            "type": int(geom.type),
            "size": np.array(geom.size, copy=True),
            "rgba": np.array(geom.rgba, copy=True),
            "dataid": int(geom.dataid),
            "emission": float(geom.emission),
            "specular": float(geom.specular),
            "shininess": float(geom.shininess),
        }

    return ghost_geoms


def render_ghost_robot(
    viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    alpha: float = 0.5,
    ghost_geoms: dict[int, dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Render or update a translucent ghost robot in a passive MuJoCo viewer."""

    if viewer is None:
        return {}

    if ghost_geoms is None or len(ghost_geoms) == 0:
        # Build the cache once from the model geoms, then only update transforms.
        ghost_geoms = _build_ghost_geoms(viewer, mj_model, mj_data)

    for scn_id, geom in ghost_geoms.items():
        geom_model_id = geom["model_id"]
        rgba = np.array(geom["rgba"], copy=True)
        rgba[3] = alpha

        decorative_geom = viewer.user_scn.geoms[scn_id]
        mujoco.mjv_initGeom(
            decorative_geom,
            type=geom["type"],
            rgba=rgba,
            size=geom["size"],
            pos=mj_data.geom_xpos[geom_model_id],
            mat=mj_data.geom_xmat[geom_model_id].reshape(9),
        )
        decorative_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        decorative_geom.segid = -1
        decorative_geom.objid = -1
        decorative_geom.dataid = geom["dataid"]
        decorative_geom.emission = geom["emission"]
        decorative_geom.specular = geom["specular"]
        decorative_geom.shininess = geom["shininess"]
        decorative_geom.reflectance = 0.0

    return ghost_geoms


def render_ghost_trajectory(
    viewer,
    mj_model: mujoco.MjModel,
    qpos_sequence: np.ndarray,
    alphas: np.ndarray,
    ghost_geoms: list[dict[int, dict[str, Any]] | None] | None = None,
    scratch_data: mujoco.MjData | None = None,
    subsample: int = 20,
) -> tuple[list[dict[int, dict[str, Any]]], mujoco.MjData]:
    """Render a sequence of ghost robots, typically used for planned trajectories."""

    qpos_sequence = np.asarray(qpos_sequence)
    alphas = np.asarray(alphas)
    if subsample > 1:
        qpos_sequence = qpos_sequence[::subsample]
        alphas = alphas[::subsample]

    if scratch_data is None:
        scratch_data = mujoco.MjData(mj_model)
    if ghost_geoms is None or len(ghost_geoms) != len(qpos_sequence):
        ghost_geoms = [None] * len(qpos_sequence)

    for idx, (qpos, alpha) in enumerate(zip(qpos_sequence, alphas, strict=False)):
        scratch_data.qpos = qpos
        mujoco.mj_forward(mj_model, scratch_data)
        ghost_geoms[idx] = render_ghost_robot(
            viewer,
            mj_model,
            scratch_data,
            alpha=float(alpha),
            ghost_geoms=ghost_geoms[idx],
        )

    return ghost_geoms, scratch_data
#----------------------------------------------------------------------------
def setup_tracking_camera(
    viewer,
    model: mujoco.MjModel,
    body_name: str = "base",
    distance: float = 2.5,
    azimuth: float = 135.0,
    elevation: float = -20.0,
) -> None:
    """Set the passive viewer camera to track a robot body.

    Call once after ``viewer.sync()`` inside the ``with launch_passive(...)`` block.

    Args:
        viewer:     MuJoCo passive viewer handle.
        model:      The MjModel used by the viewer.
        body_name:  Name of the body to track (e.g. ``"base"``).
        distance:   Camera distance from the tracked body [m].
        azimuth:    Horizontal orbit angle [deg]. 180 = behind, 90 = right side.
        elevation:  Vertical angle [deg]. Negative looks down at the robot.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(
            f"Body {body_name!r} not found in model. "
            "Check the body name with mujoco.mj_id2name()."
        )
    viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body_id
    viewer.cam.distance    = distance
    viewer.cam.azimuth     = azimuth
    viewer.cam.elevation   = elevation