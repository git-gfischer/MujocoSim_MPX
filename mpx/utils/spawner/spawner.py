"""
Randomize robot free-joint X, Y and yaw inside a map region; base Z is fixed from ``p0``.

Before committing a pose, :class:`RobotMapSpawner` can run two collision checks:
self-collision (robot–robot) and environment collision (non-foot robot parts vs scene).

Typical use after ``QuadrupedEnv.reset()``::

    spawner = RobotMapSpawner(..., foot_geom_names=("FL", "FR", "RL", "RR"))
    qpos = spawner.apply(env, config.p0, config.quat0, config.q0)
    mpc.reset(qpos.copy(), np.zeros_like(env.mjData.qvel))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import mujoco
import numpy as np

import jax.numpy as jnp


from mpx.utils.sim_utils import (
    _alloc_decor_geom, 
    _body_label, 
    _bodies_direct_parent_child,
    _geom_label,
    _leg_prefix_from_body_name, # 
    geom_belongs_to_robot_under_root,
    resolve_foot_geom_ids,
)

from mpx.utils.math_utils.quad_math import (
    _quat_mul_wxyz,
    _quat_normalize_wxyz,
    quat_yaw_wxyz,
)

def reset_robot_and_mpc(env, config, mpc, spawner: RobotMapSpawner | None = None):
    """
    Place the robot from ``config`` / spawner (never XML keyframe 0) and sync MPC.

    Avoids ``env.reset(random=False)`` with no ``qpos``, which can leave a zero-norm
    quaternion on procedural scenes and break ``base_configuration`` during ``render()``.
    """
    if spawner is not None:
        spawner.apply(env, config.p0, config.quat0, config.q0)
    else:
        qpos = np.asarray(
            jnp.concatenate([config.p0, config.quat0, config.q0]), dtype=np.float64
        )
        qvel = np.zeros(env.mjModel.nv, dtype=np.float64)
        env.reset(qpos=qpos, qvel=qvel, random=False)

    env.step_num = 0
    env.mjData.time = 0.0
    mpc.reset(
        np.asarray(env.mjData.qpos, dtype=np.float64).copy(),
        np.asarray(env.mjData.qvel, dtype=np.float64).copy(),
    )
    q = jnp.asarray(env.mjData.qpos[7 : 7 + config.n_joints], dtype=jnp.float32)
    tau = jnp.zeros(config.n_joints)
    return q, tau


def _random_map_respawn(env, config, mpc, _spawner):
    """Re-sample X/Y/yaw in the spawn region, sync MPC and low-level ``q``/``tau``."""
    q, tau = reset_robot_and_mpc(env, config, mpc, _spawner)
    return env, q, tau

class SpawnCollisionError(RuntimeError):
    """No collision-free pose was found within ``max_spawn_attempts``."""

    pass


def _self_collision_penetration_tol(
    model: mujoco.MjModel,
    body_a: int,
    body_b: int,
    *,
    same_leg_tol: float,
    adjacent_link_tol: float,
    cross_body_tol: float,
) -> float:
    """
    Penetration tolerance (metres, as positive ``-dist``) for a robot–robot pair.

    Same-leg and parent/child pairs use looser margins (joint neighbourhood).
    Different legs or base↔limb use a tight margin (catches leg-through-torso spawns).
    """
    if _bodies_direct_parent_child(model, body_a, body_b):
        return adjacent_link_tol
    name_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_a)
    name_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_b)
    leg_a = _leg_prefix_from_body_name(name_a)
    leg_b = _leg_prefix_from_body_name(name_b)
    if leg_a is not None and leg_a == leg_b:
        return same_leg_tol
    return cross_body_tol


def robot_self_collision_reason(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_root_body_id: int,
    foot_geom_ids: Sequence[int] | None = None,
    ignore_foot_foot: bool = True,
    same_leg_tol: float = 4e-3,
    adjacent_link_tol: float = 6e-3,
    cross_body_tol: float = 2e-3,
) -> str | None:
    """
    Reject poses where robot geoms interpenetrate (leg inside torso, cross-leg clash, …).

    Uses MuJoCo ``contact.dist`` after ``mj_forward``. Pair-dependent tolerances apply.
    Optional ``ignore_foot_foot``: skip foot–foot contacts (margins often overlap in nominal stand).
    """
    foot_set = frozenset(int(g) for g in foot_geom_ids) if foot_geom_ids else frozenset()

    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if ignore_foot_foot and foot_set and g1 in foot_set and g2 in foot_set:
            continue
        if not geom_belongs_to_robot_under_root(model, g1, robot_root_body_id):
            continue
        if not geom_belongs_to_robot_under_root(model, g2, robot_root_body_id):
            continue

        dist = float(c.dist)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        tol = _self_collision_penetration_tol(
            model,
            b1,
            b2,
            same_leg_tol=same_leg_tol,
            adjacent_link_tol=adjacent_link_tol,
            cross_body_tol=cross_body_tol,
        )
        if dist < -tol:
            n1 = _geom_label(model, g1)
            n2 = _geom_label(model, g2)
            bn1 = _body_label(model, b1)
            bn2 = _body_label(model, b2)
            return (
                f"self: {bn1}/{n1} vs {bn2}/{n2}, dist={dist:.5f} (tol {-tol:.5f})"
            )
    return None


def robot_env_collision_reason(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_root_body_id: int,
    foot_geom_ids: Sequence[int],
    foot_max_penetration: float = 4e-3,
) -> str | None:
    """
    Reject poses where non-foot robot parts touch the scene (boxes; ground OK via feet).

    Feet may contact terrain/obstacles; penetration deeper than ``foot_max_penetration`` is rejected.
    """
    foot_set = frozenset(int(g) for g in foot_geom_ids)
    if not foot_set:
        raise ValueError("foot_geom_ids must be non-empty for environment collision checks.")

    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        dist = float(c.dist)
        r1 = geom_belongs_to_robot_under_root(model, g1, robot_root_body_id)
        r2 = geom_belongs_to_robot_under_root(model, g2, robot_root_body_id)

        if r1 == r2:
            continue

        robot_gid = g1 if r1 else g2
        other_gid = g2 if r1 else g1
        rn = _geom_label(model, robot_gid)
        on = _geom_label(model, other_gid)

        if robot_gid not in foot_set:
            if dist < 0.0:
                return f"env: non-foot {rn}[{robot_gid}] vs {on}[{other_gid}], dist={dist:.5f}"
            continue

        if dist < -float(foot_max_penetration):
            return (
                f"env: foot {rn}[{robot_gid}] vs {on}[{other_gid}], "
                f"dist={dist:.5f} (tol {-foot_max_penetration:.5f})"
            )
    return None


def forbidden_spawn_contact_reason(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_root_body_id: int,
    foot_geom_ids: Sequence[int],
    check_self_collision: bool = True,
    check_env_collision: bool = True,
    foot_max_penetration: float = 4e-3,
    same_leg_tol: float = 4e-3,
    adjacent_link_tol: float = 6e-3,
    cross_body_tol: float = 2e-3,
) -> str | None:
    """Run self- then environment-collision checks; return first failure reason or ``None``."""
    if check_self_collision:
        reason = robot_self_collision_reason(
            model,
            data,
            robot_root_body_id=robot_root_body_id,
            foot_geom_ids=foot_geom_ids if foot_geom_ids else None,
            same_leg_tol=same_leg_tol,
            adjacent_link_tol=adjacent_link_tol,
            cross_body_tol=cross_body_tol,
        )
        if reason is not None:
            return reason
    if check_env_collision:
        return robot_env_collision_reason(
            model,
            data,
            robot_root_body_id=robot_root_body_id,
            foot_geom_ids=foot_geom_ids,
            foot_max_penetration=foot_max_penetration,
        )
    return None


def spawn_pose_has_forbidden_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_root_body_id: int,
    foot_geom_ids: Sequence[int],
    check_self_collision: bool = True,
    check_env_collision: bool = True,
    foot_max_penetration: float = 4e-3,
    same_leg_tol: float = 4e-3,
    adjacent_link_tol: float = 6e-3,
    cross_body_tol: float = 2e-3,
) -> bool:
    """Return True if the pose fails self and/or environment collision rules."""
    return (
        forbidden_spawn_contact_reason(
            model,
            data,
            robot_root_body_id=robot_root_body_id,
            foot_geom_ids=foot_geom_ids,
            check_self_collision=check_self_collision,
            check_env_collision=check_env_collision,
            foot_max_penetration=foot_max_penetration,
            same_leg_tol=same_leg_tol,
            adjacent_link_tol=adjacent_link_tol,
            cross_body_tol=cross_body_tol,
        )
        is not None
    )


def contact_free_spawn_pose(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    robot_root_body_id: int,
    foot_geom_ids: Sequence[int],
    check_self_collision: bool = True,
    check_env_collision: bool = True,
    foot_max_penetration: float = 4e-3,
    same_leg_tol: float = 4e-3,
    adjacent_link_tol: float = 6e-3,
    cross_body_tol: float = 2e-3,
) -> bool:
    """Return True if ``qpos`` passes both collision checks (uses a scratch ``MjData``)."""
    trial = mujoco.MjData(model)
    trial.qpos[:] = np.asarray(qpos, dtype=np.float64).reshape(model.nq)
    trial.qvel[:] = 0.0
    mujoco.mj_forward(model, trial)
    return not spawn_pose_has_forbidden_contacts(
        model,
        trial,
        robot_root_body_id=robot_root_body_id,
        foot_geom_ids=foot_geom_ids,
        check_self_collision=check_self_collision,
        check_env_collision=check_env_collision,
        foot_max_penetration=foot_max_penetration,
        same_leg_tol=same_leg_tol,
        adjacent_link_tol=adjacent_link_tol,
        cross_body_tol=cross_body_tol,
    )


@dataclass
class SpawnRegion:
    """Uniform sampling bounds for X, Y and yaw (metres, radians). Z is not sampled."""

    #default values for the spawn region
    x: tuple[float, float] = (-4.0, 4.0)
    y: tuple[float, float] = (-4.0, 4.0)
    yaw: tuple[float, float] = (-np.pi, np.pi)


    def xy_center_half_extents(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame rectangle center (3,) and half-sizes (hx, hy) for XY overlay."""
        x_lo, x_hi = self.x
        y_lo, y_hi = self.y
        center = np.array(
            [0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), 0.0],
            dtype=np.float64,
        )
        half = np.array(
            [0.5 * (x_hi - x_lo), 0.5 * (y_hi - y_lo)],
            dtype=np.float64,
        )
        return center, half


@dataclass
class SpawnRegionVisual:
    """Handles for ring edge geoms in ``viewer.user_scn`` (re-use each frame)."""
    edges: tuple[int, int, int, int]
def _init_decor_capsule_edge(
    geom: mujoco.MjvGeom,
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    width: float,
    rgba: np.ndarray,
) -> None:
    """Capsule along the segment ``p0`` → ``p1`` (matches gym_quadruped ``render_line``)."""
    vector = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        length = 1e-9
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    vec_z = vector / length
    rand_vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(rand_vec, vec_z)) > 0.9:
        rand_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vec_x = rand_vec - np.dot(rand_vec, vec_z) * vec_z
    vec_x /= np.linalg.norm(vec_x)
    vec_y = np.cross(vec_z, vec_x)
    ori_mat = np.column_stack([vec_x, vec_y, vec_z])
    w = float(width)
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([w, length / 2.0 + w / 4.0, w], dtype=np.float64),
        pos=(np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64)) / 2.0,
        mat=ori_mat.flatten(),
        rgba=np.asarray(rgba, dtype=np.float64).reshape(4),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1
    geom.emission = 0.35


def render_spawn_region(
    viewer: Any,
    region: SpawnRegion,
    *,
    z: float = 0.08,
    ring_rgba: np.ndarray | Sequence[float] | None = None,
    ring_width: float = 0.1,
    visual: SpawnRegionVisual | None = None,
) -> SpawnRegionVisual | None:
    """
    Draw a red transparent rectangular ring (XY bounds) in the passive MuJoCo viewer.

    Call **after** ``env.render()`` (viewer must exist). Re-call each frame with ``visual``.
    """
    if viewer is None:
        return None

    if ring_rgba is None:
        ring_rgba = np.array([1.0, 0.05, 0.05, 0.55], dtype=np.float64)

    x_lo, x_hi = region.x
    y_lo, y_hi = region.y
    zf = float(z)
    corners = [
        np.array([x_lo, y_lo, zf], dtype=np.float64),
        np.array([x_hi, y_lo, zf], dtype=np.float64),
        np.array([x_hi, y_hi, zf], dtype=np.float64),
        np.array([x_lo, y_hi, zf], dtype=np.float64),
    ]

    if visual is None:
        edge_ids = tuple(_alloc_decor_geom(viewer) for _ in range(4))
        visual = SpawnRegionVisual(edges=edge_ids)

    rgba = np.asarray(ring_rgba, dtype=np.float64)
    for i, eid in enumerate(visual.edges):
        _init_decor_capsule_edge(
            viewer.user_scn.geoms[eid],
            p0=corners[i],
            p1=corners[(i + 1) % 4],
            width=ring_width,
            rgba=rgba,
        )
    return visual


class RobotMapSpawner:
    """
    Randomize free-joint X, Y and yaw inside ``region``; base height ``Z`` is fixed from ``p0``.

    With collision checking enabled, each candidate pose is validated in two steps:

    1. **Self** — robot geoms must not interpenetrate (e.g. leg through torso).
    2. **Environment** — only foot geoms may touch scene objects (ground/boxes).

    After each sampled X/Y/yaw, **foot vertical relief** (optional, on by default) raises the
    free-joint ``z`` in small steps up to ``foot_relief_max`` so feet can clear the floor when
    random yaw would otherwise drive them slightly underground at fixed ``p0[2]``.
    """

    def __init__(
        self,
        region: SpawnRegion | None = None,
        *,
        rng: np.random.Generator | None = None,
        reset_velocities: bool = True,
        check_collisions: bool = False,
        check_self_collision: bool | None = None,
        check_env_collision: bool | None = None,
        foot_geom_names: Sequence[str] | None = None,
        robot_root_body_name: str = "base",
        foot_max_penetration: float = 4e-3,
        same_leg_tol: float = 4e-3,
        adjacent_link_tol: float = 6e-3,
        cross_body_tol: float = 2e-3,
        max_spawn_attempts: int = 512,
        on_collision_exhausted: Literal["raise", "origin"] = "raise",
        verbose: bool = False,
        verbose_progress_every: int = 75,
        try_foot_vertical_relief: bool = True,
        foot_relief_step: float = 0.005,
        foot_relief_max: float = 0.10,
    ) -> None:
        self.region = region or SpawnRegion()
        self.rng = rng or np.random.default_rng()
        self.reset_velocities = reset_velocities
        self.check_self_collision = (
            bool(check_collisions) if check_self_collision is None else bool(check_self_collision)
        )
        self.check_env_collision = (
            bool(check_collisions) if check_env_collision is None else bool(check_env_collision)
        )
        self.check_collisions = self.check_self_collision or self.check_env_collision
        self.foot_geom_names = tuple(foot_geom_names) if foot_geom_names else ()
        self.robot_root_body_name = robot_root_body_name
        self.foot_max_penetration = float(foot_max_penetration)
        self.same_leg_tol = float(same_leg_tol)
        self.adjacent_link_tol = float(adjacent_link_tol)
        self.cross_body_tol = float(cross_body_tol)
        self.max_spawn_attempts = int(max_spawn_attempts)
        self.on_collision_exhausted = on_collision_exhausted
        self.verbose = bool(verbose)
        self.verbose_progress_every = max(1, int(verbose_progress_every))

        fr_step = float(foot_relief_step)
        fr_max = float(foot_relief_max)
        if fr_step <= 0:
            raise ValueError("foot_relief_step must be positive.")
        if fr_max <= 0:
            raise ValueError("foot_relief_max must be positive.")
        self.try_foot_vertical_relief = bool(try_foot_vertical_relief)
        self.foot_relief_step = fr_step
        self.foot_relief_max = fr_max

        self._trial_cache: tuple[int, mujoco.MjData] | None = None
        self._geom_model_id_cache: int | None = None
        self._geom_root_feet_cache: tuple[int, tuple[int, ...]] | None = None

    def _get_trial_data(self, model: mujoco.MjModel) -> mujoco.MjData:
        key = id(model)
        if self._trial_cache is None or self._trial_cache[0] != key:
            self._trial_cache = (key, mujoco.MjData(model))
        return self._trial_cache[1]

    def _root_and_feet(self, model: mujoco.MjModel) -> tuple[int, tuple[int, ...]]:
        """Cache MuJoCo body id for the robot root and foot geom IDs per compiled model."""
        mid = id(model)
        if self._geom_model_id_cache != mid or self._geom_root_feet_cache is None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.robot_root_body_name)
            if bid < 0:
                raise ValueError(f"Robot root body {self.robot_root_body_name!r} not in model.")
            feet: tuple[int, ...]
            if self.check_env_collision:
                if not self.foot_geom_names:
                    raise ValueError(
                        "Environment collision check requires foot_geom_names "
                        "(e.g. ('FL','FR','RL','RR'))."
                    )
                feet = tuple(resolve_foot_geom_ids(model, self.foot_geom_names))
            elif self.foot_geom_names:
                # Resolve feet for self-collision (e.g. ignore foot–foot margin overlap).
                feet = tuple(resolve_foot_geom_ids(model, self.foot_geom_names))
            else:
                feet = ()
            self._geom_model_id_cache = mid
            self._geom_root_feet_cache = (bid, feet)
        return self._geom_root_feet_cache

    def _debug(self, msg: str) -> None:
        if self.verbose:
            print(f"[RobotMapSpawner] {msg}", flush=True)
    
    @classmethod
    def from_config(
        cls,
        cfg,  # BasePoseRandomizationConfig OR SpawnConfig
        *,
        foot_geom_names=(),
        check_collisions: bool = False,
        **kwargs,
    ) -> "RobotMapSpawner":
        """Build a spawner from BasePoseRandomizationConfig or SpawnConfig.

        SpawnConfig: uses spawn_region() (manual_region_x/y + region_yaw).
        BasePoseRandomizationConfig: uses yaw_range_deg only, XY fixed at 0.
        """
        # SpawnConfig path — has XY + yaw from manual_region_* and region_yaw
        if hasattr(cfg, 'spawn_region'):
            return cls(
                region=cfg.spawn_region(),
                foot_geom_names=foot_geom_names,
                check_collisions=check_collisions,
                **kwargs,
            )
        # BasePoseRandomizationConfig path — yaw only, XY fixed at origin
        if not cfg.enabled:
            region = SpawnRegion(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))
        else:
            yaw_lo = float(np.deg2rad(cfg.yaw_range_deg[0]))
            yaw_hi = float(np.deg2rad(cfg.yaw_range_deg[1]))
            region = SpawnRegion(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(yaw_lo, yaw_hi))
        return cls(
            region=region,
            foot_geom_names=foot_geom_names,
            check_collisions=check_collisions,
            **kwargs,
        )

    def _evaluate_pose(self, model: mujoco.MjModel, qpos: np.ndarray) -> tuple[str | None, int]:
        trial = self._get_trial_data(model)
        trial.qpos[:] = np.asarray(qpos, dtype=np.float64).reshape(model.nq)
        trial.qvel[:] = 0.0
        mujoco.mj_forward(model, trial)
        ncon = int(trial.ncon)
        robot_root_body_id, foot_ids = self._root_and_feet(model)
        if not self.check_self_collision and not self.check_env_collision:
            return None, ncon
        reason = forbidden_spawn_contact_reason(
            model,
            trial,
            robot_root_body_id=robot_root_body_id,
            foot_geom_ids=foot_ids if foot_ids else (),
            check_self_collision=self.check_self_collision,
            check_env_collision=self.check_env_collision,
            foot_max_penetration=self.foot_max_penetration,
            same_leg_tol=self.same_leg_tol,
            adjacent_link_tol=self.adjacent_link_tol,
            cross_body_tol=self.cross_body_tol,
        )
        return reason, ncon

    def _seek_vertical_relief(
        self, model: mujoco.MjModel, qpos_base: np.ndarray
    ) -> tuple[np.ndarray | None, str | None, int]:
        """
        Try ``qpos_base``; if rejected, raise free-joint ``z`` in ``foot_relief_step`` steps up to ``foot_relief_max``.

        Keeps sampled X, Y, yaw and joint angles; only adjusts base height so feet clear the floor/boxes.
        """
        q = np.asarray(qpos_base, dtype=np.float64).copy()
        reason, ncon = self._evaluate_pose(model, q)
        if reason is None:
            return q, None, ncon
        if not self.try_foot_vertical_relief:
            return None, reason, ncon

        z0 = float(q[2])
        z_cap = z0 + self.foot_relief_max
        n_steps = max(1, int(np.ceil(self.foot_relief_max / self.foot_relief_step)))
        last_reason = reason
        for k in range(1, n_steps + 1):
            z_new = min(z0 + float(k) * self.foot_relief_step, z_cap)
            q[2] = z_new
            reason, ncon = self._evaluate_pose(model, q)
            last_reason = reason
            if reason is None:
                self._debug(
                    f"foot vertical relief: Δz={z_new - z0:.4f}m (cap={self.foot_relief_max}m); ncon={ncon}"
                )
                return q.copy(), None, ncon

        self._debug(
            f"foot vertical relief: failed up to Δz={self.foot_relief_max}m → {last_reason}"
        )
        return None, last_reason, ncon

    def _evaluate_self_collision_only(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
    ) -> tuple[str | None, int]:
        """
        Check only robot self-collision.

        For this spawner, XY and yaw do not change the robot's internal joint
        configuration.  Therefore, if the initial joint pose is already
        self-colliding, randomizing map placement cannot fix it.  Failing fast
        here avoids spending ``max_spawn_attempts`` on impossible samples.
        """
        if not self.check_self_collision:
            return None, 0

        trial = self._get_trial_data(model)
        trial.qpos[:] = np.asarray(qpos, dtype=np.float64).reshape(model.nq)
        trial.qvel[:] = 0.0
        mujoco.mj_forward(model, trial)

        robot_root_body_id, foot_ids = self._root_and_feet(model)
        reason = robot_self_collision_reason(
            model,
            trial,
            robot_root_body_id=robot_root_body_id,
            foot_geom_ids=foot_ids if foot_ids else None,
            same_leg_tol=self.same_leg_tol,
            adjacent_link_tol=self.adjacent_link_tol,
            cross_body_tol=self.cross_body_tol,
        )
        return reason, int(trial.ncon)

    @staticmethod
    def _reason_bucket(reason: str) -> str:
        """Coarse rejection category used in debug summaries."""
        if reason.startswith("self:"):
            return "self"
        if reason.startswith("env: foot"):
            return "env_foot_penetration"
        if reason.startswith("env: non-foot"):
            return "env_non_foot"
        return "other"

    def render_spawn_region(
        self,
        viewer: Any,
        *,
        z: float = 0.08,
        ring_rgba: np.ndarray | Sequence[float] | None = None,
        ring_width: float = 0.1,
        visual: SpawnRegionVisual | None = None,
    ) -> SpawnRegionVisual | None:
        """Red transparent ring for ``self.region``; call after ``env.render()`` each frame."""
        return render_spawn_region(
            viewer,
            self.region,
            z=z,
            ring_rgba=ring_rgba,
            ring_width=ring_width,
            visual=visual,
        )

    def sample_xy_yaw(self) -> np.ndarray:
        """Return ``array([x, y, yaw])`` with independent uniform samples."""
        # Spawn the base of the robot inside the spawn region
        x_lo, x_hi = self.region.x
        y_lo, y_hi = self.region.y
        yaw_lo, yaw_hi = self.region.yaw
        return np.array(
            [
                self.rng.uniform(x_lo, x_hi),
                self.rng.uniform(y_lo, y_hi),
                self.rng.uniform(yaw_lo, yaw_hi),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def orient_with_yaw(base_quat_wxyz: Sequence[float] | np.ndarray, yaw_rad: float) -> np.ndarray:
        """Apply world yaw on the left: ``q_yaw * q_base`` (normalized wxyz)."""
        qy = quat_yaw_wxyz(yaw_rad)
        qb = _quat_normalize_wxyz(np.asarray(base_quat_wxyz, dtype=np.float64))
        return _quat_normalize_wxyz(_quat_mul_wxyz(qy, qb))

    def build_qpos(
        self,
        p0: Sequence[float] | np.ndarray,
        quat0: Sequence[float] | np.ndarray,
        q0: Sequence[float] | np.ndarray,
        xy_yaw: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Assemble full ``qpos`` from random or supplied ``[x, y, yaw]``.

        Only X and Y are taken from the sample; ``Z`` is always ``p0[2]`` (not randomized).
        """
        if xy_yaw is None:
            xy_yaw = self.sample_xy_yaw()
        x, y, yaw = np.asarray(xy_yaw, dtype=np.float64).reshape(3)
        p = np.asarray(p0, dtype=np.float64).copy().reshape(3)
        p[0] = x
        p[1] = y
        # Z fixed from nominal pose p0
        q_wxyz = self.orient_with_yaw(quat0, yaw)
        qj = np.asarray(q0, dtype=np.float64).ravel()
        return np.concatenate([p, q_wxyz, qj])

    def apply(
        self,
        env: Any,
        p0: Sequence[float] | np.ndarray,
        quat0: Sequence[float] | np.ndarray,
        q0: Sequence[float] | np.ndarray,
        *,
        xy_yaw: Sequence[float] | np.ndarray | None = None,
        forward: bool = True,
    ) -> np.ndarray:
        """
        Pick a feasible pose (resampling if collisions are forbidden) and write ``env.mjData``.

        ``env`` must expose ``mjModel`` and ``mjData``.
        """
        model = env.mjModel
        data = env.mjData

        def finalize(qvec: np.ndarray) -> np.ndarray:
            data.qpos[:] = np.asarray(qvec, dtype=np.float64).reshape(model.nq)
            if self.reset_velocities:
                data.qvel[:] = 0.0
            if forward:
                mujoco.mj_forward(model, data)
            return np.asarray(data.qpos.copy(), dtype=np.float64)

        qpos_candidate = self.build_qpos(p0, quat0, q0, xy_yaw=xy_yaw)

        if not self.check_collisions:
            self._debug(f"apply: collision check OFF; committing pose directly (nq={model.nq})")
            return finalize(qpos_candidate)

        robot_root_body_id, foot_ids = self._root_and_feet(model)
        self._debug(
            f"apply: collision ON self={self.check_self_collision} env={self.check_env_collision}; "
            f"robot_root={self.robot_root_body_name!r} bid={robot_root_body_id}; "
            f"feet={list(self.foot_geom_names)} ids={list(foot_ids)}; "
            f"tol cross={self.cross_body_tol} same_leg={self.same_leg_tol} foot={self.foot_max_penetration}; "
            f"region xy={self.region.x}×{self.region.y} yaw={self.region.yaw}; max_attempts={self.max_spawn_attempts}"
        )

        # Stage 1: self-collision depends only on joints + base orientation (invariant to XY / yaw).
        p_nom = np.asarray(p0, dtype=np.float64).reshape(3)
        qpos_invariant = self.build_qpos(
            p0, quat0, q0, xy_yaw=(float(p_nom[0]), float(p_nom[1]), 0.0)
        )
        invariant_self_reason, invariant_ncon = self._evaluate_self_collision_only(
            model, qpos_invariant
        )
        if invariant_self_reason is not None:
            self._debug(
                "apply: FAIL invariant self-collision before map sampling "
                f"ncon={invariant_ncon} → {invariant_self_reason}"
            )
            raise SpawnCollisionError(
                "The nominal joint/base orientation pose is already self-colliding; "
                "randomizing X/Y/yaw cannot fix it. "
                f"First failure: {invariant_self_reason}"
            )

        # Stage 2: explicit placement is validated exactly once (with optional base-z lift).
        if xy_yaw is not None:
            q_ok, relief_reason, ncon = self._seek_vertical_relief(model, qpos_candidate)
            if q_ok is not None:
                self._debug(f"apply: OK explicit pose; MuJoCo ncon={ncon}")
                return finalize(q_ok)
            self._debug(
                f"apply: FAIL explicit xy_yaw={np.asarray(xy_yaw, dtype=float).tolist()} "
                f"ncon={ncon} → {relief_reason}"
            )
            raise SpawnCollisionError(
                f"Provided xy_yaw produces forbidden contacts (after vertical relief): {relief_reason}"
            )

        # Stage 3: sample world placement and accept the first contact-valid proposal.
        n_reject = 0
        reject_buckets = {
            "self": 0,
            "env_foot_penetration": 0,
            "env_non_foot": 0,
            "other": 0,
        }
        last_reason: str | None = None

        for attempt in range(1, self.max_spawn_attempts + 1):
            q_try = self.build_qpos(p0, quat0, q0, xy_yaw=None)
            q_ok, fail_reason, ncon = self._seek_vertical_relief(model, q_try)
            if q_ok is not None:
                self._debug(
                    f"apply: ACCEPT rejects={n_reject} tries={attempt} "
                    f"base_xyz=({float(q_ok[0]):.3f},{float(q_ok[1]):.3f},{float(q_ok[2]):.3f}) "
                    f"ncon={ncon}; reject_breakdown={reject_buckets}"
                )
                return finalize(q_ok)

            n_reject += 1
            last_reason = fail_reason
            if fail_reason is not None:
                reject_buckets[self._reason_bucket(fail_reason)] += 1

            if self.verbose:
                if n_reject <= 6:
                    self._debug(
                        f"apply: reject #{n_reject} (attempt={attempt}) "
                        f"base_xyz=({float(q_try[0]):.3f},{float(q_try[1]):.3f},{float(q_try[2]):.3f}) "
                        f"ncon={ncon} → {fail_reason}"
                    )
                elif attempt % self.verbose_progress_every == 0:
                    self._debug(
                        f"apply: still sampling attempts={attempt} rejects={n_reject} "
                        f"last_base_xyz=({float(q_try[0]):.3f},{float(q_try[1]):.3f},{float(q_try[2]):.3f}) "
                        f"ncon={ncon}; reject_breakdown={reject_buckets}; "
                        f"last_reason={last_reason}"
                    )

        self._debug(
            f"apply: random sampling exhausted tries={self.max_spawn_attempts} rejects={n_reject}; "
            f"reject_breakdown={reject_buckets}; last_reason={last_reason}; "
            f"next on_collision_exhausted={self.on_collision_exhausted!r}"
        )

        if self.on_collision_exhausted == "origin":
            xc = 0.5 * (float(self.region.x[0]) + float(self.region.x[1]))
            yc = 0.5 * (float(self.region.y[0]) + float(self.region.y[1]))
            yaw_c = float(self.rng.uniform(self.region.yaw[0], self.region.yaw[1]))
            origin_q = self.build_qpos(p0, quat0, q0, xy_yaw=(xc, yc, yaw_c))
            q_ok, origin_fail, ncon = self._seek_vertical_relief(model, origin_q)
            if q_ok is not None:
                self._debug(
                    f"apply: ACCEPT origin fallback base_xyz=({float(q_ok[0]):.3f},{float(q_ok[1]):.3f},"
                    f"{float(q_ok[2]):.3f}) yaw={yaw_c:.4f} ncon={ncon}"
                )
                return finalize(q_ok)
            self._debug(f"apply: FAIL origin fallback ncon={ncon} → {origin_fail}")

        raise SpawnCollisionError(
            f"No collision-free spawn within {self.max_spawn_attempts} attempts "
            f"(rejects={n_reject}, breakdown={reject_buckets}, last_reason={last_reason!r}). "
            "Inspect whether p0[2] is too low, foot geom names are correct, the region is mostly occupied, "
            "or increase ``foot_relief_max`` / set ``try_foot_vertical_relief=True``. "
            "Set verbose=True for per-attempt traces."
        )

    def apply_to_data(self, model, data, p0, quat0, q0, *, xy_yaw=None, forward=True):

        def finalize(qvec):
            data.qpos[:] = np.asarray(qvec, dtype=np.float64).reshape(model.nq)
            if self.reset_velocities:
                data.qvel[:] = 0.0
            if forward:
                mujoco.mj_forward(model, data)
            return np.asarray(data.qpos.copy(), dtype=np.float64)

        qpos_candidate = self.build_qpos(p0, quat0, q0, xy_yaw=xy_yaw)

        if not self.check_collisions:
            return finalize(qpos_candidate)

        # Stage 1: self-collision (invariant to XY/yaw)
        p_nom = np.asarray(p0, dtype=np.float64).reshape(3)
        qpos_inv = self.build_qpos(p0, quat0, q0, xy_yaw=(float(p_nom[0]), float(p_nom[1]), 0.0))
        reason, _ = self._evaluate_self_collision_only(model, qpos_inv)
        if reason is not None:
            raise SpawnCollisionError(f"Nominal pose is self-colliding: {reason}")

        # Stage 2: explicit xy_yaw — validate once with vertical relief
        if xy_yaw is not None:
            q_ok, reason, _ = self._seek_vertical_relief(model, qpos_candidate)
            if q_ok is not None:
                return finalize(q_ok)
            raise SpawnCollisionError(f"Explicit xy_yaw failed after relief: {reason}")

        # Stage 3: random sampling with rejection
        for _ in range(self.max_spawn_attempts):
            q_try = self.build_qpos(p0, quat0, q0)
            q_ok, _, _ = self._seek_vertical_relief(model, q_try)
            if q_ok is not None:
                return finalize(q_ok)

        raise SpawnCollisionError(
            f"No collision-free spawn within {self.max_spawn_attempts} attempts."
        )


