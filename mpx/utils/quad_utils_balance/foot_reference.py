"""
Foot-reference helpers for reduced-support quadruped balance (tripod mode).

This module now supports two usage styles:

1. Backward-compatible function calls (existing call sites keep working).
2. A class-based API via :class:`FootReferenceManager` with settings from
   ``mpx.config.sim_config.config_foot_ref_config``.

Typical class-based use::

    from mpx.config.sim_config.config_foot_ref_config import foot_ref_config
    from mpx.utils.quadruped_dyn_models.foot_reference import FootReferenceManager

    foot_ref = FootReferenceManager(foot_ref_config)
    markers = foot_ref.attach_desired_foot_markers(env.viewer, n_contact=4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from mpx.config.sim_config.config_foot_ref_config import (
    FootReferenceConfig,
    foot_ref_config,
    RandomSwingFootConfig,
    random_swing_foot_config,
)
from mpx.utils.simulation_utils.sim_utils import alloc_decor_geom
from mpx.utils.math_utils.rotation import yaw_rotation_from_quat


# Backward-compatible aliases (prefer ``foot_ref_config`` and ``FootReferenceManager``).
DEFAULT_FOOT_REF_COLORS_RGBA: tuple[np.ndarray, ...] = foot_ref_config.foot_ref_colors_rgba
DEFAULT_FOOT_REF_SPHERE_RADIUS = foot_ref_config.foot_ref_sphere_radius
DEFAULT_SHOW_DESIRED_FOOT_MARKERS = foot_ref_config.show_desired_foot_markers
DEFAULT_SWING_GOAL_RADIUS = foot_ref_config.swing_goal_radius
DEFAULT_SWING_GOAL_COLOR_RGBA = foot_ref_config.swing_goal_color_rgba
DEFAULT_SHOW_SWING_FOOT_GOAL = foot_ref_config.show_swing_foot_goal
DEFAULT_SHOW_SWING_WORKSPACE_MARKER = foot_ref_config.show_swing_workspace_marker
DEFAULT_SWING_WORKSPACE_RADII_XYZ = foot_ref_config.swing_workspace_radii_xyz
DEFAULT_SWING_WORKSPACE_COLOR_RGBA = foot_ref_config.swing_workspace_color_rgba
DEFAULT_FOOT_REF_SAMPLING_MODE: Literal["box", "ellipsoid"] = (
    foot_ref_config.sampling_mode
)




def is_tripod_contact_mask(contact_mask) -> jnp.ndarray:
    """True when exactly three feet are in nominal stance (tripod presets)."""
    return jnp.sum(contact_mask.astype(jnp.float32)) == 3.0


def _broadcast_sigma(sigma, n_contact: int, dtype):
    sigma = jnp.asarray(sigma, dtype=dtype)
    if sigma.shape == ():
        return jnp.full((3 * n_contact,), sigma)
    if sigma.shape == (3,):
        return jnp.tile(sigma, n_contact)
    return sigma


def _sigma_to_per_foot_xyz(sigma, n_contact: int, dtype):
    sigma = _broadcast_sigma(sigma, n_contact, dtype)
    sigma = jnp.asarray(sigma, dtype=dtype).reshape(-1)
    if sigma.size != 3 * n_contact:
        raise ValueError(
            f"sigma must broadcast to shape {(3 * n_contact,)}, got {sigma.shape}"
        )
    return sigma.reshape(n_contact, 3)


def sample_nominal_foot_offsets_base_box(key, foot0, n_contact: int, sigma):
    """
    Uniform perturbation of nominal base-frame foot offsets.

    ``foot0`` is the flat ``(3 * n_contact,)`` vector (FL, FR, RL, RR × XYZ), same layout as ``p_legs0``.
    """
    sigma = _broadcast_sigma(sigma, n_contact, foot0.dtype)
    noise = jax.random.uniform(
        key, (3 * n_contact,), minval=-sigma, maxval=sigma, dtype=foot0.dtype
    )
    return foot0 + noise


def sample_nominal_foot_offsets_base_ellipsoid(key, foot0, n_contact: int, sigma):
    """
    Uniform perturbation inside per-foot XYZ ellipsoids in base frame.

    ``sigma`` defines ellipsoid radii per axis (broadcasted exactly like box sampling).
    """
    foot0 = jnp.asarray(foot0)
    radii = _sigma_to_per_foot_xyz(sigma, n_contact, foot0.dtype)
    key_dir, key_rad = jax.random.split(key)
    direction_raw = jax.random.normal(key_dir, (n_contact, 3), dtype=foot0.dtype)
    norm = jnp.linalg.norm(direction_raw, axis=1, keepdims=True)
    direction = direction_raw / jnp.maximum(norm, jnp.asarray(1e-9, dtype=foot0.dtype))
    unit_radius = jax.random.uniform(
        key_rad, (n_contact, 1), minval=0.0, maxval=1.0, dtype=foot0.dtype
    ) ** (1.0 / 3.0)
    local_delta = direction * unit_radius * radii
    return foot0.reshape(n_contact, 3).reshape(-1) + local_delta.reshape(-1)


def sample_nominal_foot_offsets_base(
    key,
    foot0,
    n_contact: int,
    sigma,
    *,
    sampling_mode: Literal["box", "ellipsoid"] = DEFAULT_FOOT_REF_SAMPLING_MODE,
):
    """Dispatch sampling by mode (``box`` or ``ellipsoid``)."""
    if sampling_mode == "box":
        return sample_nominal_foot_offsets_base_box(key, foot0, n_contact, sigma)
    if sampling_mode == "ellipsoid":
        return sample_nominal_foot_offsets_base_ellipsoid(key, foot0, n_contact, sigma)
    raise ValueError(
        f"Unsupported sampling_mode={sampling_mode!r}; expected 'box' or 'ellipsoid'."
    )


def base_frame_feet_to_world(p, quat, foot_base_flat, n_contact: int):
    """Project base-frame foot offsets to world frame (same convention as locomotion ref gen)."""
    ryaw = yaw_rotation_from_quat(quat)
    blk = jax.scipy.linalg.block_diag(*([ryaw] * n_contact))
    return jnp.tile(p, n_contact) + foot_base_flat @ blk.T


def tripod_foot_reference_world(
    key,
    p,
    quat,
    foot0,
    n_contact: int,
    sigma,
    *,
    sampling_mode: Literal["box", "ellipsoid"] = DEFAULT_FOOT_REF_SAMPLING_MODE,
    measured_foot=None,
    contact_mask=None,
):
    """
    World-frame foot reference from a random sample around nominal base-frame poses.

    All feet use the sampled nominal pose in the base frame (projected with yaw). Stance legs
    are tracked in the MPC cost; swing legs are not (``contact_map`` in the balance objective),
    but ``p_leg_ref`` for the swing leg is still this nominal XYZ — the desired swing target.
    """
    del measured_foot, contact_mask  # kept for call-site compatibility
    foot_sample = sample_nominal_foot_offsets_base(
        key,
        foot0,
        n_contact,
        sigma,
        sampling_mode=sampling_mode,
    )
    return base_frame_feet_to_world(p, quat, foot_sample, n_contact)


def swing_foot_anchor_from_target(
    current_anchor: np.ndarray,
    leg_idx: int,
    target_xyz_world: np.ndarray,
) -> np.ndarray:
    """
    Return an updated world-frame foot anchor with the swing leg redirected to a specific target.

    Stance legs keep their positions from ``current_anchor`` unchanged; only the slice
    corresponding to ``leg_idx`` is replaced with ``target_xyz_world``.

    Args:
        current_anchor: Flat ``(3 * n_contact,)`` world-frame anchor vector (FL, FR, RL, RR × XYZ).
        leg_idx:        Index of the swing leg (0-based).
        target_xyz_world: Desired world-frame XYZ landing position for the swing foot.

    Returns:
        Updated flat ``(3 * n_contact,)`` anchor vector.
    """
    anchor = np.asarray(current_anchor, dtype=np.float64).copy()
    anchor[3 * leg_idx : 3 * leg_idx + 3] = np.asarray(target_xyz_world, dtype=np.float64).reshape(3)
    return anchor


def foot_target_foot_local_to_world(
    foot_current_world: np.ndarray,
    quat: np.ndarray,
    xyz_foot_local: np.ndarray,
) -> np.ndarray:
    """
    Convert a foot-local frame target to a world-frame target.

    The foot-local frame has its origin at the current foot position in the world and
    its axes yaw-aligned with the robot base (same yaw rotation as the base frame, but
    translated to the foot).  This means:

    - X points forward along the robot heading
    - Y points left
    - Z points up

    The conversion is:  ``xyz_world = foot_current_world + R_yaw @ xyz_foot_local``

    Args:
        foot_current_world: Current foot position in the world frame, shape ``(3,)``.
        quat:               Base orientation quaternion ``[w, x, y, z]``, shape ``(4,)``.
        xyz_foot_local:     Desired target expressed in the foot-local frame, shape ``(3,)``.

    Returns:
        Target position in the world frame, shape ``(3,)`` as a float64 numpy array.
    """
    ryaw = yaw_rotation_from_quat(jnp.asarray(quat, dtype=jnp.float32))
    offset = np.asarray(ryaw @ jnp.asarray(xyz_foot_local, dtype=jnp.float32), dtype=np.float64)
    return np.asarray(foot_current_world, dtype=np.float64).reshape(3) + offset


def base_yaw_offset_to_world(
    base_pos_world: np.ndarray,
    quat: np.ndarray,
    xyz_base: np.ndarray,
) -> np.ndarray:
    """
    Convert a yaw-aligned base-frame offset to a world-frame point.

    The base frame has its origin at the robot base (``qpos[:3]``) and axes:
    X forward, Y left, Z up (yaw from ``quat`` only, roll/pitch ignored).

    ``xyz_world = base_pos_world + R_yaw @ xyz_base``
    """
    ryaw = yaw_rotation_from_quat(jnp.asarray(quat, dtype=jnp.float32))
    offset = np.asarray(ryaw @ jnp.asarray(xyz_base, dtype=jnp.float32), dtype=np.float64)
    return np.asarray(base_pos_world, dtype=np.float64).reshape(3) + offset


def swing_goal_xyz_from_foot_ref(
    foot_ref_flat: np.ndarray, contact_mask: np.ndarray, n_contact: int
) -> tuple[int, np.ndarray | None]:
    """Return ``(leg_index, xyz)`` for the first nominal swing leg (mask 0), else ``(-1, None)``."""
    foot_ref_flat = np.asarray(foot_ref_flat, dtype=np.float64).reshape(-1)
    mask = np.asarray(contact_mask, dtype=np.float64).reshape(-1)
    for i in range(n_contact):
        if mask[i] < 0.5:
            return i, foot_ref_flat[3 * i : 3 * i + 3].copy()
    return -1, None


@dataclass
class SwingFootGoalMarker:
    """
    Single fixed viewer marker at the swing-foot placement goal (world XYZ).

    Position is set only via :meth:`set_goal` (MPC reset / resample), not each control step.
    """

    geom_id: int = -1
    leg_index: int = -1
    goal_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    radius: float = DEFAULT_SWING_GOAL_RADIUS
    color: np.ndarray = field(
        default_factory=lambda: DEFAULT_SWING_GOAL_COLOR_RGBA.copy()
    )

    @classmethod
    def create(
        cls,
        viewer,
        *,
        radius: float = DEFAULT_SWING_GOAL_RADIUS,
        color: np.ndarray | None = None,
    ) -> SwingFootGoalMarker:
        if viewer is None:
            raise ValueError("viewer is None; call env.render() before attach_swing_foot_goal_marker")
        marker = cls(
            radius=radius,
            color=DEFAULT_SWING_GOAL_COLOR_RGBA.copy()
            if color is None
            else np.asarray(color, dtype=np.float64).reshape(4),
        )
        marker.geom_id = alloc_decor_geom(viewer)
        marker.draw(viewer, sync=False)
        return marker

    def set_goal(self, leg_index: int, goal_xyz: np.ndarray) -> None:
        """Fix goal to this world position until the next resample."""
        self.leg_index = int(leg_index)
        self.goal_xyz = np.asarray(goal_xyz, dtype=np.float64).reshape(3).copy()

    def clear(self) -> None:
        self.leg_index = -1

    def draw(self, viewer, *, sync: bool = True) -> None:
        if viewer is None or self.geom_id < 0:
            return
        geom = viewer.user_scn.geoms[self.geom_id]
        if self.leg_index < 0:
            rgba = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            pos = np.zeros(3, dtype=np.float64)
            emission = 0.0
        else:
            rgba = np.asarray(self.color, dtype=np.float64).reshape(4)
            pos = self.goal_xyz
            emission = 0.4
        r = float(self.radius)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([r, r, r], dtype=np.float64),
            pos=pos,
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=rgba,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.segid = -1
        geom.objid = -1
        geom.emission = emission
        if sync:
            viewer.sync()


@dataclass
class SwingWorkspaceMarker:
    """
    Ellipsoidal viewer marker for swing-foot reachable region in world frame.

    The marker center is typically set to the current swing-foot goal (or any desired
    world-frame center), and radii define admissible XYZ variation around that center.
    """

    geom_id: int = -1
    leg_index: int = -1
    center_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    radii_xyz: np.ndarray = field(
        default_factory=lambda: np.asarray(DEFAULT_SWING_WORKSPACE_RADII_XYZ, dtype=np.float64)
    )
    color: np.ndarray = field(
        default_factory=lambda: np.asarray(DEFAULT_SWING_WORKSPACE_COLOR_RGBA, dtype=np.float64)
    )

    @classmethod
    def create(
        cls,
        viewer,
        *,
        radii_xyz: Sequence[float] = DEFAULT_SWING_WORKSPACE_RADII_XYZ,
        color: np.ndarray | None = None,
    ) -> SwingWorkspaceMarker:
        if viewer is None:
            raise ValueError(
                "viewer is None; call env.render() before attach_swing_workspace_marker"
            )
        marker = cls(
            radii_xyz=np.asarray(radii_xyz, dtype=np.float64).reshape(3),
            color=np.asarray(DEFAULT_SWING_WORKSPACE_COLOR_RGBA, dtype=np.float64)
            if color is None
            else np.asarray(color, dtype=np.float64).reshape(4),
        )
        marker.geom_id = alloc_decor_geom(viewer)
        marker.draw(viewer, sync=False)
        return marker

    def set_center(self, leg_index: int, center_xyz: np.ndarray) -> None:
        self.leg_index = int(leg_index)
        self.center_xyz = np.asarray(center_xyz, dtype=np.float64).reshape(3).copy()

    def clear(self) -> None:
        self.leg_index = -1

    def draw(self, viewer, *, sync: bool = True) -> None:
        if viewer is None or self.geom_id < 0:
            return
        geom = viewer.user_scn.geoms[self.geom_id]
        if self.leg_index < 0:
            rgba = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            pos = np.zeros(3, dtype=np.float64)
            emission = 0.0
        else:
            rgba = np.asarray(self.color, dtype=np.float64).reshape(4)
            pos = self.center_xyz
            emission = 0.2
        rx, ry, rz = np.asarray(self.radii_xyz, dtype=np.float64).reshape(3)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=np.array([rx, ry, rz], dtype=np.float64),
            pos=pos,
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=rgba,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.segid = -1
        geom.objid = -1
        geom.emission = emission
        if sync:
            viewer.sync()


@dataclass
class DesiredFootMarkers:
    """MuJoCo spheres for desired foot XYZ in ``p_leg_ref`` (by default: nominal swing leg only)."""

    geom_ids: list[int] = field(default_factory=list)
    radius: float = DEFAULT_FOOT_REF_SPHERE_RADIUS
    colors: Sequence[np.ndarray] = field(default_factory=lambda: DEFAULT_FOOT_REF_COLORS_RGBA)

    @classmethod
    def create(
        cls,
        viewer,
        n_contact: int = 4,
        *,
        radius: float = DEFAULT_FOOT_REF_SPHERE_RADIUS,
        colors: Sequence[np.ndarray] | None = None,
    ) -> DesiredFootMarkers:
        """Allocate decorative geoms (call after ``env.render()`` so viewer exists)."""
        if viewer is None:
            raise ValueError("viewer is None; call env.render() before attach_desired_foot_markers")
        palette = tuple(colors) if colors is not None else DEFAULT_FOOT_REF_COLORS_RGBA
        if len(palette) < n_contact:
            raise ValueError(f"Need {n_contact} foot colors, got {len(palette)}")
        markers = cls(radius=radius, colors=palette)
        markers.geom_ids = [alloc_decor_geom(viewer) for _ in range(n_contact)]
        markers.draw(viewer, np.zeros(3 * n_contact, dtype=np.float64), sync=False)
        return markers

    def draw(
        self,
        viewer,
        foot_ref_flat: np.ndarray,
        *,
        contact_mask: np.ndarray | None = None,
        swing_only: bool = True,
        sync: bool = True,
    ) -> None:
        """
        Update markers from flat ``(3 * n_contact,)`` ``p_leg_ref`` (FL, FR, RL, RR).

        With ``swing_only=True`` and ``contact_mask`` (1 = stance, 0 = swing), only the
        nominal swing leg(s) are drawn; stance markers are hidden.
        """
        if viewer is None or not self.geom_ids:
            return
        foot_ref_flat = np.asarray(foot_ref_flat, dtype=np.float64).reshape(-1)
        n = len(self.geom_ids)
        if foot_ref_flat.size < 3 * n:
            return
        mask = None
        if contact_mask is not None:
            mask = np.asarray(contact_mask, dtype=np.float64).reshape(-1)
        r = float(self.radius)
        for i, gid in enumerate(self.geom_ids):
            if swing_only and mask is not None and mask[i] > 0.5:
                rgba = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                pos = foot_ref_flat[3 * i : 3 * i + 3]
            else:
                rgba = np.asarray(self.colors[i], dtype=np.float64).reshape(4)
                pos = foot_ref_flat[3 * i : 3 * i + 3]
            geom = viewer.user_scn.geoms[gid]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([r, r, r], dtype=np.float64),
                pos=pos,
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=rgba,
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.segid = -1
            geom.objid = -1
            geom.emission = 0.25 if rgba[3] > 0.01 else 0.0
        if sync:
            viewer.sync()


# Backward-compatible alias
FootReferenceMarkers = DesiredFootMarkers


def swing_foot_arrival_distance(
    measured_world: np.ndarray,
    target_world: np.ndarray,
    *,
    foot_geom_radius_m: float = 0.0,
) -> tuple[float, float]:
    """Return ``(xy_error, z_error)`` from measured foot geom center to target."""
    diff = (
        np.asarray(measured_world, dtype=np.float64).reshape(3)
        - np.asarray(target_world, dtype=np.float64).reshape(3)
    )
    xy = float(np.linalg.norm(diff[:2]))
    z = max(0.0, abs(float(diff[2])) - foot_geom_radius_m)
    return xy, z


def swing_foot_at_goal(
    measured_world: np.ndarray,
    target_world: np.ndarray,
    cfg: RandomSwingFootConfig,
) -> tuple[bool, float, float]:
    """True when measured swing foot is close enough to the goal."""
    xy, z = swing_foot_arrival_distance(
        measured_world, target_world, foot_geom_radius_m=cfg.foot_geom_radius_m
    )
    on_ground = (
        float(np.asarray(measured_world).reshape(3)[2]) < cfg.ground_contact_z_max
        and float(np.asarray(target_world).reshape(3)[2]) < cfg.ground_contact_z_max
    )
    if on_ground:
        arrived = xy <= cfg.arrival_tolerance_xy_m and z <= cfg.arrival_tolerance_z_m
    else:
        arrived = float(np.hypot(xy, z)) <= cfg.arrival_tolerance_xy_m
    return arrived, xy, z


def random_swing_bounds_box_pose(
    base_pos_world: np.ndarray,
    base_quat: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-frame center, half-extents, and rotation for the sampling AABB.

    The box is aligned with the yaw-aligned **base frame** (origin at ``base_pos_world``).
    Its faces correspond to ``x_bounds``, ``y_bounds``, and ``z_bounds``.
    """
    center_base = np.array(
        [
            0.5 * (x_bounds[0] + x_bounds[1]),
            0.5 * (y_bounds[0] + y_bounds[1]),
            0.5 * (z_bounds[0] + z_bounds[1]),
        ],
        dtype=np.float64,
    )
    half_base = np.array(
        [
            0.5 * (x_bounds[1] - x_bounds[0]),
            0.5 * (y_bounds[1] - y_bounds[0]),
            0.5 * (z_bounds[1] - z_bounds[0]),
        ],
        dtype=np.float64,
    )
    center_world = base_yaw_offset_to_world(base_pos_world, base_quat, center_base)
    ryaw = np.asarray(
        yaw_rotation_from_quat(jnp.asarray(base_quat, dtype=jnp.float32)),
        dtype=np.float64,
    )
    return center_world, half_base, ryaw.reshape(9)


@dataclass
class RandomSwingBoundsBoxMarker:
    """Semi-transparent box for visualizing random swing-foot XYZ bounds in base frame."""

    geom_id: int = -1
    base_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    base_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    color: np.ndarray = field(
        default_factory=lambda: np.array([0.15, 0.95, 0.35, 0.22], dtype=np.float64)
    )
    active: bool = False

    @classmethod
    def create(
        cls,
        viewer,
        *,
        color: np.ndarray | None = None,
    ) -> RandomSwingBoundsBoxMarker:
        if viewer is None:
            raise ValueError(
                "viewer is None; call env.render() before attach_random_swing_bounds_box"
            )
        marker = cls(
            color=np.array([0.15, 0.95, 0.35, 0.22], dtype=np.float64)
            if color is None
            else np.asarray(color, dtype=np.float64).reshape(4),
        )
        marker.geom_id = alloc_decor_geom(viewer)
        marker.draw(viewer, random_swing_foot_config, sync=False)
        return marker

    def set_frame(self, base_pos_world: np.ndarray, base_quat: np.ndarray) -> None:
        self.base_pos_world = np.asarray(base_pos_world, dtype=np.float64).reshape(3).copy()
        self.base_quat = np.asarray(base_quat, dtype=np.float64).reshape(4).copy()
        self.active = True

    def clear(self) -> None:
        self.active = False

    def draw(
        self,
        viewer,
        cfg: RandomSwingFootConfig,
        *,
        sync: bool = True,
    ) -> None:
        if viewer is None or self.geom_id < 0:
            return
        geom = viewer.user_scn.geoms[self.geom_id]
        if not self.active or not cfg.show_bounds_box:
            rgba = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            pos = np.zeros(3, dtype=np.float64)
            size = np.ones(3, dtype=np.float64) * 1e-6
            mat = np.eye(3, dtype=np.float64).reshape(9)
            emission = 0.0
        else:
            center, half, mat = random_swing_bounds_box_pose(
                self.base_pos_world,
                self.base_quat,
                cfg.x_bounds,
                cfg.y_bounds,
                cfg.z_bounds,
            )
            rgba = np.asarray(self.color, dtype=np.float64).reshape(4)
            pos = center
            size = np.maximum(half, 1e-4)
            emission = 0.15
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            mat=mat,
            rgba=rgba,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.segid = -1
        geom.objid = -1
        geom.emission = emission
        if sync:
            viewer.sync()


class RandomSwingFootSampler:
    """Samples a random swing-foot world-frame target within configurable XYZ bounds.

    Bounds are expressed in the **yaw-aligned base frame** (origin at ``qpos[:3]``):

    - X : forward along robot heading.
    - Y : lateral (positive = left).
    - Z : vertical (positive = up from base).

    A uniform sample ``[x, y, z]`` inside the bounds box is mapped to world frame via
    ``base_pos + R_yaw @ [x, y, z]``.

    Keyboard shortcuts in the simulator (GLFW):
      R  — sample a new random swing target immediately.
      N  — toggle auto-randomise on every respawn on/off.
    """

    def __init__(self, cfg: RandomSwingFootConfig | None = None) -> None:
        self.cfg: RandomSwingFootConfig = cfg if cfg is not None else random_swing_foot_config
        self._enabled: bool = self.cfg.enabled
        self._rng: np.random.Generator = np.random.default_rng()
        self._arrival_cooldown_steps: int = 0
        self._arrival_hold_steps: int = 0

    # ── public properties ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True when random mode is active."""
        return self._enabled

    @property
    def resample_on_respawn(self) -> bool:
        """True when a new random target should be drawn on every respawn."""
        return self._enabled and self.cfg.resample_on_respawn

    @property
    def resample_on_arrival(self) -> bool:
        """True when a new random target should be drawn after reaching the goal."""
        return self._enabled and self.cfg.resample_on_arrival

    # ── control ─────────────────────────────────────────────────────────────

    def reset_arrival_state(self) -> None:
        """Clear arrival-resample cooldown and hold counter (call on respawn / manual resample)."""
        self._arrival_cooldown_steps = 0
        self._arrival_hold_steps = 0

    def toggle(self) -> bool:
        """Toggle random mode on/off.  Returns the **new** enabled state."""
        self._enabled = not self._enabled
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample_offset_base(self) -> np.ndarray:
        """Draw a random XYZ sample inside ``cfg`` bounds (base frame)."""
        return np.array([
            self._rng.uniform(*self.cfg.x_bounds),
            self._rng.uniform(*self.cfg.y_bounds),
            self._rng.uniform(*self.cfg.z_bounds),
        ], dtype=np.float64)

    def sample_offset_local(self) -> np.ndarray:
        """Alias for :meth:`sample_offset_base` (backward compatibility)."""
        return self.sample_offset_base()

    def sample_swing_world(
        self,
        base_pos_world: np.ndarray,
        base_quat: np.ndarray,
    ) -> np.ndarray:
        """Return a random world-frame swing-foot target from base-frame bounds.

        Args:
            base_pos_world: Robot base position in world frame ``qpos[:3]``.
            base_quat:      Base orientation quaternion ``[w, x, y, z]`` ``qpos[3:7]``.

        Returns:
            New world-frame target position ``(3,)`` as float64.
        """
        xyz_base = self.sample_offset_base()
        return base_yaw_offset_to_world(base_pos_world, base_quat, xyz_base)

    def try_resample_on_arrival(
        self,
        foot_anchor: np.ndarray,
        swing_leg_idx: int,
        measured_swing_world: np.ndarray,
        base_pos_world: np.ndarray,
        base_quat: np.ndarray,
        *,
        sim_dt: float,
        cooldown_steps: int | None = None,
        hold_steps: int | None = None,
    ) -> tuple[np.ndarray, int, int, bool]:
        """Resample swing target if the foot has reached the current goal.

        The next target is sampled uniformly in the base-frame bounds box.

        Args:
            foot_anchor: Flat ``(3 * n_contact,)`` world-frame anchor.
            swing_leg_idx: Index of the swing leg.
            measured_swing_world: Current measured swing foot XYZ in world frame.
            base_pos_world:     Robot base position ``qpos[:3]`` (sampling frame origin).
            base_quat: Base orientation ``[w, x, y, z]``.
            sim_dt: Simulation timestep [s] (for cooldown conversion).
            cooldown_steps: Optional per-env cooldown override (multi-env).
            hold_steps: Optional per-env hold-step override (multi-env).

        Returns:
            ``(updated_foot_anchor, new_cooldown_steps, new_hold_steps, did_resample)``.
        """
        if not self.resample_on_arrival:
            cd = 0 if cooldown_steps is None else int(cooldown_steps)
            hs = 0 if hold_steps is None else int(hold_steps)
            return np.asarray(foot_anchor, dtype=np.float64), cd, hs, False

        if cooldown_steps is None:
            if self._arrival_cooldown_steps > 0:
                self._arrival_cooldown_steps -= 1
                return (
                    np.asarray(foot_anchor, dtype=np.float64),
                    self._arrival_cooldown_steps,
                    self._arrival_hold_steps,
                    False,
                )
            cd_after = 0
            hold_after = self._arrival_hold_steps
        elif cooldown_steps > 0:
            hs = 0 if hold_steps is None else int(hold_steps)
            return np.asarray(foot_anchor, dtype=np.float64), cooldown_steps - 1, hs, False
        else:
            cd_after = 0
            hold_after = 0 if hold_steps is None else int(hold_steps)

        target = np.asarray(
            foot_anchor[3 * swing_leg_idx : 3 * swing_leg_idx + 3], dtype=np.float64
        )
        measured = np.asarray(measured_swing_world, dtype=np.float64).reshape(3)
        arrived, xy_err, z_err = swing_foot_at_goal(measured, target, self.cfg)

        if not arrived:
            if cooldown_steps is None:
                self._arrival_hold_steps = 0
            hold_after = 0
            return np.asarray(foot_anchor, dtype=np.float64), 0, hold_after, False

        if cooldown_steps is None:
            self._arrival_hold_steps += 1
            hold_after = self._arrival_hold_steps
        else:
            hold_after = int(hold_steps) + 1

        if hold_after < self.cfg.arrival_hold_steps:
            return np.asarray(foot_anchor, dtype=np.float64), 0, hold_after, False

        # Sample next target in base-frame bounds.
        new_target = self.sample_swing_world(base_pos_world, base_quat)
        updated = swing_foot_anchor_from_target(foot_anchor, swing_leg_idx, new_target)
        cooldown = max(1, int(round(self.cfg.resample_cooldown_s / sim_dt)))
        if cooldown_steps is None:
            self._arrival_cooldown_steps = cooldown
            self._arrival_hold_steps = 0
            cd_after = cooldown
            hold_after = 0
        else:
            cd_after = cooldown
            hold_after = 0
        return updated, cd_after, hold_after, True

    def bounds_summary(self) -> str:
        """Short human-readable summary of the current bounds."""
        c = self.cfg
        return (
            f"x∈[{c.x_bounds[0]:+.2f}, {c.x_bounds[1]:+.2f}]  "
            f"y∈[{c.y_bounds[0]:+.2f}, {c.y_bounds[1]:+.2f}]  "
            f"z∈[{c.z_bounds[0]:+.2f}, {c.z_bounds[1]:+.2f}]"
        )

    def attach_bounds_box_marker(
        self,
        viewer,
        *,
        enabled: bool | None = None,
        color: np.ndarray | None = None,
    ) -> RandomSwingBoundsBoxMarker | None:
        """Create a passive-viewer box for ``x_bounds`` / ``y_bounds`` / ``z_bounds``."""
        if enabled is None:
            enabled = self.cfg.show_bounds_box
        if not enabled:
            return None
        marker_color = self.cfg.bounds_box_color_rgba if color is None else color
        return RandomSwingBoundsBoxMarker.create(viewer, color=marker_color)

    def update_bounds_box_marker(
        self,
        marker: RandomSwingBoundsBoxMarker | None,
        viewer,
        base_pos_world: np.ndarray,
        base_quat: np.ndarray,
        *,
        sync: bool = False,
    ) -> None:
        """Redraw the base-frame sampling box (moves with the robot base)."""
        if marker is None or viewer is None:
            return
        if self.enabled and self.cfg.show_bounds_box:
            marker.set_frame(base_pos_world, base_quat)
        else:
            marker.clear()
        marker.draw(viewer, self.cfg, sync=sync)


@dataclass
class FootReferenceManager:
    """
    Class-based façade for foot-reference sampling and marker visualization.

    Parameters are supplied by :class:`FootReferenceConfig`, typically imported as
    ``foot_ref_config`` from ``config_foot_ref_config.py``.
    """

    config: FootReferenceConfig = field(default_factory=lambda: foot_ref_config)

    def tripod_foot_reference_world(
        self,
        key,
        p,
        quat,
        foot0,
        n_contact: int,
        sigma,
        *,
        sampling_mode: Literal["box", "ellipsoid"] | None = None,
        measured_foot=None,
        contact_mask=None,
    ):
        mode = self.config.sampling_mode if sampling_mode is None else sampling_mode
        return tripod_foot_reference_world(
            key=key,
            p=p,
            quat=quat,
            foot0=foot0,
            n_contact=n_contact,
            sigma=sigma,
            sampling_mode=mode,
            measured_foot=measured_foot,
            contact_mask=contact_mask,
        )

    def nominal_swing_foot_world(
        self,
        p,
        quat,
        foot0,
        n_contact: int,
        swing_leg_idx: int,
    ) -> np.ndarray:
        """Unperturbed nominal swing-foot XYZ in world frame (``p_legs0`` + yaw)."""
        flat = self.tripod_foot_reference_world(
            key=jax.random.PRNGKey(0),
            p=p,
            quat=quat,
            foot0=foot0,
            n_contact=n_contact,
            sigma=np.array([0.0, 0.0, 0.0]),
        )
        arr = np.asarray(flat, dtype=np.float64).reshape(-1)
        return arr[3 * swing_leg_idx : 3 * swing_leg_idx + 3].copy()

    def swing_goal_xyz_from_foot_ref(
        self, foot_ref_flat: np.ndarray, contact_mask: np.ndarray, n_contact: int
    ) -> tuple[int, np.ndarray | None]:
        return swing_goal_xyz_from_foot_ref(foot_ref_flat, contact_mask, n_contact)

    def swing_foot_anchor_from_target(
        self,
        current_anchor: np.ndarray,
        leg_idx: int,
        target_xyz_world: np.ndarray,
    ) -> np.ndarray:
        """Redirect swing leg ``leg_idx`` in ``current_anchor`` to ``target_xyz_world``."""
        return swing_foot_anchor_from_target(current_anchor, leg_idx, target_xyz_world)

    def foot_target_foot_local_to_world(
        self,
        foot_current_world: np.ndarray,
        quat: np.ndarray,
        xyz_foot_local: np.ndarray,
    ) -> np.ndarray:
        """Convert a foot-local frame offset to a world-frame target."""
        return foot_target_foot_local_to_world(foot_current_world, quat, xyz_foot_local)

    def attach_desired_foot_markers(
        self,
        viewer,
        n_contact: int = 4,
        *,
        enabled: bool | None = None,
        radius: float | None = None,
        colors: Sequence[np.ndarray] | None = None,
    ) -> DesiredFootMarkers | None:
        """Create per-foot desired-position spheres in passive viewer."""
        if enabled is None:
            enabled = self.config.show_desired_foot_markers
        if not enabled:
            return None
        marker_radius = self.config.foot_ref_sphere_radius if radius is None else radius
        marker_colors = (
            self.config.foot_ref_colors_rgba if colors is None else tuple(colors)
        )
        return DesiredFootMarkers.create(
            viewer,
            n_contact,
            radius=marker_radius,
            colors=marker_colors,
        )

    def update_desired_foot_markers(
        self,
        markers: DesiredFootMarkers | None,
        viewer,
        mpc,
        contact_mask: np.ndarray | None = None,
        *,
        swing_only: bool = True,
    ) -> None:
        """Show ``mpc.last_foot_ref`` (``p_leg_ref``), optionally hiding stance feet."""
        if markers is None:
            return
        foot_ref = getattr(mpc, "last_foot_ref", None)
        if foot_ref is None:
            return
        if contact_mask is None:
            contact_mask = getattr(
                getattr(mpc, "config", None), "balance_fixed_contact_mask", None
            )
        markers.draw(
            viewer,
            foot_ref,
            contact_mask=contact_mask,
            swing_only=swing_only,
            sync=True,
        )

    def attach_swing_foot_goal_marker(
        self,
        viewer,
        *,
        enabled: bool | None = None,
        radius: float | None = None,
        color: np.ndarray | None = None,
    ) -> SwingFootGoalMarker | None:
        """Create a passive-viewer marker for fixed swing-foot landing goal."""
        if enabled is None:
            enabled = self.config.show_swing_foot_goal
        if not enabled:
            return None
        marker_radius = self.config.swing_goal_radius if radius is None else radius
        marker_color = self.config.swing_goal_color_rgba if color is None else color
        return SwingFootGoalMarker.create(viewer, radius=marker_radius, color=marker_color)

    def update_swing_foot_goal_marker(
        self,
        marker: SwingFootGoalMarker | None,
        viewer,
        mpc,
    ) -> None:
        """Redraw goal from ``mpc.swing_foot_goal_*`` state."""
        if marker is None or viewer is None:
            return
        leg = int(getattr(mpc, "swing_foot_goal_leg", -1))
        if leg < 0:
            marker.clear()
        else:
            xyz = getattr(mpc, "swing_foot_goal_xyz", None)
            if xyz is not None:
                marker.set_goal(leg, xyz)
                if leg < len(self.config.foot_ref_colors_rgba):
                    marker.color = np.asarray(
                        self.config.foot_ref_colors_rgba[leg], dtype=np.float64
                    )
        marker.draw(viewer, sync=True)

    def attach_swing_workspace_marker(
        self,
        viewer,
        *,
        enabled: bool | None = None,
        radii_xyz: Sequence[float] | None = None,
        color: np.ndarray | None = None,
    ) -> SwingWorkspaceMarker | None:
        """Create a passive-viewer ellipsoid for swing-foot reachable XYZ region."""
        if enabled is None:
            enabled = self.config.show_swing_workspace_marker
        if not enabled:
            return None
        marker_radii = (
            self.config.swing_workspace_radii_xyz if radii_xyz is None else radii_xyz
        )
        marker_color = self.config.swing_workspace_color_rgba if color is None else color
        return SwingWorkspaceMarker.create(
            viewer,
            radii_xyz=marker_radii,
            color=marker_color,
        )

    def update_swing_workspace_marker(
        self,
        marker: SwingWorkspaceMarker | None,
        viewer,
        mpc,
    ) -> None:
        """
        Center reachable ellipsoid on swing-goal world XYZ.

        This visualizes admissible XYZ variation around the chosen swing-foot target.
        """
        if marker is None or viewer is None:
            return
        leg = int(getattr(mpc, "swing_foot_goal_leg", -1))
        if leg < 0:
            marker.clear()
        else:
            xyz = getattr(mpc, "swing_foot_goal_xyz", None)
            if xyz is not None:
                marker.set_center(leg, xyz)
                if leg < len(self.config.foot_ref_colors_rgba):
                    leg_rgba = np.asarray(
                        self.config.foot_ref_colors_rgba[leg], dtype=np.float64
                    )
                    marker.color = leg_rgba.copy()
                    marker.color[3] = min(marker.color[3], 0.28)
        marker.draw(viewer, sync=True)


# Default manager for backward-compatible function API.
default_foot_reference_manager = FootReferenceManager(foot_ref_config)


def attach_desired_foot_markers(
    viewer,
    n_contact: int = 4,
    *,
    enabled: bool = DEFAULT_SHOW_DESIRED_FOOT_MARKERS,
    radius: float = DEFAULT_FOOT_REF_SPHERE_RADIUS,
    colors: Sequence[np.ndarray] | None = None,
) -> DesiredFootMarkers | None:
    """Compatibility wrapper over ``default_foot_reference_manager``."""
    return default_foot_reference_manager.attach_desired_foot_markers(
        viewer,
        n_contact=n_contact,
        enabled=enabled,
        radius=radius,
        colors=colors,
    )


def update_desired_foot_markers(
    markers: DesiredFootMarkers | None,
    viewer,
    mpc,
    contact_mask: np.ndarray | None = None,
    *,
    swing_only: bool = True,
) -> None:
    """Compatibility wrapper over ``default_foot_reference_manager``."""
    default_foot_reference_manager.update_desired_foot_markers(
        markers,
        viewer,
        mpc,
        contact_mask=contact_mask,
        swing_only=swing_only,
    )


def attach_swing_foot_goal_marker(
    viewer,
    *,
    enabled: bool = DEFAULT_SHOW_SWING_FOOT_GOAL,
    radius: float = DEFAULT_SWING_GOAL_RADIUS,
    color: np.ndarray | None = None,
) -> SwingFootGoalMarker | None:
    """Compatibility wrapper over ``default_foot_reference_manager``."""
    return default_foot_reference_manager.attach_swing_foot_goal_marker(
        viewer,
        enabled=enabled,
        radius=radius,
        color=color,
    )


def update_swing_foot_goal_marker(
    marker: SwingFootGoalMarker | None,
    viewer,
    mpc,
) -> None:
    """Compatibility wrapper over ``default_foot_reference_manager``."""
    default_foot_reference_manager.update_swing_foot_goal_marker(marker, viewer, mpc)


def attach_swing_workspace_marker(
    viewer,
    *,
    enabled: bool = DEFAULT_SHOW_SWING_WORKSPACE_MARKER,
    radii_xyz: Sequence[float] = DEFAULT_SWING_WORKSPACE_RADII_XYZ,
    color: np.ndarray | None = None,
) -> SwingWorkspaceMarker | None:
    """Compatibility wrapper over ``default_foot_reference_manager``."""
    return default_foot_reference_manager.attach_swing_workspace_marker(
        viewer,
        enabled=enabled,
        radii_xyz=radii_xyz,
        color=color,
    )


def update_swing_workspace_marker(
    marker: SwingWorkspaceMarker | None,
    viewer,
    mpc,
) -> None:
    """Compatibility wrapper over ``default_foot_reference_manager``."""
    default_foot_reference_manager.update_swing_workspace_marker(marker, viewer, mpc)
