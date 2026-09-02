"""
Foot-reference generation and viewer marker defaults for quadruped balance.

Typical use::

    from mpx.config.sim_config.config_foot_ref_config import (
        FootReferenceConfig,
        foot_ref_config,
        RandomSwingFootConfig,
        random_swing_foot_config,
    )
    from mpx.utils.quadruped_dyn_models.foot_reference import FootReferenceManager

    # Use defaults:
    foot_ref = FootReferenceManager(foot_ref_config)

    # Or customize:
    custom_cfg = FootReferenceConfig(
        foot_ref_sphere_radius=0.03,
        swing_goal_radius=0.06,
        show_desired_foot_markers=False,
    )
    foot_ref = FootReferenceManager(custom_cfg)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class FootReferenceConfig:
    """Config for desired-foot references and passive-viewer debug markers."""

    # FL, FR, RL, RR in contact-frame order.
    foot_ref_colors_rgba: tuple[np.ndarray, ...] = field(
        default_factory=lambda: (
            np.array([1.0, 0.2, 0.2, 0.9]),
            np.array([0.2, 1.0, 0.2, 0.9]),
            np.array([0.25, 0.45, 1.0, 0.9]),
            np.array([1.0, 0.85, 0.1, 0.9]),
        )
    )

    # Desired-foot marker settings.
    foot_ref_sphere_radius: float = 0.02
    show_desired_foot_markers: bool = True

    # Fixed swing-goal marker settings (small sphere at the exact target center).
    swing_goal_radius: float = 0.025
    swing_goal_color_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.15, 1.0, 0.95])
    )
    show_swing_foot_goal: bool = True

    # Sampling mode for nominal base-frame foot offsets.
    # - "box": independent uniform XYZ in [-sigma, +sigma]
    # - "ellipsoid": uniform inside ellipsoid with radii derived from sigma
    sampling_mode: Literal["box", "ellipsoid"] = "ellipsoid"

    # Viewer marker for swing-foot reachable workspace (ellipsoid).
    show_swing_workspace_marker: bool = True
    swing_workspace_radii_xyz: tuple[float, float, float] = (0.08, 0.05, 0.035)
    swing_workspace_color_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.95, 1.0, 0.28])
    )


# Default profile used across quadruped WB utilities.
foot_ref_config = FootReferenceConfig()


@dataclass
class RandomSwingFootConfig:
    """Bounds for random swing-foot target placement.

    All bounds are expressed as offsets in the **yaw-aligned base frame**
    (origin at ``qpos[:3]``, robot base COM):

    - X: forward along the robot heading.
    - Y: lateral (positive = left).
    - Z: vertical (positive = up from base; negative = below base toward ground).

    A uniform sample inside the bounds box is mapped to world frame via
    ``base_pos + R_yaw @ [x, y, z]``.

    For Go2 standing (~0.27 m base height), ground contact is roughly
    ``z ≈ -0.27`` in this frame; tune ``z_bounds`` accordingly.

    Keyboard shortcuts in the simulator (GLFW):
      R  — sample a new random swing target immediately.
      N  — toggle auto-randomise on every respawn on/off.

    When ``resample_on_arrival`` is True, a new random target is drawn automatically
    once the measured swing foot is within ``arrival_tolerance_xy_m`` / ``arrival_tolerance_z_m``.
    """

    # Set to True to enable random swing-foot mode at startup.
    enabled: bool = True

    # Base-frame offset ranges (origin at qpos[:3], yaw-aligned axes).
    # Tune these to match your robot's reachable workspace.
    x_bounds: tuple[float, float] = (-0.05, 0.4)   # forward   [m]
    y_bounds: tuple[float, float] = (0.05, 0.3)  # lateral   [m]
    z_bounds: tuple[float, float] = (-0.27, -0.20)  # vertical  [m]  (negative = below base)

    # When True, a new random target is drawn automatically on every respawn.
    resample_on_respawn: bool = True

    # When True (and random mode is enabled), sample a new target once the swing
    # foot reaches the current goal within ``arrival_tolerance_m``.
    resample_on_arrival: bool = True

    # Distance [m] for arrival (separate XY / Z — tripod tracking rarely hits tight 3D).
    arrival_tolerance_xy_m: float = 0.12
    arrival_tolerance_z_m: float = 0.1

    # When both foot and goal are below this Z [m], use XY+Z tolerances (foot on ground).
    ground_contact_z_max: float = 0.08

    # Consecutive sim steps within tolerance before resampling (debounce).
    arrival_hold_steps: int = 3

    # MuJoCo foot collision sphere radius [m] (Go2: 0.0175). Used to compensate
    # arrival checks in Z: the geom center sits this far above ground contact.
    foot_geom_radius_m: float = 0.0175

    # Minimum wait [s] after an arrival resample before checking again.
    resample_cooldown_s: float = 0.3

    # Semi-transparent box showing the random swing-foot sampling region
    # (base-frame XYZ bounds relative to qpos[:3]).
    show_bounds_box: bool = True
    bounds_box_color_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.15, 0.95, 0.35, 0.22])
    )


# Default profile — disabled by default; flip ``enabled=True`` to activate.
random_swing_foot_config = RandomSwingFootConfig()
