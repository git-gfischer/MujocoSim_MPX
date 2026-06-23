"""
Foot-reference generation and viewer marker defaults for quadruped balance.

Typical use::

    from mpx.config.sim_config.config_foot_ref_config import (
        FootReferenceConfig,
        foot_ref_config,
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

    # Fixed swing-goal marker settings.
    swing_goal_radius: float = 0.08
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
