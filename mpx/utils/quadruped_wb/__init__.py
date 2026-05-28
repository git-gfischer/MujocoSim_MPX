"""Quadruped whole-body MPC: behaviour-specific dynamics and objective helpers.

- :func:`quadruped_wb_dynamics` — default gait / locomotion rollout (exported from ``models.py`` too).
- :func:`quadruped_wb_dynamics_balance` — same model with tuning aimed at tripod / reduced support.
"""

from mpx.utils.quadruped_wb.balance import quadruped_wb_dynamics_balance
from mpx.utils.quadruped_wb.base_pose import (
    DesiredBasePoseVisualizationConfig,
    DesiredBasePoseVisualizer,
    FourLegBalanceBasePoseRandomizationConfig,
    FourLegBalanceBasePoseRandomizer,
)
from mpx.utils.quadruped_wb.foot_reference import (
    DEFAULT_FOOT_REF_COLORS_RGBA,
    DEFAULT_FOOT_REF_SAMPLING_MODE,
    DEFAULT_FOOT_REF_SPHERE_RADIUS,
    DEFAULT_SHOW_DESIRED_FOOT_MARKERS,
    DEFAULT_SHOW_SWING_FOOT_GOAL,
    DEFAULT_SHOW_SWING_WORKSPACE_MARKER,
    DEFAULT_SWING_WORKSPACE_COLOR_RGBA,
    DEFAULT_SWING_WORKSPACE_RADII_XYZ,
    DesiredFootMarkers,
    FootReferenceManager,
    FootReferenceMarkers,
    SwingFootGoalMarker,
    SwingWorkspaceMarker,
    attach_desired_foot_markers,
    attach_swing_foot_goal_marker,
    attach_swing_workspace_marker,
    base_frame_feet_to_world,
    default_foot_reference_manager,
    is_tripod_contact_mask,
    sample_nominal_foot_offsets_base,
    sample_nominal_foot_offsets_base_box,
    sample_nominal_foot_offsets_base_ellipsoid,
    swing_goal_xyz_from_foot_ref,
    tripod_foot_reference_world,
    update_desired_foot_markers,
    update_swing_foot_goal_marker,
    update_swing_workspace_marker,
)
from mpx.utils.quadruped_wb.locomotion import quadruped_wb_dynamics

__all__ = [
    "quadruped_wb_dynamics",
    "quadruped_wb_dynamics_balance",
    "DesiredBasePoseVisualizationConfig",
    "DesiredBasePoseVisualizer",
    "FourLegBalanceBasePoseRandomizationConfig",
    "FourLegBalanceBasePoseRandomizer",
    "DEFAULT_FOOT_REF_COLORS_RGBA",
    "DEFAULT_FOOT_REF_SAMPLING_MODE",
    "DEFAULT_FOOT_REF_SPHERE_RADIUS",
    "DEFAULT_SHOW_DESIRED_FOOT_MARKERS",
    "DEFAULT_SHOW_SWING_FOOT_GOAL",
    "DEFAULT_SHOW_SWING_WORKSPACE_MARKER",
    "DEFAULT_SWING_WORKSPACE_COLOR_RGBA",
    "DEFAULT_SWING_WORKSPACE_RADII_XYZ",
    "DesiredFootMarkers",
    "FootReferenceManager",
    "FootReferenceMarkers",
    "SwingFootGoalMarker",
    "SwingWorkspaceMarker",
    "default_foot_reference_manager",
    "attach_desired_foot_markers",
    "attach_swing_foot_goal_marker",
    "attach_swing_workspace_marker",
    "swing_goal_xyz_from_foot_ref",
    "update_desired_foot_markers",
    "update_swing_foot_goal_marker",
    "update_swing_workspace_marker",
    "base_frame_feet_to_world",
    "is_tripod_contact_mask",
    "sample_nominal_foot_offsets_base",
    "sample_nominal_foot_offsets_base_box",
    "sample_nominal_foot_offsets_base_ellipsoid",
    "tripod_foot_reference_world",
]
