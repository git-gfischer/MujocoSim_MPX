# Foot Reference Usage

This module supports both **class-based usage** (recommended) and **legacy function usage**.

## 1) Configure parameters

Tune defaults in `mpx/config/sim_config/config_foot_ref_config.py`:

- `foot_ref_sphere_radius`
- `show_desired_foot_markers`
- `swing_goal_radius`
- `show_swing_foot_goal`
- `foot_ref_colors_rgba`
- `sampling_mode` (`"ellipsoid"` or `"box"`)
- `show_swing_workspace_marker`
- `swing_workspace_radii_xyz`

Import the shared config instance:

```python
from mpx.config.sim_config.config_foot_ref_config import foot_ref_config
```

## 2) Recommended class-based API

```python
from mpx.utils.quadruped_wb.foot_reference import FootReferenceManager

foot_ref = FootReferenceManager(foot_ref_config)

desired_markers = foot_ref.attach_desired_foot_markers(env.viewer, n_contact=4)
swing_goal_marker = foot_ref.attach_swing_foot_goal_marker(env.viewer)
workspace_marker = foot_ref.attach_swing_workspace_marker(env.viewer)

# In your simulation loop:
foot_ref.update_desired_foot_markers(desired_markers, env.viewer, mpc)
foot_ref.update_swing_foot_goal_marker(swing_goal_marker, env.viewer, mpc)
foot_ref.update_swing_workspace_marker(workspace_marker, env.viewer, mpc)
```

## 3) Sampling tripod world-frame references

```python
foot_ref_world = foot_ref.tripod_foot_reference_world(
    key=key,
    p=base_pos,
    quat=base_quat,
    foot0=foot_nominal_base,
    n_contact=4,
    sigma=sigma_xyz,
)
```

By default (`sampling_mode="ellipsoid"`), random XYZ offsets are sampled in base/foot
frame inside an ellipsoid and then projected to world frame. Use `sampling_mode="box"`
to recover axis-aligned uniform sampling.

## 4) Legacy API compatibility

Existing code can continue to call:

- `attach_desired_foot_markers(...)`
- `update_desired_foot_markers(...)`
- `attach_swing_foot_goal_marker(...)`
- `update_swing_foot_goal_marker(...)`
- `attach_swing_workspace_marker(...)`
- `update_swing_workspace_marker(...)`

These wrappers now call an internal default `FootReferenceManager`.
