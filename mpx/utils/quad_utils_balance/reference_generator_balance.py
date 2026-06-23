import jax
from jax import numpy as jnp
from functools import partial
from mpx.estimators.terrain_orientation import terrain_orientation


# region reference_generator_balance
@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def reference_generator_balance(
    use_terrain_estimator,
    N,
    dt,
    n_joints,
    n_contact,
    mass,
    foot0,
    q0,
    t_timer,
    x,
    foot,
    input,
    duty_factor,
    step_freq,
    step_height,
    liftoff,
    contact,
    clearence_speed,
    fixed_contact_mask,
    foot_ref_anchor,
    use_foot_ref_anchor,
    base_quat_ref,
    use_base_quat_ref,
):
    """Static contact schedule + constant foot anchors (whole-body MPC balance)."""
    _, _, _, _, _, _, _ = (
        foot0,
        t_timer,
        duty_factor,
        step_freq,
        step_height,
        contact,
        clearence_speed,
    )

    p = x[:3]
    quat = x[3:7]
    yaw = jnp.arctan2(
        2 * (quat[0] * quat[3] + quat[1] * quat[2]),
        1 - 2 * (quat[2] * quat[2] + quat[3] * quat[3]),
    )
    Ryaw = jnp.array(
        [
            [jnp.cos(yaw), -jnp.sin(yaw), 0],
            [jnp.sin(yaw), jnp.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    proprio_height = input[6] + jnp.sum(liftoff[2::3]) / n_contact
    p = jnp.array([p[0], p[1], proprio_height])
    if use_terrain_estimator:
        quat_ref = jnp.tile(terrain_orientation(liftoff, Ryaw), (N + 1, 1))
    else:
        # Match current base attitude (incl. spawn yaw). Identity here fought random map yaw and
        # caused large quat_sub costs → in-place spin / falls while "balancing".
        quat_n = quat / (jnp.linalg.norm(quat) + 1e-8)
        quat_ref_n = base_quat_ref / (jnp.linalg.norm(base_quat_ref) + 1e-8)
        quat_ref = jax.lax.cond(
            use_base_quat_ref,
            lambda _: jnp.tile(quat_ref_n, (N + 1, 1)),
            lambda _: jnp.tile(quat_n, (N + 1, 1)),
            operand=None,
        )
    q_ref = jnp.tile(q0, (N + 1, 1))
    pitch = jnp.arcsin(
        2 * (quat_ref[0, 0] * quat_ref[0, 2] - quat_ref[0, 3] * quat_ref[0, 1])
    )
    Rpitch = jnp.array(
        [
            [jnp.cos(pitch), 0, jnp.sin(pitch)],
            [0, 1, 0],
            [-jnp.sin(pitch), 0, jnp.cos(pitch)],
        ]
    )

    ref_lin_vel = Ryaw @ Rpitch @ input[:3]
    ref_ang_vel = input[3:6]
    p_ref_x = jnp.arange(N + 1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N + 1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N + 1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    dp_ref = jnp.tile(ref_lin_vel, (N + 1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N + 1, 1))
    foot_track = jnp.where(use_foot_ref_anchor, foot_ref_anchor, foot)
    foot_ref = jnp.tile(foot_track, (N + 1, 1))
    grf_ref = jnp.zeros((N + 1, 3 * n_contact))

    mask = fixed_contact_mask.astype(jnp.float32).reshape((n_contact,))
    contact_sequence = jnp.tile(mask, (N + 1, 1))
    sum_m = jnp.sum(mask) + 1e-6
    grf_z_per_leg = mask * (mass * 9.81 / sum_m)
    grf_ref = grf_ref.at[:, 2::3].set(jnp.broadcast_to(grf_z_per_leg, (N + 1, n_contact)))
    reference = jnp.concatenate(
        [
            p_ref,
            quat_ref,
            q_ref,
            dp_ref,
            omega_ref,
            foot_ref,
            contact_sequence,
            grf_ref,
        ],
        axis=1,
    )
    parameter = jnp.concatenate([contact_sequence], axis=1)
    return reference, parameter, liftoff
#endregion
