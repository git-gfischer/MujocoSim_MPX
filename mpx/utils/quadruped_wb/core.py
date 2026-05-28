"""Shared constrained whole-body rollout for planar foot contacts.

Contact handling uses Jacobian column masking plus Tikhonov regularization so the
implicit contact solve excludes swing legs (see locomotion vs balance wrappers
for nominal tuning).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math


def quadruped_wb_step(
    *,
    baumgarte_alpha: float,
    ridge_stance: float,
    ridge_swing: float,
    model,
    mjx_model,
    contact_id,
    body_id,
    n_joints,
    dt,
    x,
    u,
    t,
    parameter,
):
    """One semi-implicit whole-body integration step under bilinear frictionless-style contacts."""

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=x[: n_joints + 7], qvel=x[n_joints + 7 : 2 * n_joints + 13])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t, :4]

    tau = jnp.concatenate([jnp.zeros(6), u])

    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)
    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)

    alpha = jnp.float32(baumgarte_alpha)
    rs = jnp.float32(ridge_stance)
    rw = jnp.float32(ridge_swing)

    contact_col = jnp.repeat(contact.astype(J.dtype), 3)
    Jc = J * contact_col
    qvel_slice = x[n_joints + 7 : 13 + 2 * n_joints]
    g_dot = Jc.T @ qvel_slice
    baumgarte_term = -2 * alpha * g_dot

    JT_M_invJ = Jc.T @ jax.scipy.linalg.cho_solve((M, False), Jc)
    reg_diag = jnp.where(contact_col > 0.5, rs, rw)
    JT_M_invJ = JT_M_invJ + jnp.diag(reg_diag)

    rhs = -Jc.T @ jax.scipy.linalg.cho_solve((M, False), tau - D) + baumgarte_term
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ)
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ, rhs)
    grf = grf * contact_col

    v = qvel_slice + jax.scipy.linalg.cho_solve((M, False), tau - D + J @ grf) * dt
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7 : 7 + n_joints] + v[6 : 6 + n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v, current_leg, grf])

    return x_next
