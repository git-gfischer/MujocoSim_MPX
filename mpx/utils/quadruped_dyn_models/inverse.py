"""Robot-agnostic inverse-dynamics MPC transcription.

State ``x = [qpos (nq), qvel (nv)]``. Control
``u = [qacc (nv), tau (n_joints), grf (3 * n_contact)]``.

Dynamics integrate ``qacc``. Physics is the equality residual
``M qacc + bias - J f - [0_6; tau] = 0``.
"""

from functools import partial

import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math


def inv_dyn_dims(n_joints: int, n_contact: int):
    """Return ``(nq, nv, n, m, equality_dim)``."""
    nq = 7 + n_joints
    nv = 6 + n_joints
    n = nq + nv
    m = nv + n_joints + 3 * n_contact
    return nq, nv, n, m, nv


def inv_dyn_control_slices(n_joints: int, n_contact: int):
    """Slices of ``u`` for ``qacc``, ``tau``, and stacked GRFs."""
    nq, nv, n, m, _ = inv_dyn_dims(n_joints, n_contact)
    del nq, n
    return (
        slice(0, nv),
        slice(nv, nv + n_joints),
        slice(nv + n_joints, m),
    )


def pack_inverse_state(qpos, qvel):
    """Concatenate MuJoCo ``qpos`` / ``qvel`` into the inverse MPC state."""
    return jnp.concatenate([jnp.ravel(qpos), jnp.ravel(qvel)])


def _state_parts(x, nq, nv):
    return x[:nq], x[nq : nq + nv]


def _integrate_state(qpos, qvel, qacc, dt):
    qvel_next = qvel + qacc * dt
    qpos_next = jnp.concatenate(
        [
            qpos[:3] + qvel_next[:3] * dt,
            math.quat_integrate(qpos[3:7], qvel_next[3:6], dt),
            qpos[7:] + qvel_next[6:] * dt,
        ]
    )
    return qpos_next, qvel_next


def _mask_grf(grf, contact, n_contact):
    return (grf.reshape((n_contact, 3)) * contact[:n_contact, None]).reshape(-1)


def quadruped_inv_dyn_foot_positions(mjx_model, contact_id, qpos, qvel):
    data = mjx.make_data(mjx_model)
    data = data.replace(qpos=qpos, qvel=qvel)
    data = mjx.fwd_position(mjx_model, data)
    return jnp.concatenate([data.geom_xpos[cid] for cid in contact_id])


def quadruped_inv_dyn_dynamics(nq, nv, n_joints, n_contact, dt, x, u, t, parameter):
    del t, parameter
    qacc_slice, _, _ = inv_dyn_control_slices(n_joints, n_contact)
    qpos, qvel = _state_parts(x, nq, nv)
    qpos_next, qvel_next = _integrate_state(qpos, qvel, u[qacc_slice], dt)
    return jnp.concatenate([qpos_next, qvel_next])


def quadruped_inv_dyn_equality(
    mjx_model,
    contact_id,
    body_id,
    nq,
    nv,
    n_joints,
    n_contact,
    x,
    u,
    t,
    parameter,
):
    qacc_slice, tau_slice, grf_slice = inv_dyn_control_slices(n_joints, n_contact)
    qpos, qvel = _state_parts(x, nq, nv)
    qacc = u[qacc_slice]
    tau = u[tau_slice]
    grf = u[grf_slice]
    data = mjx.make_data(mjx_model)
    data = data.replace(qpos=qpos, qvel=qvel, qacc=qacc)
    data = mjx.fwd_position(mjx_model, data)
    data = mjx.fwd_velocity(mjx_model, data)
    mass_matrix = mjx.full_m(mjx_model, data)
    qfrc_inverse = mass_matrix @ qacc + data.qfrc_bias
    jacobians = []
    for cid, bid in zip(contact_id, body_id):
        jac, _ = mjx.jac(mjx_model, data, data.geom_xpos[cid], bid)
        jacobians.append(jac)
    contact_jacobian = jnp.concatenate(jacobians, axis=1)
    contact = parameter[t, :n_contact]
    grf = _mask_grf(grf, contact, n_contact)
    contact_wrench = contact_jacobian @ grf
    generalized_actuation = jnp.concatenate([jnp.zeros(6, dtype=u.dtype), tau])
    return qfrc_inverse - contact_wrench - generalized_actuation


def _locomotion_reference_parts(reference, t, n_joints, n_contact):
    row = reference[t]
    p_ref = row[:3]
    quat_ref = row[3:7]
    q_ref = row[7 : 7 + n_joints]
    dp_ref = row[7 + n_joints : 10 + n_joints]
    omega_ref = row[10 + n_joints : 13 + n_joints]
    p_leg_ref = row[13 + n_joints : 13 + n_joints + 3 * n_contact]
    grf_ref = row[13 + n_joints + 4 * n_contact : 13 + n_joints + 7 * n_contact]
    return p_ref, quat_ref, q_ref, dp_ref, omega_ref, p_leg_ref, grf_ref


def quadruped_inv_dyn_obj(
    mjx_model,
    contact_id,
    nq,
    nv,
    n_joints,
    n_contact,
    N,
    W,
    reference,
    x,
    u,
    t,
):
    qacc_slice, tau_slice, grf_slice = inv_dyn_control_slices(n_joints, n_contact)
    qpos, qvel = _state_parts(x, nq, nv)
    p = qpos[:3]
    quat = qpos[3:7]
    q = qpos[7:]
    dp = qvel[:3]
    omega = qvel[3:6]
    dq = qvel[6:]
    acc = u[qacc_slice]
    tau = u[tau_slice]
    grf = u[grf_slice]
    p_leg = quadruped_inv_dyn_foot_positions(mjx_model, contact_id, qpos, qvel)
    p_ref, quat_ref, q_ref, dp_ref, omega_ref, p_leg_ref, grf_ref = (
        _locomotion_reference_parts(reference, t, n_joints, n_contact)
    )
    quat_err = math.quat_sub(quat, quat_ref)
    stage_cost = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + quat_err.T @ W["rot"] @ quat_err
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
        + (p_leg - p_leg_ref).T @ W["contact"] @ (p_leg - p_leg_ref)
        + acc.T @ W["acc"] @ acc
        + tau.T @ W["tau"] @ tau
        + (grf - grf_ref).T @ W["grf"] @ (grf - grf_ref)
    )
    terminal_cost = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + quat_err.T @ W["rot"] @ quat_err
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
    )
    return jnp.where(t == N, 0.5 * terminal_cost, 0.5 * stage_cost)


def inverse_dynamics_factories(n_joints, n_contact, dt, N):
    """Return ``(dynamics, equality, cost)`` factories matching WB config style."""
    nq, nv, _, _, _ = inv_dyn_dims(n_joints, n_contact)

    def dynamics_factory(model, mjx_model, contact_id, body_id):
        del model, mjx_model, contact_id, body_id
        return partial(
            quadruped_inv_dyn_dynamics, nq, nv, n_joints, n_contact, dt
        )

    def equality_factory(model, mjx_model, contact_id, body_id):
        del model
        return partial(
            quadruped_inv_dyn_equality,
            mjx_model,
            contact_id,
            body_id,
            nq,
            nv,
            n_joints,
            n_contact,
        )

    def cost_factory(model, mjx_model, contact_id, body_id):
        del model, body_id
        return partial(
            quadruped_inv_dyn_obj,
            mjx_model,
            contact_id,
            nq,
            nv,
            n_joints,
            n_contact,
            N,
        )

    return dynamics_factory, equality_factory, cost_factory
