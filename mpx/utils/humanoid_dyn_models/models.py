import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math
#=============================================================================
#=========================H1 Humanoid=========================================
#=============================================================================
# region h1_wb_dynamics
def h1_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:2*n_joints+13])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t,:4]

    tau = jnp.concatenate([jnp.zeros(6),u])

    FL = mjx_data.geom_xpos[contact_id[0]]
    RL = mjx_data.geom_xpos[contact_id[1]]
    FR = mjx_data.geom_xpos[contact_id[2]]
    RR = mjx_data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL, body_id[0])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR,  body_id[1])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR,  body_id[1])
    J = jnp.concatenate([J_FL,J_RL,J_FR,J_RR],axis=1)
    g_dot = J.T @ x[n_joints+7:13+2*n_joints]  # Velocity-level constraint violation
    
    alpha = 5
    # beta = 2*jnp.sqrt(alpha)
    # Stabilization term
    baumgarte_term = - 2*alpha * g_dot #- beta * beta * g

    JT_M_invJ = J.T @ jax.scipy.linalg.cho_solve((M, False), J)


    rhs = -J.T @ jax.scipy.linalg.cho_solve((M, False),tau - D) + baumgarte_term 
    epsilon = 1e-3
    JT_M_invJ_reg = JT_M_invJ + epsilon * jnp.eye(JT_M_invJ.shape[0])
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ_reg)
    
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ,rhs)
    grf = jnp.concatenate([grf[0:3]*contact[0],grf[3:6]*contact[1],grf[6:9]*contact[2],grf[9:12]*contact[3]])
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False),tau - D + J@grf)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v,FL,RL,FR,RR,grf])
    
    return x_next
#endregion

# region h1 contact kinematics
def _h1_contact_kinematics(mjx_model, mjx_data, contact_id, body_id):
    fl = mjx_data.geom_xpos[contact_id[0]]
    rl = mjx_data.geom_xpos[contact_id[1]]
    fr = mjx_data.geom_xpos[contact_id[2]]
    rr = mjx_data.geom_xpos[contact_id[3]]

    j_fl, _ = mjx.jac(mjx_model, mjx_data, fl, body_id[0])
    j_rl, _ = mjx.jac(mjx_model, mjx_data, rl, body_id[0])
    j_fr, _ = mjx.jac(mjx_model, mjx_data, fr, body_id[1])
    j_rr, _ = mjx.jac(mjx_model, mjx_data, rr, body_id[1])

    feet = jnp.concatenate([fl, rl, fr, rr], axis=0)
    jacobian = jnp.concatenate([j_fl, j_rl, j_fr, j_rr], axis=1)
    return feet, jacobian
#endregion

# region h1_kinodynamic_dynamics
def h1_kinodynamic_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    
    qpos = x[: n_joints + 7]
    qvel = x[n_joints + 7 : 2 * n_joints + 13]
    dq = x[13 + n_joints : 13 + 2 * n_joints]
    dq_next = u[:n_joints]
    contact = parameter[t, :4]
    grf = _mask_contact_forces(u[n_joints:], contact)

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    mass_matrix = mjx.full_m(mjx_model, mjx_data)
    bias = mjx_data.qfrc_bias
    feet_next, jacobian = _h1_contact_kinematics(mjx_model, mjx_data, contact_id, body_id)

    qdd_joints = (dq_next - dq) / dt
    rhs = (jacobian @ grf)[:6] - bias[:6] - mass_matrix[:6, 6:] @ qdd_joints
    qdd_base = jnp.linalg.solve(mass_matrix[:6, :6] + 1e-6 * jnp.eye(6), rhs)

    base_velocity_next = qvel[:6] + qdd_base * dt
    qvel_next = jnp.concatenate([base_velocity_next, dq_next])

    p_next = x[:3] + qvel_next[:3] * dt
    quat_next = math.quat_integrate(x[3:7], qvel_next[3:6], dt)
    q_next = x[7 : 7 + n_joints] + dq_next * dt

    return jnp.concatenate([p_next, quat_next, q_next, qvel_next, feet_next])
#endregion

# region h1_kinodynamic_torques
def h1_kinodynamic_torques(
    model,
    mjx_model,
    contact_id,
    body_id,
    n_joints,
    dt,
    joint_kp,
    joint_kd,
    x0,
    X,
    U,
    reference,
    parameter,
):
    del reference
    qpos = x0[: n_joints + 7]
    qvel = x0[n_joints + 7 : 2 * n_joints + 13]
    qvel_next = X[1, n_joints + 7 : 2 * n_joints + 13]
    qacc = (qvel_next - qvel) / dt

    contact = parameter[0, :4]
    grf = _mask_contact_forces(U[0, n_joints:], contact)

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, qacc=qacc)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)
    _, jacobian = _h1_contact_kinematics(mjx_model, mjx_data, contact_id, body_id)
    mjx_data = mjx.inverse(mjx_model, mjx_data)

    tau_ff = (mjx_data.qfrc_inverse - jacobian @ grf)[6:]
    q_des = X[1, 7 : 7 + n_joints]
    dq_des = qvel_next[6:]
    tau_fb = joint_kp * (q_des - qpos[7:]) + joint_kd * (dq_des - qvel[6:])
    return tau_ff + tau_fb
#endregion

#=============================================================================
#=========================Talos=================================================
#=============================================================================
# region talos_wb_dynamics
def talos_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:2*n_joints+13])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t,:8]

    tau = jnp.concatenate([jnp.zeros(6),u[:n_joints]])

    left_foot_1 = mjx_data.geom_xpos[contact_id[0]]
    left_foot_2 = mjx_data.geom_xpos[contact_id[1]]
    left_foot_3 = mjx_data.geom_xpos[contact_id[2]]
    left_foot_4 = mjx_data.geom_xpos[contact_id[3]]

    right_foot_1 = mjx_data.geom_xpos[contact_id[4]]
    right_foot_2 = mjx_data.geom_xpos[contact_id[5]]
    right_foot_3 = mjx_data.geom_xpos[contact_id[6]]
    right_foot_4 = mjx_data.geom_xpos[contact_id[7]]

    J_fl_1, _ = mjx.jac(mjx_model, mjx_data, left_foot_1, body_id[0])
    J_fl_2, _ = mjx.jac(mjx_model, mjx_data, left_foot_2, body_id[0])
    J_fl_3, _ = mjx.jac(mjx_model, mjx_data, left_foot_3, body_id[0])
    J_fl_4, _ = mjx.jac(mjx_model, mjx_data, left_foot_4, body_id[0])

    J_rl_1, _ = mjx.jac(mjx_model, mjx_data, right_foot_1, body_id[1])
    J_rl_2, _ = mjx.jac(mjx_model, mjx_data, right_foot_2, body_id[1])
    J_rl_3, _ = mjx.jac(mjx_model, mjx_data, right_foot_3, body_id[1])
    J_rl_4, _ = mjx.jac(mjx_model, mjx_data, right_foot_4, body_id[1])
    
    J = jnp.concatenate([J_fl_1,J_fl_2,J_fl_3,J_fl_4,J_rl_1,J_rl_2,J_rl_3,J_rl_4],axis=1)
    grf = u[n_joints:]
    grf = jnp.concatenate([grf[0:3]*contact[0],grf[3:6]*contact[1],grf[6:9]*contact[2],grf[9:12]*contact[3],
                           grf[12:15]*contact[4],grf[15:18]*contact[5],grf[18:21]*contact[6],grf[21:24]*contact[7]])
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False),tau - D + J@grf)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v,left_foot_1,left_foot_2,left_foot_3,left_foot_4,right_foot_1,right_foot_2,right_foot_3,right_foot_4])
    
    return x_next
#endregion