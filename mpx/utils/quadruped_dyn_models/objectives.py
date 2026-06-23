import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math
from functools import partial

def penalty(constraint,alpha = 0.1,sigma = 5):
        def safe_log(x):
            x = jnp.clip(x,1e-10,1e6)
            return jnp.log(x)
        quadratic_barrier = alpha/2*(jnp.square((constraint-2*sigma)/sigma)-jnp.ones_like(constraint))
        log_barrier = -alpha*safe_log(constraint)
        return jnp.clip(jnp.where(constraint>sigma,log_barrier,quadratic_barrier+log_barrier),0,1e8)

#=============================================================================
#=========================Quadruped============================================
#=============================================================================
# region quadruped_srbd_obj
def quadruped_srbd_obj(n_contact,N,W,reference,x, u, t):

    p = x[:3]
    quat = x[3:7]
    dp = x[7:10]
    omega = x[10:13]
    grf = u

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    dp_ref = reference[t,7:10]
    omega_ref = reference[t,10:13]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-2)
    friction_cone = jnp.clip(penalty(friction_cone),1e-6,1e6)
    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) +\
                 (dp - dp_ref).T @ W[6:9,6:9] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9:12,9:12] @ (omega - omega_ref) +\
                 grf.T@W[12:12+3*n_contact,12:12+3*n_contact]@grf + jnp.sum(friction_cone)
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) +\
                (dp - dp_ref).T @ W[6:9,6:9] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9:12,9:12] @ (omega - omega_ref)

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

def quadruped_srbd_hessian_gn(n_contact,W,reference,x, u, t):

    def residual(x,u):
        p = x[:3]
        quat = x[3:7]
        dp = x[7:10]
        omega = x[10:13]
        grf = u

        p_ref = reference[t,:3]
        quat_ref = reference[t,3:7]
        dp_ref = reference[t,7:10]
        omega_ref = reference[t,10:13]

        p_res = (p - p_ref).T
        quat_res = math.quat_sub(quat,quat_ref).T

        dp_res = (dp - dp_ref).T
        omega_res = (omega - omega_ref).T
        
        grf_res = grf.T

        return jnp.concatenate([p_res,quat_res,dp_res,omega_res,grf_res])
    
    def friction_constraint(u):
        grf = u
        mu = 0.7
        friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-2)
        return friction_cone
    J_x = jax.jacobian(residual,0)
    J_u = jax.jacobian(residual,1)
    contact = reference[t,13 : 13+n_contact]
    hessian_penalty = jax.grad(jax.grad(penalty))
    J_friction_cone = jax.jacobian(friction_constraint)
    H_penalty = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(friction_constraint(u)),1e-6, 1e6)*contact)
    H_constraint = J_friction_cone(u).T@H_penalty@J_friction_cone(u)
    return J_x(x,u).T@W@J_x(x,u), J_u(x,u).T@W@J_u(x,u) + H_constraint, J_x(x,u).T@W@J_u(x,u)
#endregion

# region quadruped_wb_obj
def quadruped_wb_obj(swing_tracking,n_joints,n_contact,N,W,reference,x, u, t):
    
    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
    grf = x[13+2*n_joints+3*n_contact:]
    tau = u[:n_joints]

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    q_ref = reference[t,7:7+n_joints]
    dp_ref = reference[t,7+n_joints:10+n_joints]
    omega_ref = reference[t,10+n_joints:13+n_joints]
    p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    grf_ref = reference[t,13+n_joints+4*n_contact:13+n_joints+7*n_contact]
    mu = 0.5
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-2)
    friction_cone = penalty(friction_cone)
    torque_limits = jnp.array([
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44 ])
    #min grf 
    # min_force = grf[2::3] - jnp.ones(n_contact)*10
    torque_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2

    joint_speed_limits = jnp.ones(2*n_joints)*10
    joint_speed_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@dq + joint_speed_limits + jnp.ones_like(joint_speed_limits)*1e-2

    #contact_map, grf_map = quadruped_wb_residual_masks(swing_tracking, contact, n_contact)

    if swing_tracking:
        contact_map = jnp.ones(3*n_contact)
    else:
        contact_map = jnp.array([jnp.ones(3)*contact[i] for i in range(n_contact)]).flatten()


    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq +\
                 (contact_map*(p_leg - p_leg_ref)).T @W[12+2*n_joints:12+2*n_joints+3*n_contact,12+2*n_joints:12+2*n_joints+3*n_contact]@ (contact_map*(p_leg - p_leg_ref))+ \
                 tau.T @ W[12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact,12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact] @ tau +\
                 (grf-grf_ref).T @ W[12+3*n_joints+3*n_contact:12+3*n_joints+6*n_contact,12+3*n_joints+3*n_contact:12+3*n_joints+6*n_contact] @ (grf-grf_ref) +\
                 jnp.sum(penalty(torque_limits,1,1)) + jnp.sum(friction_cone*contact) + jnp.sum(penalty(joint_speed_limits,1,1))
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)
#endregion

# region quadruped_wb_hessian_gn
def quadruped_wb_hessian_gn(swing_tracking,n_joints,n_contact,W,reference,x, u, t):

    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]

    def residual(x,u):

        p = x[:3]
        quat = x[3:7]
        q = x[7:7+n_joints]
        dp = x[7+n_joints:10+n_joints]
        omega = x[10+n_joints:13+n_joints]
        dq = x[13+n_joints:13+2*n_joints]
        p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
        grf = x[13+2*n_joints+3*n_contact:]
        tau = u[:n_joints]

        p_ref = reference[t,:3]
        quat_ref = reference[t,3:7]
        q_ref = reference[t,7:7+n_joints]
        dp_ref = reference[t,7+n_joints:10+n_joints]
        omega_ref = reference[t,10+n_joints:13+n_joints]
        p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
        grf_ref = reference[t,13+n_joints+4*n_contact:13+n_joints+7*n_contact]
        p_res = (p - p_ref).T
        quat_res = math.quat_sub(quat,quat_ref).T
        q_res = (q - q_ref).T
        dp_res = (dp - dp_ref).T
        omega_res = (omega - omega_ref).T
        dq_res = dq.T

        if swing_tracking:
            contact_map = jnp.ones(3*n_contact)
        else:
            contact_map = jnp.array([jnp.ones(3)*contact[i] for i in range(n_contact)]).flatten()
 
        p_leg_res = (contact_map*(p_leg - p_leg_ref)).T
        tau_res = tau.T
        grf_res = (grf-grf_ref).T
        return jnp.concatenate([p_res,quat_res,q_res,dp_res,omega_res,dq_res,p_leg_res,tau_res,grf_res])
    def friction_constraint(x):
        grf = x[13+2*n_joints+3*n_contact:]
        mu = 0.5
        friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
        return friction_cone
    def torque_constraint(u):
        tau = u[:n_joints]
        torque_limits = jnp.array([
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44 ])
        return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    def speed_constarint(x):
        dq = x[13+n_joints:13+2*n_joints]
        joint_speed_limits = jnp.ones(2*n_joints)*10
        joint_speed_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@dq + joint_speed_limits + jnp.ones_like(joint_speed_limits)*1e-2
        return joint_speed_limits
    def min_force_constraint(x):
        grf = x[13+2*n_joints+3*n_contact:]
        min_force = grf[2::3] - jnp.ones(n_contact)*10
        return min_force
    J_x = jax.jacobian(residual,0)
    J_u = jax.jacobian(residual,1)
    hessian_penalty = jax.grad(jax.grad(penalty))
    J_friction_cone = jax.jacobian(friction_constraint)
    J_torque = jax.jacobian(torque_constraint)
    J_speed = jax.jacobian(speed_constarint)
    # J_min_force = jax.jacobian(min_force_constraint)
    hessian_penalty_torque = partial(hessian_penalty,alpha = 1,sigma = 1)
    hessian_penalty_collision = partial(hessian_penalty,alpha = 10,sigma = 0.01)
    # W = W.at[12+2*n_joints + 6:12+2*n_joints+3*n_contact,12+2*n_joints + 6:12+2*n_joints+3*n_contact].set(W[12+2*n_joints + 6:12+2*n_joints+3*n_contact,12+2*n_joints + 6:12+2*n_joints+3*n_contact]*stand_up_flag)
    H_penalty = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(friction_constraint(x)), -1e6, 1e6)*contact)
    H_penalty_torque = jnp.diag(jnp.clip(jax.vmap(hessian_penalty_torque)(torque_constraint(u)), -1e6, 1e6))
    H_penalty_speed = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(speed_constarint(x)), -1e6, 1e6))
    # H_penalty_min_force = jnp.diag(jnp.clip(jax.vmap(hessian_penalty_torque)(min_force_constraint(x)), -1e6, 1e6)*contact)
    H_constraint = J_friction_cone(x).T@H_penalty@J_friction_cone(x)
    H_constraint_u = J_torque(u).T@H_penalty_torque@J_torque(u)
    H_constraint += J_speed(x).T@H_penalty_speed@J_speed(x)
    # H_constraint += J_min_force(x).T@H_penalty_min_force@J_min_force(x)
    return J_x(x,u).T@W@J_x(x,u) + H_constraint, J_u(x,u).T@W@J_u(x,u) + H_constraint_u, J_x(x,u).T@W@J_u(x,u)
#endregion


quadruped_wb_locomotion_obj = partial(quadruped_wb_obj, True)
quadruped_wb_balance_obj = partial(quadruped_wb_obj, False)
quadruped_wb_locomotion_hessian_gn = partial(quadruped_wb_hessian_gn, True)
quadruped_wb_balance_hessian_gn = partial(quadruped_wb_hessian_gn, False)