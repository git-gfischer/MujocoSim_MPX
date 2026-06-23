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
#=========================H1 Humanoid=========================================
#=============================================================================

# region h1_wb_obj
def h1_wb_obj(n_joints,n_contact,N,W,reference,x, u, t):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    grf = x[13+2*n_joints+n_contact*3:]
    tau = u[:n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    q_ref = reference[t,7:7+n_joints]
    dp_ref = reference[t,7+n_joints:10+n_joints]
    omega_ref = reference[t,10+n_joints:13+n_joints]
    p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
    grf_ref = reference[t,13+n_joints+4*n_contact:13+n_joints+7*n_contact]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
    # joints_limits = jnp.array([
    # 0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    # 0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    # 2.35, 2.35, 
    # 2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25, 
    # 2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
    # joints_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
    # torque_limits = jnp.array([
    #     200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
    #     200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
    #     200, 200,
    #     40, 40, 40, 40, 18, 18, 18, 18,
    #     40, 40, 40, 40, 18, 18, 18, 18])
    # torque_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]

    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq +\
                 (p_leg - p_leg_ref).T @W[12+2*n_joints:12+2*n_joints+3*n_contact,12+2*n_joints:12+2*n_joints+3*n_contact]@ (p_leg - p_leg_ref) + \
                 tau.T @ W[12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact,12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact] @ tau +\
                + (grf - grf_ref).T@W[12+3*n_joints+3*n_contact:12+3*n_joints+6*n_contact,12+3*n_joints+3*n_contact:12+3*n_joints+6*n_contact]@(grf - grf_ref)#+ jnp.sum(penalty(joints_limits)) + jnp.sum(penalty(torque_limits))
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)
#endregion


# region h1_wb_obj
@partial(jax.jit, static_argnums=(0,1,2,3))
def h1_wb_hessian_gn(n_joints,n_contact,W,reference,x, u, t):

    return J_x(x,u).T@W@J_x(x,u) + H_constraint, J_u(x,u).T@W@J_u(x,u) + H_constraint_u, J_x(x,u).T@W@J_u(x,u)
#endregion

# region h1_wb_hessian_gn
def h1_wb_hessian_gn(n_joints,n_contact,W,reference,x, u, t):

    
    def residual(x,u):
        p = x[:3]
        quat = x[3:7]
        q = x[7:7+n_joints]
        dp = x[7+n_joints:10+n_joints]
        omega = x[10+n_joints:13+n_joints]
        dq = x[13+n_joints:13+2*n_joints]
        tau = u[:n_joints]
        grf = x[13+2*n_joints+n_contact*3:]
        p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

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
        p_leg_res = (p_leg - p_leg_ref).T
        tau_res = tau.T
        grf_res = (grf - grf_ref).T

        return jnp.concatenate([p_res,quat_res,q_res,dp_res,omega_res,dq_res,p_leg_res,tau_res,grf_res])
    
    def friction_constraint(x):
        grf = x[13+2*n_joints+n_contact*3:]
        mu = 0.7
        friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
        return friction_cone
    def joint_constraint(x):
        q = x[7:7+n_joints]
        joints_limits = jnp.array([
        0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
        0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
        2.35, 2.35, 
        2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25, 
        2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
        return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
    def torque_constraint(u):
        tau = u[:n_joints]
        torque_limits = jnp.array([
        200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
        200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
        200, 200,
        40, 40, 40, 40, 18, 18, 18, 18,
        40, 40, 40, 40, 18, 18, 18, 18])
        return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
        
    J_x = jax.jacobian(residual,0)
    J_u = jax.jacobian(residual,1)
    hessian_penalty = jax.grad(jax.grad(penalty))
    J_friction_cone = jax.jacobian(friction_constraint)
    # J_joint = jax.jacobian(joint_constraint)
    # J_torque = jax.jacobian(torque_constraint)
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    H_penalty_friction = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(friction_constraint(x)), -1e6, 1e6)*contact)
    # H_penalty_joint = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(joint_constraint(x)), -1e6, 1e6))
    # H_penalty_torque = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(torque_constraint(u)), -1e6, 1e6))
    H_constraint = J_friction_cone(x).T@H_penalty_friction@J_friction_cone(x)
    # H_constraint += J_joint(x).T@H_penalty_joint@J_joint(x)
    # H_constraint_u = J_torque(u).T@H_penalty_torque@J_torque(u)

    return J_x(x,u).T@W@J_x(x,u), J_u(x,u).T@W@J_u(x,u), J_x(x,u).T@W@J_u(x,u)

#endregion

# region talos_wb_obj
def talos_wb_obj(n_joints,n_contact,N,W,reference,x, u, t):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    # grf = x[13+2*n_joints+n_contact*3:]
    tau = u[:n_joints]
    grf = u[n_joints:]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
    # foot_speed = x[13+2*n_joints+3*n_contact:13+2*n_joints+6*n_contact]
    
    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    q_ref = reference[t,7:7+n_joints]
    dp_ref = reference[t,7+n_joints:10+n_joints]
    omega_ref = reference[t,10+n_joints:13+n_joints]
    p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
    grf_ref = reference[t,13+n_joints+4*n_contact:13+n_joints+7*n_contact]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
    # joints_limits = jnp.array([
    # 0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    # 0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    # 2.35, 2.35, 
    # 2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25, 
    # 2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
    # joints_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
    # torque_limits = jnp.array([
    #     200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
    #     200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
    #     200, 200,
    #     40, 40, 40, 40, 18, 18, 18, 18,
    #     40, 40, 40, 40, 18, 18, 18, 18])
    # torque_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    # zero the speed for the leg in swing phase
    # foot_speed = jnp.concatenate([foot_speed[0:3]*contact[0],foot_speed[3:6]*contact[1],foot_speed[6:9]*contact[2],foot_speed[9:12]*contact[3],
    #                        foot_speed[12:15]*contact[4],foot_speed[15:18]*contact[5],foot_speed[18:21]*contact[6],foot_speed[21:24]*contact[7]])
    
    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq +\
                 (p_leg - p_leg_ref).T @W[12+2*n_joints:12+2*n_joints+3*n_contact,12+2*n_joints:12+2*n_joints+3*n_contact]@ (p_leg - p_leg_ref) + \
                 tau.T @ W[12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact,12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact] @ tau +\
                + (grf - grf_ref).T@W[12+3*n_joints+3*n_contact:12+3*n_joints+6*n_contact,12+3*n_joints+3*n_contact:12+3*n_joints+6*n_contact]@(grf - grf_ref) #+ foot_speed.T@W[12+3*n_joints+6*n_contact:12+3*n_joints+9*n_contact,12+3*n_joints+6*n_contact:12+3*n_joints+9*n_contact]@foot_speed#+ jnp.sum(penalty(joints_limits)) + jnp.sum(penalty(torque_limits))
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)
#endregion

# region talos_wb_hessian_gnpenalty