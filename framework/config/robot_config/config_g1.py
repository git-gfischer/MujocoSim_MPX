"""Unitree G1 (23 DoF) whole-body locomotion MPC configuration.

Consumed by ``framework.utils.quad_utils_locomotion.mpc_wrapper_locomotion.MPCWrapper``.
Each foot carries four point contacts (``0..3_left`` / ``0..3_right``), so the
gait timer and every contact-indexed array have eight entries: left foot first.
"""
from functools import partial
import os

import jax
import jax.numpy as jnp

import framework.utils.humanoid_dyn_models.models as mpc_dyn_model
import framework.utils.humanoid_dyn_models.objectives as mpc_objectives

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..', '..', 'data', 'unitree_g1', 'g1.xml'))
scene_path = os.path.abspath(os.path.join(dir_path, '..', '..', 'data', 'unitree_g1', 'scene.xml'))

contact_frame = ['0_left', '1_left', '2_left', '3_left',
                 '0_right', '1_right', '2_right', '3_right']
body_name = ['left_ankle_roll_link'] * 4 + ['right_ankle_roll_link'] * 4

dt = 0.02
N = 25
mpc_frequency = 50
solver_mode = "primal_dual"

# Gait: left foot points share one phase, right foot points the other.
timer_t = jnp.array([0.5] * 4 + [0.0] * 4)
duty_factor = 0.7
step_freq = 1.2
step_height = 0.08
clearance_speed = 0.2
robot_height = 0.728  # base height above the soles in the nominal pose [m]
initial_height = robot_height
use_terrain_estimation = False

p0 = jnp.array([0.0, 0.0, robot_height + 0.001])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
_leg0 = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0]  # hip pitch/roll/yaw, knee, ankle pitch/roll
q0 = jnp.array(_leg0 + _leg0 + [0.0] + [0.0] * 5 + [0.0] * 5)  # legs, torso, left arm, right arm

# Contact points in the base frame for q0, z on the ground.
p_legs0 = jnp.array([ 0.10214,  0.13787, 0.0,
                      0.10214,  0.09787, 0.0,
                     -0.08786,  0.13787, 0.0,
                     -0.08786,  0.09787, 0.0,
                      0.10214, -0.09787, 0.0,
                      0.10214, -0.13787, 0.0,
                     -0.08786, -0.09787, 0.0,
                     -0.08786, -0.13787, 0.0])

n_joints = 23
n_contact = len(contact_frame)
n = 13 + 2 * n_joints + 6 * n_contact
m = n_joints
grf_as_state = True
u_ref = jnp.zeros(m)
initial_state = jnp.concatenate([p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0, jnp.zeros(3 * n_contact)])

# Actuator ctrlrange from g1.xml.
torque_limits = jnp.array([88, 88, 88, 139, 40, 40,
                           88, 88, 88, 139, 40, 40,
                           88,
                           20, 20, 20, 20, 20,
                           20, 20, 20, 20, 20], dtype=jnp.float32)
max_torque = torque_limits
min_torque = -torque_limits

_leg_w = [1e0, 1e2, 1e2, 1e0, 1e0, 1e1]  # hip roll/yaw held stiff so the feet stay aligned with the base
Qp = jnp.diag(jnp.array([0.0, 0.0, 1e4]))
Qrot = jnp.diag(jnp.array([1e3, 1e3, 0.0]))
Qq = jnp.diag(jnp.array(_leg_w + _leg_w + [1e2] + [1e1] * 10))
Qdp = jnp.diag(jnp.ones(3)) * 5e3
Qomega = jnp.diag(jnp.array([1e2, 1e2, 1e3]))
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qleg = jnp.diag(jnp.tile(jnp.array([1e4, 1e4, 1e5]), n_contact))
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-2
Qgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Qgrf)

cost = partial(mpc_objectives.humanoid_wb_obj, n_joints, n_contact, N, torque_limits)
hessian_approx = None


def dynamics(model, mjx_model, contact_id, body_id):
    return partial(mpc_dyn_model.humanoid_wb_dynamics, model, mjx_model, contact_id, body_id, n_joints, dt)
