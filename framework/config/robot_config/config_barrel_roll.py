"""Barrel roll, solved once offline with ``MPCWrapper.runOffline``.

Go2 only: the Proprioceptive Image network is trained on ``go2_dls``.
The aliengo version of this file is in git history (commit 482d545).
"""
from functools import partial

import jax.numpy as jnp
import mujoco

import framework.utils.quadruped_dyn_models.objectives as mpc_objectives
from framework.config.robot_config.config_go2 import (
    _DLS_MPC_MODEL_PATH,
    Go2Locomotion,
    LocomotionWeights,
)
from framework.config.sim_config.config_motor_model import motor_model_config
from framework.utils.quad_utils_locomotion.reference_generator_locomotion import reference_barell_roll

# Lift-off and landing base height relative to standing height [m]. Tuning knobs.
LIFTOFF_RISE = 0.07
LANDING_DROP = 0.05
# Joint-limit barrier: distance kept from the model limit [rad] and its weight.
JOINT_LIMIT_MARGIN = 0.1
JOINT_LIMIT_WEIGHT = 50.0
# Torque barrier: fraction of the actuator limit the plan may use, and its weight.
TORQUE_LIMIT_FRACTION = 0.9
TORQUE_LIMIT_WEIGHT = 10.0
SPEED_LIMIT_WEIGHT = 10.0


class Go2BarrelRoll(Go2Locomotion):
    dt = 0.01
    N = 100

    def __init__(self, robot: str = "go2_dls") -> None:
        super().__init__()
        if robot == "go2_dls":
            self.model_path = _DLS_MPC_MODEL_PATH
        self.weights = LocomotionWeights(
            pos=(0.0, 0.0, 5e4),
            rot=(100.0, 100.0, 100.0),
            joint_pos=1e2,
            lin_vel=100.0,
            ang_vel=(1e2, 1e2, 1e2),
            joint_vel=1e-1,
            torque=1e-1,
            grf=0.0,
            foot=(1e3, 1e3, 5e4),
        )
        h = self.robot_height
        self.reference = partial(
            reference_barell_roll,
            z_stand=h,
            z_start=h + LIFTOFF_RISE,
            z_land=h - LANDING_DROP,
        )

    @property
    def cost(self):
        # False: feet in the air are not pulled to a reference.
        base = partial(mpc_objectives.quadruped_wb_obj, False, self.n_joints, self.n_contact, self.N)
        # The shared objective has no joint limits. Without them the plan sends
        # the support hips to -1.6 rad (limit 1.05) and the simulator cannot follow.
        # It also allows 44 N m on every joint; the Go2 hip and thigh give 23.7.
        model = mujoco.MjModel.from_xml_path(self.model_path)
        limits = model.jnt_range[1 : 1 + self.n_joints]
        lo = jnp.asarray(limits[:, 0] + JOINT_LIMIT_MARGIN)
        hi = jnp.asarray(limits[:, 1] - JOINT_LIMIT_MARGIN)
        tau_max = jnp.asarray(model.actuator_ctrlrange[:, 1] * TORQUE_LIMIT_FRACTION)
        # The motor model gives full torque only below the corner speed.
        dq_max = jnp.tile(jnp.asarray(motor_model_config.corner_speed_rad_s), self.n_contact)
        nj = self.n_joints

        def cost(W, reference, x, u, t):
            q = x[7 : 7 + nj]
            dq = x[13 + nj : 13 + 2 * nj]
            tau = u[:nj]
            room = jnp.concatenate([q - lo, hi - q])
            tau_room = jnp.concatenate([tau + tau_max, tau_max - tau])
            dq_room = jnp.concatenate([dq + dq_max, dq_max - dq])
            return (
                base(W, reference, x, u, t)
                + jnp.sum(mpc_objectives.penalty(room, JOINT_LIMIT_WEIGHT, JOINT_LIMIT_MARGIN))
                + jnp.sum(mpc_objectives.penalty(tau_room, TORQUE_LIMIT_WEIGHT, 1.0))
                + jnp.sum(mpc_objectives.penalty(dq_room, SPEED_LIMIT_WEIGHT, 1.0))
            )

        return cost
