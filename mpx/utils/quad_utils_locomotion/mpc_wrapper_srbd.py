"""Single-rigid-body (SRBD) locomotion MPC with the same public API as ``MPCWrapper``."""

from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mpx.utils.mpc_wrapper import MPCData
from mpx.utils.quad_utils_locomotion.mpc_wrapper_locomotion import build_solver_step
from mpx.utils.quad_utils_locomotion.reference_generator_locomotion import (
    reference_generator_srbd,
    whole_body_interface,
)
from mpx.utils.simulation_utils.sim_utils import timer_run


@partial(jax.jit, static_argnums=(0, 1))
def _update_warm_start(horizon, shift, u_ref, x0, X_prev, U_prev, X, U, V):
    """Shift the SRBD (centroidal) solution; no joint slice."""

    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate([trajectory[shift:], tail], axis=0)

    def safe_update():
        return shift_trajectory(U), shift_trajectory(X), shift_trajectory(V)

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(X_prev),
        )

    valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
    return jax.lax.cond(valid_solution, safe_update, unsafe_update)


class SrbdMPCWrapper:
    """Centroidal SRBD QP plus Cartesian whole-body torque mapping.

    Public flow matches locomotion ``MPCWrapper``::

        data = wrapper.make_data()
        data, tau = wrapper.run(data, x0, command, contact)
    """

    def __init__(self, config, limited_memory=False):
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = max(1, int(1 / (config.dt * config.mpc_frequency)))
        self.default_contact = jnp.zeros(config.n_contact)
        self.qpos_slice = slice(0, 7 + config.n_joints)
        self.qvel_slice = slice(
            self.qpos_slice.stop, self.qpos_slice.stop + 6 + config.n_joints
        )
        self.foot_slice = slice(
            self.qvel_slice.stop,
            self.qvel_slice.stop + 3 * config.n_contact,
        )

        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)

        self.contact_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]
        self.body_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in config.body_name
        ]

        self.initial_state = jnp.asarray(config.initial_state)
        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))
        self.initial_liftoff = jnp.zeros(3 * config.n_contact)

        dynamics = config.dynamics(
            model=self.model,
            mjx_model=self.mjx_model,
            contact_id=self.contact_id,
            body_id=self.body_id,
        )
        _, solve = build_solver_step(
            config,
            config.cost,
            dynamics,
            config.hessian_approx,
            limited_memory,
        )
        self._solve = jax.jit(solve)

        use_terrain = bool(
            getattr(
                config,
                "use_terrain_estimator",
                getattr(config, "use_terrain_estimation", False),
            )
        )
        clearance_speed = getattr(
            config, "clearance_speed", getattr(config, "clearence_speed", 0.2)
        )
        self._ref_gen = jax.jit(
            partial(
                reference_generator_srbd,
                use_terrain,
                config.N,
                config.dt,
                config.n_contact,
                mass=config.mass,
                foot0=config.p_legs0,
                clearence_speed=clearance_speed,
            )
        )
        self._timer_run = jax.jit(timer_run)
        self._update_warm_start = partial(
            _update_warm_start,
            config.N,
            self.shift,
            config.u_ref,
        )

        sim_frequency = getattr(config, "whole_body_frequency", 500)
        self._whole_body_interface = jax.jit(
            partial(
                whole_body_interface,
                self.model,
                self.mjx_model,
                self.contact_id,
                self.body_id,
                sim_frequency,
                config.Kp,
                config.Kd,
            )
        )

    def pack_state(self, qpos, qvel, foot):
        return jnp.concatenate(
            [jnp.ravel(qpos), jnp.ravel(qvel), jnp.ravel(foot)]
        )

    def make_data(self):
        return MPCData(
            dt=self.config.dt,
            duty_factor=self.config.duty_factor,
            step_freq=self.config.step_freq,
            step_height=self.config.step_height,
            contact_time=self.config.timer_t,
            liftoff=self.initial_liftoff,
            X0=self.initial_X0,
            U0=self.initial_U0,
            V0=self.initial_V0,
            W=self.config.W,
        )

    def _centroidal(self, x0):
        qpos = x0[self.qpos_slice]
        qvel = x0[self.qvel_slice]
        foot = x0[self.foot_slice]
        x_srbd = jnp.concatenate([qpos[:7], qvel[:6]])
        return qpos, qvel, foot, x_srbd

    def _run_impl(self, data, x0, input, contact):
        qpos, qvel, foot, x_srbd = self._centroidal(x0)
        planned_contact, contact_time = self._timer_run(
            data.duty_factor,
            data.step_freq,
            data.contact_time,
            1 / self.mpc_frequency,
        )
        reference, parameter, liftoff, foot_ref_dot = self._ref_gen(
            t_timer=data.contact_time,
            x=x_srbd,
            foot=foot,
            input=input,
            duty_factor=data.duty_factor,
            step_freq=data.step_freq,
            step_height=data.step_height,
            liftoff=data.liftoff,
            contact=contact,
        )
        X, U, V = self._solve(
            reference,
            parameter,
            data.W,
            x_srbd,
            data.X0,
            data.U0,
            data.V0,
        )
        valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
        grf = jnp.where(valid_solution, U[0], data.U0[0])
        foot_ref = parameter[0, 4:]
        foot_dot = foot_ref_dot[0]
        tau, _ = self._whole_body_interface(
            qpos, qvel, grf, foot_ref, foot_dot, planned_contact
        )
        U0, X0, V0 = self._update_warm_start(
            x_srbd,
            data.X0,
            data.U0,
            X,
            U,
            V,
        )
        data = data.replace(
            X0=X0,
            U0=U0,
            V0=V0,
            contact_time=contact_time,
            liftoff=liftoff,
        )
        return data, tau

    def run(self, data, x0, input, contact=None):
        contact = self.default_contact if contact is None else jnp.asarray(contact)
        return self._run_impl(data, x0, input, contact)

    def reset(self, data, qpos, qvel, foot):
        qpos = jnp.ravel(qpos)
        qvel = jnp.ravel(qvel)
        x_srbd = jnp.concatenate([qpos[:7], qvel[:6]])
        return data.replace(
            U0=self.initial_U0,
            X0=jnp.tile(x_srbd, (self.config.N + 1, 1)),
            V0=self.initial_V0,
            contact_time=self.config.timer_t,
            liftoff=jnp.ravel(foot),
        )
