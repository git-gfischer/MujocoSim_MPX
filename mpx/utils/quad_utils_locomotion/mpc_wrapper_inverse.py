"""Equality-constrained inverse-dynamics locomotion MPC."""

from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src.dataclasses import PyTreeNode

import mpx.jax_ocp_solvers.optimizers as optimizers
from mpx.utils.mpc_wrapper import MPCData
from mpx.utils.quad_utils_locomotion.reference_generator_locomotion import (
    reference_generator_locomotion,
)
from mpx.utils.quadruped_dyn_models.inverse import (
    inv_dyn_control_slices,
    inv_dyn_dims,
    pack_inverse_state,
    quadruped_inv_dyn_foot_positions,
)
from mpx.utils.simulation_utils.sim_utils import timer_run


class InverseDynamicsMPCData(MPCData):
    """Locomotion MPC carry plus equality-multiplier warm-start."""

    Veq0: jnp.ndarray
    regularization: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _update_warm_start(n_joints, nq, nv, horizon, shift, u_ref, x0, X_prev, U_prev, X, U, V, Veq):
    q_slice = slice(7, 7 + n_joints)
    dq_slice = slice(nq + 6, nq + nv)

    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate([trajectory[shift:], tail], axis=0)

    def safe_update():
        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            shift_trajectory(Veq),
            X[1, q_slice],
            X[1, dq_slice],
        )

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(X_prev),
            jnp.zeros((horizon, nv), dtype=X_prev.dtype),
            X_prev[1, q_slice],
            X_prev[1, dq_slice],
        )

    valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
    return jax.lax.cond(valid_solution, safe_update, unsafe_update)


class InverseDynamicsMPCWrapper:
    """Same public flow as locomotion ``MPCWrapper``: ``make_data`` / ``run`` / ``reset``."""

    def __init__(self, config, limited_memory=False):
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = max(1, int(1 / (config.dt * config.mpc_frequency)))
        self.default_contact = jnp.zeros(config.n_contact)
        nq, nv, n, m, equality_dim = inv_dyn_dims(config.n_joints, config.n_contact)
        self.nq = nq
        self.nv = nv
        self.n = n
        self.m = m
        self.equality_dim = equality_dim
        self.qpos_slice = slice(0, nq)
        self.qvel_slice = slice(nq, nq + nv)
        self.foot_slice = slice(nq + nv, nq + nv + 3 * config.n_contact)
        self.qacc_slice, self.tau_slice, self.grf_slice = inv_dyn_control_slices(
            config.n_joints, config.n_contact
        )

        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.model.opt.timestep = config.dt
        data = mujoco.MjData(self.model)
        mujoco.mj_fwdPosition(self.model, data)
        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)
        robot_mass = data.M[0]

        self.contact_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]
        self.body_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in config.body_name
        ]
        self.contact_id_mj = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]

        self.dynamics = config.dynamics(
            model=self.model,
            mjx_model=self.mjx_model,
            contact_id=self.contact_id,
            body_id=self.body_id,
        )
        self.equality = config.equality(
            model=self.model,
            mjx_model=self.mjx_model,
            contact_id=self.contact_id,
            body_id=self.body_id,
        )
        self.cost = config.cost(
            model=self.model,
            mjx_model=self.mjx_model,
            contact_id=self.contact_id,
            body_id=self.body_id,
        )
        self.hessian_approx = config.hessian_approx

        self.initial_state = jnp.asarray(config.initial_state)
        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))
        self.initial_Veq0 = jnp.zeros((config.N, equality_dim))
        self.initial_liftoff = jnp.zeros(3 * config.n_contact)

        solver = partial(
            optimizers.mpc_equality,
            self.cost,
            self.dynamics,
            self.hessian_approx,
            limited_memory,
            equality=self.equality,
            num_alpha=getattr(config, "equality_num_alpha", 10),
        )

        def solve(reference, parameter, W, x0, X0, U0, V0, Veq0, regularization):
            return solver(
                reference,
                parameter,
                W,
                x0,
                X0,
                U0,
                V0,
                Veq_in=Veq0,
                regularization=regularization,
            )

        self._solve = jax.jit(solve)

        reference_generator = getattr(
            config, "reference_generator", reference_generator_locomotion
        )
        clearance_speed = getattr(
            config, "clearance_speed", getattr(config, "clearence_speed", 0.2)
        )
        self._ref_gen = jax.jit(
            partial(
                reference_generator,
                config.use_terrain_estimation,
                config.N,
                config.dt,
                config.n_joints,
                config.n_contact,
                robot_mass,
                foot0=config.p_legs0,
                q0=config.q0,
                clearence_speed=clearance_speed,
            )
        )
        self._timer_run = jax.jit(timer_run)
        self._update_warm_start = partial(
            _update_warm_start,
            config.n_joints,
            nq,
            nv,
            config.N,
            self.shift,
            config.u_ref,
        )

    def pack_state(self, qpos, qvel, foot):
        return jnp.concatenate(
            [jnp.ravel(qpos), jnp.ravel(qvel), jnp.ravel(foot)]
        )

    def make_data(self):
        regularization = jnp.asarray(
            getattr(self.config, "regularization", 1e-6), dtype=jnp.float32
        )
        return InverseDynamicsMPCData(
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
            Veq0=self.initial_Veq0,
            regularization=regularization,
        )

    def control_output(self, x0, X, U, reference, parameter):
        del x0, X, reference, parameter
        tau = U[0, self.tau_slice]
        return jnp.clip(tau, self.config.min_torque, self.config.max_torque)

    def _split_x0(self, x0):
        x = x0[: self.n]
        if x0.shape[-1] > self.n:
            foot = x0[self.foot_slice]
        else:
            qpos, qvel = x[self.qpos_slice], x[self.qvel_slice]
            foot = quadruped_inv_dyn_foot_positions(
                self.mjx_model, self.contact_id, qpos, qvel
            )
        return x, foot

    def _run_impl(self, data, x0, input, contact):
        _, contact_time = self._timer_run(
            data.duty_factor,
            data.step_freq,
            data.contact_time,
            1 / self.mpc_frequency,
        )
        x, foot = self._split_x0(x0)
        reference, parameter, liftoff = self._ref_gen(
            duty_factor=data.duty_factor,
            step_freq=data.step_freq,
            step_height=data.step_height,
            t_timer=data.contact_time,
            x=x,
            foot=foot,
            input=input,
            liftoff=data.liftoff,
            contact=contact,
        )
        X, U, V, Veq, regularization, alpha_best, any_accepted = self._solve(
            reference,
            parameter,
            data.W,
            x,
            data.X0,
            data.U0,
            data.V0,
            data.Veq0,
            data.regularization,
        )
        del alpha_best, any_accepted
        valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
        tau = jax.lax.cond(
            valid_solution,
            lambda _: self.control_output(x, X, U, reference, parameter),
            lambda _: self.control_output(x, data.X0, data.U0, reference, parameter),
            operand=None,
        )
        U0, X0, V0, Veq0, q, dq = self._update_warm_start(
            x,
            data.X0,
            data.U0,
            X,
            U,
            V,
            Veq,
        )
        data = data.replace(
            X0=X0,
            U0=U0,
            V0=V0,
            Veq0=Veq0,
            contact_time=contact_time,
            liftoff=liftoff,
            regularization=regularization,
        )
        return data, tau, q, dq

    def run(self, data, x0, input, contact=None):
        contact = self.default_contact if contact is None else jnp.asarray(contact)
        data, tau, _, _ = self._run_impl(data, x0, input, contact)
        return data, tau

    def reset(self, data, qpos, qvel, foot):
        x = pack_inverse_state(qpos, qvel)
        return data.replace(
            U0=self.initial_U0,
            X0=jnp.tile(x, (self.config.N + 1, 1)),
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            contact_time=self.config.timer_t,
            liftoff=jnp.ravel(foot),
        )

    def foot_positions(self, qpos):
        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return jnp.array(
            [self.data.geom_xpos[idx] for idx in self.contact_id_mj]
        ).flatten()


def make_locomotion_mpc(config, limited_memory=True):
    """Build the locomotion MPC wrapper for ``config.mpc_model``."""
    if getattr(config, "mpc_model", "whole_body") == "inverse_dynamics":
        return InverseDynamicsMPCWrapper(config, limited_memory=limited_memory)
    from mpx.utils.quad_utils_locomotion.mpc_wrapper_locomotion import MPCWrapper

    return MPCWrapper(config, limited_memory=limited_memory)
