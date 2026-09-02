"""Locomotion / shared online MPC wrappers.

This module hosts the shared MPC machinery used by every behaviour:

- ``mpx_data``                  : PyTree carrying the per-environment MPC state.
- ``BatchedMPCControllerWrapper``: vectorised (multi-env) MPC runner.
- ``LocomotionMPCControllerWrapper``: single-environment online MPC. It owns the
  shared ``__init__``/``run``/``reset``/``runOffline`` path used by both locomotion
  and balance. Balance behaviour is layered on top in
  :mod:`mpx.utils.mpc_wrapper_balance` via :class:`BalanceMPCControllerWrapper`,
  which subclasses this wrapper and overrides the balance hooks.

The reference generator (locomotion vs balance) is selected automatically from the
controller ``config`` by :func:`mpx.utils.ref_gen_wrapper.whole_body_reference_partial`.
"""

import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from mpx.utils.ref_gen_wrapper import whole_body_reference_partial

import mujoco
from mujoco import mjx
import mpx.jax_ocp_solvers.optimizers as optimizers
from mujoco.mjx._src.dataclasses import PyTreeNode
from timeit import default_timer as timer
from mpx.utils.mpc_wrapper import mpx_data, MPCData
from mpx.utils.quad_utils_locomotion.reference_generator_locomotion import reference_generator_locomotion
from mpx.utils.simulation_utils.sim_utils import timer_run

def build_solver_step(config, cost, dynamics, hessian_approx, limited_memory):
    solver_mode = getattr(config, "solver_mode", "primal_dual")

    if solver_mode == "primal_dual":
        solver = partial(optimizers.mpc, cost, dynamics, hessian_approx, limited_memory)

        def solve(reference, parameter, W, x0, X0, U0, V0):
            return solver(reference, parameter, W, x0, X0, U0, V0)

        return solver_mode, solve

    if solver_mode == "fddp":
        solver = partial(
            optimizers.fddp_mpc,
            cost,
            dynamics,
            hessian_approx,
            limited_memory,
        )

        def solve(reference, parameter, W, x0, X0, U0, V0):
            X, U, defects = solver(reference, parameter, W, x0, X0, U0)
            return X, U, defects

        return solver_mode, solve

    raise ValueError(f"Unsupported MPC solver_mode: {solver_mode}")

@partial(jax.jit, static_argnums=(0, 1, 2))
def _update_warm_start(n_joints, horizon, shift, u_ref, x0, X_prev, U_prev, X, U, V):
    """Shift the solution for the next MPC step and extract the first command."""

    q_slice = slice(7, 7 + n_joints)
    dq_slice = slice(13 + n_joints, 13 + 2 * n_joints)
    u_fallback_idx = 1 if horizon > 1 else 0

    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate([trajectory[shift:], tail], axis=0)

    def safe_update():
        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            U[0, :n_joints],
            X[0, q_slice],
            X[1, dq_slice],
        )

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(X_prev),
            U_prev[u_fallback_idx, :n_joints],
            X_prev[1, q_slice],
            X_prev[1, dq_slice],
        )

    valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
    return jax.lax.cond(valid_solution, safe_update, unsafe_update)

class MPCWrapper:
    """Minimal MPC API built for `jit` and `vmap`.

    The public flow is:
    `data = wrapper.make_data()`
    `data, tau = wrapper.run(data, x0, command, contact)`

    The warm-start state always carries `V0`, even for direct solvers like FDDP.
    That keeps the pytree shape fixed and makes solver switching transparent to
    callers, batching, and JIT compilation.
    """

    def __init__(self, config, limited_memory=False):
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        self.default_contact = jnp.zeros(config.n_contact)
        self.qpos_slice = slice(0, 7 + config.n_joints)
        self.qvel_slice = slice(self.qpos_slice.stop, self.qpos_slice.stop + 6 + config.n_joints)
        self.foot_slice = slice(
            self.qvel_slice.stop,
            self.qvel_slice.stop + 3 * config.n_contact,
        )

        self.model = mujoco.MjModel.from_xml_path(config.model_path)
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

        self.cost = config.cost
        self.hessian_approx = config.hessian_approx
        self.dynamics = config.dynamics(
            model = self.model,
            mjx_model =self.mjx_model,
            contact_id =self.contact_id,
            body_id =self.body_id,
        )
   

        # The config owns the nominal state layout, including any extra states.
        self.initial_state = jnp.asarray(config.initial_state)

        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))
        self.initial_liftoff = jnp.zeros(3 * config.n_contact)

        self.solver_mode, solve = build_solver_step(
            config,
            self.cost,
            self.dynamics,
            self.hessian_approx,
            limited_memory,
        )
        self._solve = jax.jit(solve)

        reference_generator = getattr(config, "reference_generator", reference_generator_locomotion)
        clearance_speed = getattr(config, "clearance_speed", getattr(config, "clearence_speed", 0.2))
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
            config.N,
            self.shift,
            config.u_ref,
        )

    def make_data(self):
        """Allocate the pytree state used by the pure functional API."""

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

    def control_output(self, x0, X, U, reference, parameter):
        del x0, X, reference, parameter
        return U[0, : self.config.n_joints]

    def _run_impl(self, data, x0, input, contact):
        _, contact_time = self._timer_run(
            data.duty_factor,
            data.step_freq,
            data.contact_time,
            1 / self.mpc_frequency,
        )

        reference, parameter, liftoff = self._ref_gen(
            duty_factor=data.duty_factor,
            step_freq=data.step_freq,
            step_height=data.step_height,
            t_timer=data.contact_time,
            x=x0,
            foot=x0[self.foot_slice],
            input=input,
            liftoff=data.liftoff,
            contact=contact,
        )

        # Reference generation and solver execution stay on the pure JAX path.
        X, U, V = self._solve(
            reference,
            parameter,
            data.W,
            x0,
            data.X0,
            data.U0,
            data.V0,
        )
        valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
        tau = jax.lax.cond(
            valid_solution,
            lambda _: self.control_output(x0, X, U, reference, parameter),
            lambda _: self.control_output(x0, data.X0, data.U0, reference, parameter),
            operand=None,
        )
        # Shift the solution so the next call starts from the previous optimum.
        U0, X0, V0, _, q, dq = self._update_warm_start(
            x0,
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
        return data, tau, q, dq

    def run(self, data, x0, input, contact=None):
        """Run one MPC step and return the updated carry and torque command."""

        contact = self.default_contact if contact is None else jnp.asarray(contact)
        data, tau, _, _ = self._run_impl(data, x0, input, contact)
        return data, tau

    def reset(self, data, qpos, qvel, foot):
        """Reset the warm start around the provided measured state."""

        # Start from the config initial_state so any extra state entries keep
        # their configured default value.
        initial_state = (
            self.initial_state
            .at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
            .at[self.foot_slice].set(jnp.ravel(foot))
        )
        return data.replace(
            U0=self.initial_U0,
            X0=jnp.tile(initial_state, (self.config.N + 1, 1)),
            V0=self.initial_V0,
            contact_time=self.config.timer_t,
            liftoff=jnp.ravel(foot),
        )

    def foot_positions(self, qpos):
        """Return the flattened contact-point positions for the provided configuration."""

        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return jnp.array([self.data.geom_xpos[idx] for idx in self.contact_id_mj]).flatten()

    def runOffline(self, qpos, qvel, *, return_stats=False, verbose=True, max_iter=100):
        """Solve the fixed reference problem exposed by configs that define `reference`."""

        foot_op = self.foot_positions(qpos)
        x0 = (
            self.initial_state
            .at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
            .at[self.foot_slice].set(foot_op)
        )
        reference, parameter = self.config.reference(
            self.config.N + 1,
            self.config.dt,
            self.config.n_joints,
            self.config.n_contact,
            self.config.p_legs0,
            self.config.q0,
        )

        W = self.config.W

        # Keep the offline warm start aligned with the nominal reference seed.
        # The old wrapper started from `initial_X0`, not from the measured feet.
        X0 = self.initial_X0.at[:, : 13 + self.config.n_joints].set(
            reference[:, : 13 + self.config.n_joints]
        )
        U0 = self.initial_U0
        V0 = self.initial_V0

        X0, U0, _, output, stats = offline_solver.run_offline_solve(
            self._solve,
            self.cost,
            self.dynamics,
            self.config.solver_mode,
            reference,
            parameter,
            W,
            x0,
            X0,
            U0,
            V0,
            max_iter=max_iter,
            verbose=verbose,
        )

        if return_stats:
            return X0, U0, reference, output, stats
        return X0, U0, reference, output

# region Locomotion MPC Controller Wrapper ========================
# class LocomotionMPCControllerWrapper:
#     """
#     Single-environment online MPC wrapper (shared base for all behaviours).

#     This class owns the shared online MPC path used by both locomotion and
#     balance: ``__init__`` (solver/reference-generator setup and warm-start
#     buffers), ``run``, ``reset`` and ``runOffline``.

#     The reference generator is selected automatically (locomotion vs balance)
#     from ``config`` by :func:`whole_body_reference_partial`. Reduced-support
#     balance is implemented in :class:`BalanceMPCControllerWrapper`, which
#     subclasses this wrapper and adds balance buffers, ``reset`` hooks, and the
#     swing-foot / base-pose API.
#     """

#     def __init__(self, config,limited_memory=False):
#         """
#         Initializes the MPC controller wrapper.

#         Args:
#             config: Configuration object containing MPC and gait parameters.
#             mpc_frequency: Frequency (Hz) at which MPC updates occur.
#         """
#         jax.config.update("jax_compilation_cache_dir", "./jax_cache")
#         jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
#         jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

#         self.model = mujoco.MjModel.from_xml_path(config.model_path)
#         self.data = mujoco.MjData(self.model)
#         mujoco.mj_fwdPosition(self.model, self.data)
#         robot_mass = self.data.qM[0]
#         mjx_model = mjx.put_model(self.model)
#         self.config = config
#         self.mpc_frequency = config.mpc_frequency
#         self.shift = int(1 / (config.dt * config.mpc_frequency))

#         # Timer and liftoff states for the reference generator.
#         self.foot0 = config.p_legs0.copy()  # Initial foot positions (could be adjusted if needed)
#         self.q0 = config.q0.copy()          # Initial joint configuration

#         if self.config.grf_as_state:
#             self.initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
#         else:
#             self.initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0])

#         # Get contact and body IDs from configuration
#         self.contact_id = []
#         for name in config.contact_frame:
#             self.contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
#         self.body_id = []
#         for name in config.body_name:
#             self.body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))
#         # Trajectory warm-start variables (used between MPC calls)
#         self.U0 = jnp.tile(config.u_ref, (config.N, 1))
#         self.X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
#         self.V0 = jnp.zeros((config.N + 1, config.n))

#         # Define cost, hessian approximation, and dynamics functions for MPC.
#         self.cost = partial(config.cost,config.n_joints, config.n_contact, config.N)
#         hessian_approx = partial(config.hessian_approx,config.n_joints, config.n_contact)
#         self.dynamics = partial(config.dynamics,
#                                 self.model, mjx_model, self.contact_id, self.body_id,
#                                 config.n_joints, config.dt)

#         work = partial(optimizers.mpc, self.cost, self.dynamics, hessian_approx, limited_memory)

#         reference_generator = whole_body_reference_partial(
#             config, robot_mass, clearence_speed_preset=None
#         )

#         self._solve = jax.jit(work)
#         self._ref_gen = jax.jit(reference_generator)
#         self._timer_run = jax.jit(timer_run)


#         self.contact_time = config.timer_t
#         self.liftoff = config.p_legs0.copy()

#         self.tau = jnp.zeros(config.n_joints)
#         self.q = jnp.zeros(config.n_joints)
#         self.dq = jnp.zeros(config.n_joints)

#         @partial(jax.jit, static_argnums=(0,1))
#         def update_and_extract_helper(n_joints,shift,U, X, V, x0, X0, U0):
#             def safe_update():
#                 new_U0 = jnp.concatenate([U[shift:], jnp.tile(U[-1:], (shift, 1))])
#                 new_X0 = jnp.concatenate([X[shift:], jnp.tile(X[-1:], (shift, 1))])
#                 new_V0 = jnp.concatenate([V[shift:], jnp.tile(V[-1:], (shift, 1))])
#                 tau = U[0, :n_joints]
#                 # Use the next planned knot (X[1]) as the joint position target so the
#                 # low-level PD tracks the MPC posture. X[0] is pinned to the current
#                 # measured state, which makes the PD position term ~0 → the base is held
#                 # by feedforward torque alone and slowly sinks under model mismatch.
#                 q = X[1, 7:n_joints + 7]
#                 dq = X[1, 13 + n_joints:2 * n_joints + 13]
#                 return new_U0, new_X0, new_V0,tau,q ,dq
#             def unsafe_update():
#                 new_U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
#                 new_X0 = jnp.tile(x0, (self.config.N + 1, 1))
#                 new_V0 = jnp.zeros((self.config.N + 1, self.config.n ))
#                 tau = U0[1, :n_joints]
#                 q = X0[1, 7:n_joints + 7]
#                 dq = X0[1, 13 + n_joints:2 * n_joints + 13]
#                 return new_U0, new_X0, new_V0, tau, q, dq

#             return jax.lax.cond(jnp.isnan(U[0,0]),unsafe_update,safe_update)
#         update_and_extract = partial(update_and_extract_helper,self.config.n_joints,self.shift)
#         self.update_and_extract = jax.jit(update_and_extract)

#         # ---------------------------------------------------------------------
#         # Shared gait/runtime parameters used by the online reference generator.
#         # ---------------------------------------------------------------------
#         self.duty_factor = config.duty_factor
#         self.step_freq = config.step_freq
#         self.step_height = config.step_height
#         self.robot_height = (
#             float(config.robot_height)
#             if getattr(config, "behaviour", None) == "locomotion"
#             else float(config.initial_height)
#         )
#         self.tau0 = np.zeros(config.n_joints)
#         self.start_time = 0
#         self.contact = np.zeros(config.n_contact)
#         self.last_base_ref_pos = np.asarray(config.p0, dtype=np.float64).copy()
#         self.last_base_ref_quat = np.asarray(config.quat0, dtype=np.float64).copy()
#         self.last_foot_ref = np.zeros(3 * config.n_contact, dtype=np.float64)
#         self.clearence_speed = 0.4
#         self.p_collision = np.zeros(3 * config.n_contact)
#         self.collision = [0, 0, 0, 0]
#         self.collision_cycle = np.zeros(config.n_contact)

#     def _balance_ref_gen_kwargs(self):
#         """Neutral balance-anchor kwargs for locomotion (ignored by locomotion ref gen)."""
#         nc = self.config.n_contact
#         return {
#             "foot_ref_anchor": jnp.zeros(3 * nc, dtype=jnp.float32),
#             "use_foot_ref_anchor": False,
#             "base_quat_ref": jnp.asarray(self.config.quat0, dtype=jnp.float32),
#             "use_base_quat_ref": False,
#         }

#     # -------------------------------------------------------------------------
#     # Locomotion / shared online MPC path (used by locomotion and balance).
#     # -------------------------------------------------------------------------
#     def run(self, qpos, qvel, input, contact):
#         """
#         Run one online MPC step and return joint commands.

#         Args:
#             qpos: Generalized position.
#             qvel: Generalized velocity.
#             input: Control input vector.
#             contact: Contact state vector.

#         Returns:
#             A tuple (tau, q, dq) representing the computed joint torques, joint positions, and joint velocities.
#         """
#         self.contact = contact.copy()
#         #get forward kinematics for foot position

#         self.data.qpos = qpos

#         mujoco.mj_kinematics(self.model, self.data) # update the forward kinematics
#         foot_op = np.array([self.data.geom_xpos[self.contact_id[i]] for i in range(self.config.n_contact)]).flatten() # foot positions in the world frame
#         #set initial state
#         input[6] = self.robot_height

#         if self.config.grf_as_state:
#             x0 = jnp.concatenate([qpos, qvel,foot_op,jnp.zeros(3*self.config.n_contact)])
#         else:
#             x0 = jnp.concatenate([qpos, qvel,foot_op])

        
#         contact = jnp.array(contact)

#         # Update the timer state for the gait reference.
#         _ , self.contact_time = self._timer_run(self.duty_factor,self.step_freq,self.contact_time,1/self.mpc_frequency)
#         input = jnp.array(input)
#         # Reference generator is selected automatically (locomotion vs balance) from config.
#         # Balance-only anchors are ignored by locomotion reference generation.
#         reference, parameter, self.liftoff = self._ref_gen(
#             duty_factor = self.duty_factor,
#             step_freq = self.step_freq,
#             step_height = self.step_height,
#             t_timer = self.contact_time.copy(),
#             x = x0,
#             foot = foot_op,
#             input = input,
#             liftoff = self.liftoff,
#             contact = contact,
#             clearence_speed = self.clearence_speed,
#             **self._balance_ref_gen_kwargs(),
#         )
#         self.last_base_ref_pos = np.asarray(reference[0, :3], dtype=np.float64).copy()
#         self.last_base_ref_quat = np.asarray(reference[0, 3:7], dtype=np.float64).copy()
#         nj = self.config.n_joints
#         nc = self.config.n_contact
#         i0 = 13 + nj
#         self.last_foot_ref = np.asarray(reference[0, i0 : i0 + 3 * nc], dtype=np.float64)

#         # Execute the MPC optimization.
#         X, U, V = self._solve(
#             reference,
#             parameter,
#             self.config.W,
#             x0,
#             self.X0,
#             self.U0,
#             self.V0
#             )

#         # # Warm-start for the next call: shift trajectories forward.

#         self.U0, self.X0, self.V0, tau_temp, q_temp, dq_temp = self.update_and_extract(U, X, V, x0, self.X0, self.U0)

#         # TO DO change to values from config
#         tau = np.clip(np.array(tau_temp),self.config.min_torque,self.config.max_torque)
#         q = np.array(q_temp)
#         dq = np.array(dq_temp)

#         return tau, q, dq 

#     def reset(self,qpos,qvel):
#         """
#         Reset online MPC warm-start buffers and mode-specific anchors.
#         """
#         self.data.qpos = qpos
#         # self.data.qvel = qvel
#         mujoco.mj_kinematics(self.model, self.data)
#         foot_op = np.array([self.data.geom_xpos[self.contact_id[i]] for i in range(self.config.n_contact)])
#         if self.config.grf_as_state:
#             x0 = jnp.concatenate([qpos, qvel,foot_op.flatten(),jnp.zeros(3*self.config.n_contact)])
#         else:
#             x0 = jnp.concatenate([qpos, qvel,foot_op.flatten()])
#         self.contact_time = self.config.timer_t
#         self.liftoff = foot_op.flatten()
#         self.last_foot_ref = foot_op.flatten().astype(np.float64)
#         self.U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
#         self.X0 = jnp.tile(x0, (self.config.N + 1, 1))
#         self.V0 = jnp.zeros((self.config.N + 1, self.config.n))
#         print("MPC Controller Reset")
#         return

#     # -------------------------------------------------------------------------
#     # Offline debugging / batch optimization helper.
#     # -------------------------------------------------------------------------
#     def runOffline(self, qpos, qvel):
#         """
#         Runs one MPC update using the current state, input, and foot positions.

#         Args:
#             x0: Current system state vector.
#             input: Input
#             foot_op: Flattened current foot positions vector.

#         Returns:
#             A tuple (X, U, V) representing the computed state trajectory, control sequence,
#             and auxiliary variable trajectory.
#         """
#         #compensate for the time delay
#         #get forward kinematics for foot position

#         self.data.qpos = qpos

#         mujoco.mj_kinematics(self.model, self.data)
#         foot_op = np.array([self.data.geom_xpos[self.contact_id[i]] for i in range(self.config.n_contact)])
#         #set initial state

#         x0 = jnp.concatenate([qpos, qvel,foot_op.flatten(),jnp.zeros(3*self.config.n_contact)])



#         reference, parameter = self.config.reference(self.config.N + 1,self.config.dt,self.config.n_joints,self.config.n_contact,self.config.p_legs0,self.config.q0)

#         reference = jnp.array(reference)
#         parameter = jnp.array(parameter)
#         # Warm start
#         self.X0 = self.X0.at[:,:13+self.config.n_joints].set(reference[:,:13+self.config.n_joints])

#         _cost = partial(self.cost,self.config.W,reference)
#         _dynamics = partial(self.dynamics,parameter=parameter)
#         model_evaluator = partial(optimizers.model_evaluator_helper, _cost, _dynamics,x0)
#         jitted_model_evaluator = jax.jit(model_evaluator)

#         _exit = False
#         max_iter = 100
#         last_cost = 1e10
#         i = 0
#         output = []
#         output.append((self.X0))
#         while not _exit:
#             start = timer()

#             X, U, V = self._solve(
#                 reference,
#                 parameter,
#                 self.config.W,
#                 x0,
#                 self.X0,
#                 self.U0,
#                 self.V0
#                 )

#             X.block_until_ready()

#             self.X0 = X
#             self.U0 = U
#             self.V0 = V

#             output.append((self.X0))

#             g, c = jitted_model_evaluator(X,U)

#             stop = timer()

#             l2_cost = np.sum(g*g)

#             if i == 0:
#                 print("{:<10} {:<20} {:<20} {:<20}".format("Iter", "Cost", "Constraint", "Time Elapsed"))
#             print("{:<10d} {:<20.5f} {:<20.5f} {:<20.5f}".format(i, l2_cost, np.sum(c*c), stop-start))
#             i += 1

#             if i > max_iter:
#                 _exit = True
#             if last_cost - l2_cost < 1e-3 and np.sum(c*c) < 1e-5:
#                 _exit = True
#             last_cost = l2_cost

#         return self.X0, self.U0, reference, output
#endregion