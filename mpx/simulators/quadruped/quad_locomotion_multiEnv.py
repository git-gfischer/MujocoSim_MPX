"""Multi-environment locomotion simulator for the Go2 quadruped.

N independent robots run in parallel on the GPU via JAX/MJX.  Each robot has
its own MPC warm-start (``batch_mpc_data``) and its own ``PointNavigator`` goal.
The viewer shows all robots simultaneously as ghost overlays on a single CPU
``MjModel`` using ``render_ghost_robot``, arranged in a square grid.

Navigation modes (``--nav``):
  random     — each robot gets an independent random goal; auto-resampled on arrival.
  vel        — all robots share the same keyboard velocity command (for debugging).

Usage::

    python -m mpx.simulators.quadruped.quad_locomotion_multiEnv --n-env 8
    python -m mpx.simulators.quadruped.quad_locomotion_multiEnv --n-env 16 --nav random
    python -m mpx.simulators.quadruped.quad_locomotion_multiEnv --headless --n-env 64
"""

import argparse
import math
import os
import sys
import time
from functools import partial
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from mpx.utils.quad_utils_locomotion.mpc_wrapper_locomotion import MPCWrapper

from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation

from mpx.utils.simulation_utils.console import KeyboardVelocityCommand
import mpx.utils.simulation_utils.sim_utils as sim_utils

from mpx.navigation.pointNav import PointNavigator
from mpx.estimators.quad_contact_estimation import estimate_contacts


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _robot_config(robot: str):
    if robot == "go2":
        from mpx.config.robot_config.config_go2 import go2_config, Go2Mode
        return go2_config(Go2Mode.LOCOMOTION)
    raise ValueError(f"Unsupported robot: {robot}")


def _is_crashed_batch(qpos_batch: jnp.ndarray, height_threshold: float, tilt_rad: float) -> np.ndarray:
    """Return a bool array of shape (N,) — True where robot i has crashed."""
    qpos = np.asarray(qpos_batch)
    crashed = np.zeros(qpos.shape[0], dtype=bool)
    for i in range(qpos.shape[0]):
        if qpos[i, 2] < height_threshold:
            crashed[i] = True
            continue
        w, x, y, z = qpos[i, 3], qpos[i, 4], qpos[i, 5], qpos[i, 6]
        roll  = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2.0 * (w*y - z*x), -1.0, 1.0))
        if abs(roll) > tilt_rad or abs(pitch) > tilt_rad:
            crashed[i] = True
    return crashed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    headless: bool = False,
    steps: int = 2000,
    scene: str = "flat",
    robot: str = "go2",
    nav: str = "random",
    n_env: int = 8,
):
    # ── Config & CPU model ──────────────────────────────────────────────────
    config = _robot_config(robot)

    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../../data/{robot}/scene_{scene}.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1.0 / sim_frequency

    # ── MJX model & batched data ─────────────────────────────────────────────
    mjx_model = mjx.put_model(model)

    # Nominal starting pose for every robot (same qpos0, grid-separated visually).
    qpos0_single = np.concatenate([
        np.asarray(config.p0),
        np.asarray(config.quat0),
        np.asarray(config.q0),
    ]).astype(np.float64)

    data.qpos = qpos0_single
    mujoco.mj_forward(model, data)
    mjx_data_template = mjx.put_data(model, data)

    # All robots start at the same pose; the grid offset is added only for rendering.
    qpos0_batch = jnp.tile(jnp.asarray(qpos0_single), (n_env, 1))
    batch_data  = jax.vmap(lambda qp: mjx_data_template.replace(qpos=qp))(qpos0_batch)

    # ── MJX contact IDs ─────────────────────────────────────────────────────
    mjx_contact_ids = [
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in config.contact_frame
    ]

    # ── Batched MPC ──────────────────────────────────────────────────────────
    mpc = MPCWrapper(config, limited_memory=True)

    # batch_mpc_data: every field gains a leading (N,) axis via vmap.
    batch_mpc_data = jax.vmap(lambda _: mpc.make_data())(jnp.arange(n_env))

    def _run_one(mpc_data_i, x0_i, command_i):
        return mpc.run(mpc_data_i, x0_i, command_i)

    batched_solve = jax.jit(jax.vmap(_run_one))
    batched_reset = jax.jit(jax.vmap(mpc.reset, in_axes=(0, 0, 0, 0)))

    # ── Per-robot state builders (vmapped) ───────────────────────────────────
    def _build_x0(mjx_d):
        foot_pos = jnp.array(
            [mjx_d.geom_xpos[mjx_contact_ids[k]] for k in range(config.n_contact)]
        ).flatten()
        return (
            mpc.initial_state
            .at[mpc.qpos_slice].set(mjx_d.qpos)
            .at[mpc.qvel_slice].set(mjx_d.qvel)
            .at[mpc.foot_slice].set(foot_pos)
        ), foot_pos

    build_x0_batch = jax.jit(jax.vmap(_build_x0))

    # ── MJX physics step ─────────────────────────────────────────────────────
    def _mjx_step(mjx_d, action):
        return mjx.step(mjx_model, mjx_d.replace(ctrl=action))

    batched_step = jax.jit(jax.vmap(_mjx_step))

    # ── Navigators (one per robot, pure-Python) ───────────────────────────────
    navigators = [
        PointNavigator(robot_height=config.robot_height, auto_resample=(nav == "random"))
        for _ in range(n_env)
    ]
    use_navigation = (nav == "random")
    command_handle = KeyboardVelocityCommand()  # shared keyboard fallback (nav=="vel")

    def _build_batch_command(qpos_batch: jnp.ndarray) -> jnp.ndarray:
        """Compute the (N, 7) MPC command array from navigator states."""
        qpos_np = np.asarray(qpos_batch)
        commands = np.array([
            navigators[i].mpc_input(qpos_np[i], config.robot_height)
            for i in range(n_env)
        ], dtype=np.float64)
        return jnp.asarray(commands)

    def _build_keyboard_command() -> jnp.ndarray:
        """Broadcast one keyboard command to all robots."""
        cmd = command_handle.mpc_input(config.robot_height)
        return jnp.tile(jnp.asarray(cmd), (n_env, 1))

    # ── Base-force perturbations (one per robot) ──────────────────────────────
    perturbers = [
        RandomBaseForcePerturbation.from_config(
            sim_dt=1.0 / sim_frequency,
            cfg=ext_base_force_config,
        )
        for _ in range(n_env)
    ]

    # ── Viewer grid offsets (purely for visualisation) ────────────────────────
    robots_per_row = math.ceil(math.sqrt(n_env))
    grid = np.arange(robots_per_row ** 2)
    offset_xy = np.stack([
        (grid % robots_per_row)[:n_env] * 1.5,
        (grid // robots_per_row)[:n_env] * 1.5,
    ], axis=1)  # (N, 2)  — 1.5 m spacing so robots don't visually overlap

    # ── Crash detection thresholds ────────────────────────────────────────────
    CRASH_HEIGHT = config.robot_height * 0.5
    CRASH_TILT   = np.deg2rad(60.0)

    # ── Warm-up: compile all JIT/vmap kernels ────────────────────────────────
    print(f"[multiEnv] Warming up {n_env} environments …", flush=True)
    batch_x0, batch_foot = build_x0_batch(batch_data)
    batch_mpc_data = batched_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, batch_foot)
    warm_cmd = jnp.tile(
        jnp.asarray(command_handle.mpc_input(config.robot_height)), (n_env, 1)
    )
    batch_mpc_data, tau_batch = batched_solve(batch_mpc_data, batch_x0, warm_cmd)
    tau_batch.block_until_ready()

    # Reset after warm-up so all robots start fresh.
    batch_data = jax.vmap(lambda qp: mjx_data_template.replace(qpos=qp))(qpos0_batch)
    batch_x0, batch_foot = build_x0_batch(batch_data)
    batch_mpc_data = batched_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, batch_foot)
    tau_batch = jnp.zeros((n_env, config.n_joints))

    # Seed navigators with goals relative to each robot's starting XY.
    qpos_np = np.array(batch_data.qpos)
    for i, nav_i in enumerate(navigators):
        nav_i.reset(qpos_np[i])

    for p in perturbers:
        p.reset()

    period = int(sim_frequency / config.mpc_frequency)
    counter = 0
    print(
        f"[multiEnv] {n_env} robots | period {period} steps | "
        f"nav={nav} | scene={scene}",
        flush=True,
    )

    # ── Headless loop ─────────────────────────────────────────────────────────
    if headless:
        for _ in range(steps):
            if counter % period == 0:
                batch_x0, _ = build_x0_batch(batch_data)
                if use_navigation:
                    batch_cmd = _build_batch_command(batch_data.qpos)
                else:
                    batch_cmd = _build_keyboard_command()
                start = timer()
                batch_mpc_data, tau_batch = batched_solve(batch_mpc_data, batch_x0, batch_cmd)
                tau_batch.block_until_ready()
                print(f"  step {counter:5d}  MPC {1e3*(timer()-start):.1f} ms", flush=True)

            batch_data = batched_step(batch_data, tau_batch)

            # Per-robot crash reset.
            crashed = _is_crashed_batch(batch_data.qpos, CRASH_HEIGHT, CRASH_TILT)
            if crashed.any():
                qpos_np = np.array(batch_data.qpos)
                qvel_np = np.zeros_like(qpos_np[:, :len(qpos0_single) - 7])
                for i in np.where(crashed)[0]:
                    qpos_np[i] = qpos0_single
                    navigators[i].reset(qpos_np[i])
                    perturbers[i].reset()
                new_qpos = jnp.asarray(qpos_np)
                batch_data = jax.vmap(
                    lambda qp: mjx_data_template.replace(
                        qpos=qp,
                        qvel=jnp.zeros(6 + config.n_joints),
                        ctrl=jnp.zeros(config.n_joints),
                    )
                )(new_qpos)
                bx, bf = build_x0_batch(batch_data)
                batch_mpc_data = batched_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, bf)

            counter += 1
        return

    # ── Viewer loop ───────────────────────────────────────────────────────────
    # Allocate ghost-robot slots for all N robots up-front.
    scratch_data = mujoco.MjData(model)
    ghost_geoms = [None] * n_env

    def key_callback(key: int):
        command_handle.key_callback(key)

    _goal_geom_ids = [-1] * n_env   # per-robot nav-goal sphere IDs

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        # Build ghost geom caches before the loop.
        for i in range(n_env):
            qp = np.asarray(batch_data.qpos[i])
            scratch_data.qpos[:len(qpos0_single)] = qp
            mujoco.mj_forward(model, scratch_data)
            ghost_geoms[i] = sim_utils.render_ghost_robot(viewer, model, scratch_data, alpha=0.9)
        viewer.sync()

        while viewer.is_running():
            tic = timer()

            # ── MPC solve (every period steps) ───────────────────────────────
            if counter % period == 0:
                batch_x0, _ = build_x0_batch(batch_data)
                if use_navigation:
                    batch_cmd = _build_batch_command(batch_data.qpos)
                else:
                    batch_cmd = _build_keyboard_command()

                start = timer()
                batch_mpc_data, tau_batch = batched_solve(batch_mpc_data, batch_x0, batch_cmd)
                tau_batch.block_until_ready()
                print(f"  step {counter:5d}  batched MPC {1e3*(timer()-start):.1f} ms", flush=True)

            # ── Physics step ─────────────────────────────────────────────────
            batch_data = batched_step(batch_data, tau_batch)

            # ── Crash reset ──────────────────────────────────────────────────
            crashed = _is_crashed_batch(batch_data.qpos, CRASH_HEIGHT, CRASH_TILT)
            if crashed.any():
                qpos_np = np.array(batch_data.qpos)
                for i in np.where(crashed)[0]:
                    qpos_np[i] = qpos0_single
                    navigators[i].reset(qpos_np[i])
                    perturbers[i].reset()
                    print(f"  [crash] robot {i} respawned", flush=True)
                batch_data = jax.vmap(
                    lambda qp: mjx_data_template.replace(
                        qpos=qp,
                        qvel=jnp.zeros(6 + config.n_joints),
                        ctrl=jnp.zeros(config.n_joints),
                    )
                )(jnp.asarray(qpos_np))
                bx, bf = build_x0_batch(batch_data)
                batch_mpc_data = batched_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, bf)

            # ── Navigator update (auto-resample on goal arrival) ─────────────
            if use_navigation:
                qpos_np = np.array(batch_data.qpos)
                for i, nav_i in enumerate(navigators):
                    nav_i.update(qpos_np[i], viewer=None)   # no lookat needed (random mode)

            # ── Render all robots as ghost overlays ──────────────────────────
            qpos_np = np.array(batch_data.qpos)
            for i in range(n_env):
                # Apply grid offset only in XY for visual separation.
                qp_vis = qpos_np[i].copy()
                qp_vis[0] += offset_xy[i, 0]
                qp_vis[1] += offset_xy[i, 1]
                scratch_data.qpos[:len(qpos0_single)] = qp_vis
                mujoco.mj_forward(model, scratch_data)
                ghost_geoms[i] = sim_utils.render_ghost_robot(
                    viewer, model, scratch_data, alpha=0.9, ghost_geoms=ghost_geoms[i]
                )

            # ── Render per-robot navigation goals ────────────────────────────
            if use_navigation:
                for i, nav_i in enumerate(navigators):
                    if nav_i._has_goal:
                        goal_pos = np.array([
                            nav_i.goal_xy[0] + offset_xy[i, 0],
                            nav_i.goal_xy[1] + offset_xy[i, 1],
                            nav_i.ground_z + 0.05,
                        ])
                        _goal_geom_ids[i] = sim_utils.render_sphere(
                            viewer,
                            position=goal_pos,
                            diameter=2.0 * nav_i.goal_tolerance,
                            color=np.array([0.1, 0.8, 0.2, 0.6]),
                            geom_id=_goal_geom_ids[i],
                        )

            counter += 1
            toc = timer()
            if toc - tic < model.opt.timestep:
                time.sleep(model.opt.timestep - (toc - tic))
            viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-environment Go2 locomotion with individual navigation goals."
    )
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of simulation steps (headless only).")
    parser.add_argument("--scene", type=str,
                        choices=["flat", "rough", "perlin", "stairs", "ramp", "slippery"],
                        default="flat")
    parser.add_argument("--robot", type=str,
                        choices=["go2"], default="go2")
    parser.add_argument("--nav", type=str,
                        choices=["random", "vel"], default="random",
                        help="random: each robot gets its own random goal; "
                             "vel: keyboard command broadcast to all robots.")
    parser.add_argument("--n-env", type=int, default=8,
                        help="Number of parallel environments.")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
        robot=args.robot,
        nav=args.nav,
        n_env=args.n_env,
    )
