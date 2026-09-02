"""Multi-environment 4-contact balance simulator for the Go2 quadruped.

N robots run in parallel on the GPU via JAX/MJX.  Each robot has its own:
  - MPC warm-start (``batch_mpc_data``)
  - Desired pose sampled from ``DesiredPoseSampler`` (random height + orientation)
  - Foot anchor locked to its spawn foot positions
  - Per-robot desired-orientation frame rendered as RGB arrows in the viewer

All robots are shown simultaneously as ghost overlays on a single CPU MjModel,
arranged in a square grid.

Usage::

    python -m mpx.simulators.quadruped.quad_4balance_multiEnv --n-env 8
    python -m mpx.simulators.quadruped.quad_4balance_multiEnv --n-env 16 --scene rough
    python -m mpx.simulators.quadruped.quad_4balance_multiEnv --headless --n-env 64
"""

import argparse
import math
import os
import sys
import time
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

from mpx.config.robot_config.config_go2 import go2_config, Go2Mode, BalanceStance
from mpx.utils.quad_utils_balance.mpc_wrapper_4balance import MPCWrapper

from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation

from mpx.utils.quad_utils_balance.desired_pose_sampler import (
    DesiredPoseSampler, desired_pose_config,
)

import mpx.utils.simulation_utils.sim_utils as sim_utils
from mpx.utils.math_utils.quad_math import (
    yaw_from_quat, _quat_to_axes, quat_normalize_wxyz, quat_mul_wxyz,
)


# ─────────────────────────────────────────────────────────────────────────────
# Crash detection (batch)
# ─────────────────────────────────────────────────────────────────────────────

def _is_crashed_batch(qpos_batch: np.ndarray, height_threshold: float, tilt_rad: float) -> np.ndarray:
    """Return bool array of shape (N,) — True where robot i has crashed."""
    crashed = np.zeros(qpos_batch.shape[0], dtype=bool)
    for i in range(qpos_batch.shape[0]):
        if qpos_batch[i, 2] < height_threshold:
            crashed[i] = True
            continue
        w, x, y, z = qpos_batch[i, 3], qpos_batch[i, 4], qpos_batch[i, 5], qpos_batch[i, 6]
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
    n_env: int = 8,
):
    # ── Config & CPU model ──────────────────────────────────────────────────
    config = go2_config(Go2Mode.BALANCE, balance_stance=BalanceStance.FOUR)

    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../../data/{robot}/scene_{scene}.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1.0 / sim_frequency

    cpu_contact_ids = sim_utils.geom_ids(model, config.contact_frame)

    # ── MJX model & batched data ─────────────────────────────────────────────
    mjx_model = mjx.put_model(model)

    qpos0_single = np.concatenate([
        np.asarray(config.p0),
        np.asarray(config.quat0),
        np.asarray(config.q0),
    ]).astype(np.float64)

    data.qpos = qpos0_single
    mujoco.mj_forward(model, data)
    mjx_data_template = mjx.put_data(model, data)

    qpos0_batch = jnp.tile(jnp.asarray(qpos0_single), (n_env, 1))
    batch_data  = jax.vmap(lambda qp: mjx_data_template.replace(qpos=qp))(qpos0_batch)

    # ── MJX contact IDs ─────────────────────────────────────────────────────
    mjx_contact_ids = [
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in config.contact_frame
    ]

    # ── Batched MPC ──────────────────────────────────────────────────────────
    mpc = MPCWrapper(config, limited_memory=True)

    batch_mpc_data = jax.vmap(lambda _: mpc.make_data())(jnp.arange(n_env))

    fixed_contact = jnp.asarray(config.balance_fixed_contact_mask, dtype=jnp.float32)

    def _run_one(mpc_data_i, x0_i, command_i,
                 base_quat_ref_i, use_base_quat_ref_i,
                 foot_ref_anchor_i, use_foot_ref_anchor_i):
        return mpc.run(
            mpc_data_i, x0_i, command_i,
            fixed_contact,
            base_quat_ref_i, use_base_quat_ref_i,
            foot_ref_anchor_i, use_foot_ref_anchor_i,
        )

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
    batched_step = jax.jit(jax.vmap(lambda d, a: mjx.step(mjx_model, d.replace(ctrl=a))))

    # ── Per-robot desired poses ───────────────────────────────────────────────
    desired_pose_sampler = DesiredPoseSampler.from_config(config.robot_height, desired_pose_config)

    # desired_heights:    (N,)   float
    # desired_quats:      (N, 4) float32
    # foot_anchors:       (N, 12) float32
    desired_heights = np.full(n_env, config.robot_height, dtype=np.float64)
    desired_quats   = np.tile(np.asarray(config.quat0, dtype=np.float32), (n_env, 1))
    foot_anchors    = np.zeros((n_env, 3 * config.n_contact), dtype=np.float32)

    def _sample_pose_for(i: int, quat_at_spawn: np.ndarray) -> None:
        """Update desired_heights, desired_quats, foot_anchors for robot i."""
        h, delta_quat = desired_pose_sampler.sample()
        desired_heights[i] = h
        desired_quats[i]   = quat_normalize_wxyz(quat_mul_wxyz(quat_at_spawn, delta_quat)).astype(np.float32)

    # ── Base-force perturbations (one per robot) ──────────────────────────────
    perturbers = [
        RandomBaseForcePerturbation.from_config(
            sim_dt=1.0 / sim_frequency,
            cfg=ext_base_force_config,
        )
        for _ in range(n_env)
    ]

    # ── Viewer grid offsets ───────────────────────────────────────────────────
    robots_per_row = math.ceil(math.sqrt(n_env))
    grid = np.arange(robots_per_row ** 2)
    offset_xy = np.stack([
        (grid % robots_per_row)[:n_env] * 1.5,
        (grid // robots_per_row)[:n_env] * 1.5,
    ], axis=1).astype(np.float64)

    # ── Crash detection thresholds ────────────────────────────────────────────
    CRASH_HEIGHT = config.robot_height * 0.5
    CRASH_TILT   = np.deg2rad(60.0)

    # ── Helper: reset one robot back to spawn ────────────────────────────────
    def _reset_robot(i: int, qpos_np: np.ndarray) -> None:
        """Reset robot i in-place in qpos_np and refresh its desired pose + anchor."""
        qpos_np[i] = qpos0_single
        spawn_quat = qpos_np[i, 3:7].astype(np.float64)
        _sample_pose_for(i, spawn_quat)
        # foot anchor = nominal foot positions at the reset pose (flat ground)
        foot_anchors[i] = np.asarray(config.p_legs0, dtype=np.float32)
        perturbers[i].reset()

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"[4balance_multiEnv] Warming up {n_env} environments …", flush=True)

    qpos_np = np.array(batch_data.qpos)
    for i in range(n_env):
        _reset_robot(i, qpos_np)

    batch_data = jax.vmap(
        lambda qp: mjx_data_template.replace(qpos=qp)
    )(jnp.asarray(qpos_np))

    batch_x0, batch_foot = build_x0_batch(batch_data)
    batch_mpc_data = batched_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, batch_foot)

    # Foot anchors from first forward pass
    foot_anchors = np.array(batch_foot, dtype=np.float32)

    warm_cmd = jnp.zeros((n_env, 7)).at[:, 6].set(jnp.asarray(desired_heights, dtype=jnp.float32))
    batch_mpc_data, tau_batch = batched_solve(
        batch_mpc_data, batch_x0, warm_cmd,
        jnp.asarray(desired_quats),
        jnp.ones(n_env, dtype=bool),
        jnp.asarray(foot_anchors),
        jnp.ones(n_env, dtype=bool),
    )
    tau_batch.block_until_ready()

    # Re-reset after warm-up.
    batch_data = jax.vmap(
        lambda qp: mjx_data_template.replace(qpos=qp)
    )(jnp.asarray(qpos_np))
    _, batch_foot = build_x0_batch(batch_data)
    batch_mpc_data = batched_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, batch_foot)
    tau_batch = jnp.zeros((n_env, config.n_joints))

    period = int(sim_frequency / config.mpc_frequency)
    counter = 0
    print(
        f"[4balance_multiEnv] {n_env} robots | period {period} steps | scene={scene}",
        flush=True,
    )

    # ── Headless loop ─────────────────────────────────────────────────────────
    if headless:
        for _ in range(steps):
            if counter % period == 0:
                batch_x0, _ = build_x0_batch(batch_data)
                batch_cmd = jnp.zeros((n_env, 7)).at[:, 6].set(
                    jnp.asarray(desired_heights, dtype=jnp.float32)
                )
                start = timer()
                batch_mpc_data, tau_batch = batched_solve(
                    batch_mpc_data, batch_x0, batch_cmd,
                    jnp.asarray(desired_quats),
                    jnp.ones(n_env, dtype=bool),
                    jnp.asarray(foot_anchors),
                    jnp.ones(n_env, dtype=bool),
                )
                tau_batch.block_until_ready()
                print(f"  step {counter:5d}  MPC {1e3*(timer()-start):.1f} ms", flush=True)

            batch_data = batched_step(batch_data, tau_batch)

            crashed = _is_crashed_batch(np.array(batch_data.qpos), CRASH_HEIGHT, CRASH_TILT)
            if crashed.any():
                qpos_np = np.array(batch_data.qpos)
                for i in np.where(crashed)[0]:
                    _reset_robot(i, qpos_np)
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
                foot_anchors = np.array(bf, dtype=np.float32)

            counter += 1
        return

    # ── Viewer loop ───────────────────────────────────────────────────────────
    scratch_data = mujoco.MjData(model)
    ghost_geoms  = [None] * n_env

    # Desired-orientation frame arrows per robot: [x, y, z] geom IDs each.
    _frame_geoms = [[-1, -1, -1] for _ in range(n_env)]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        # Pre-build ghost geom caches.
        qpos_np = np.array(batch_data.qpos)
        for i in range(n_env):
            qp = qpos_np[i].copy()
            qp[0] += offset_xy[i, 0]
            qp[1] += offset_xy[i, 1]
            scratch_data.qpos[:len(qpos0_single)] = qp
            mujoco.mj_forward(model, scratch_data)
            ghost_geoms[i] = sim_utils.render_ghost_robot(
                viewer, model, scratch_data, alpha=0.9
            )
        viewer.sync()

        while viewer.is_running():
            tic = timer()

            # ── MPC solve ────────────────────────────────────────────────────
            if counter % period == 0:
                batch_x0, _ = build_x0_batch(batch_data)
                batch_cmd = jnp.zeros((n_env, 7)).at[:, 6].set(
                    jnp.asarray(desired_heights, dtype=jnp.float32)
                )
                start = timer()
                batch_mpc_data, tau_batch = batched_solve(
                    batch_mpc_data, batch_x0, batch_cmd,
                    jnp.asarray(desired_quats),
                    jnp.ones(n_env, dtype=bool),
                    jnp.asarray(foot_anchors),
                    jnp.ones(n_env, dtype=bool),
                )
                tau_batch.block_until_ready()
                print(f"  step {counter:5d}  batched MPC {1e3*(timer()-start):.1f} ms", flush=True)

            # ── Physics step ─────────────────────────────────────────────────
            batch_data = batched_step(batch_data, tau_batch)

            # ── Crash reset ──────────────────────────────────────────────────
            qpos_np = np.array(batch_data.qpos)
            crashed = _is_crashed_batch(qpos_np, CRASH_HEIGHT, CRASH_TILT)
            if crashed.any():
                for i in np.where(crashed)[0]:
                    _reset_robot(i, qpos_np)
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
                foot_anchors = np.array(bf, dtype=np.float32)
                qpos_np = np.array(batch_data.qpos)

            # ── Render ghost robots ───────────────────────────────────────────
            for i in range(n_env):
                qp_vis = qpos_np[i].copy()
                qp_vis[0] += offset_xy[i, 0]
                qp_vis[1] += offset_xy[i, 1]
                scratch_data.qpos[:len(qpos0_single)] = qp_vis
                mujoco.mj_forward(model, scratch_data)
                ghost_geoms[i] = sim_utils.render_ghost_robot(
                    viewer, model, scratch_data, alpha=0.9, ghost_geoms=ghost_geoms[i]
                )

            # ── Render desired-orientation frames (RGB arrows per robot) ──────
            for i in range(n_env):
                frame_pos = np.array([
                    qpos_np[i, 0] + offset_xy[i, 0],
                    qpos_np[i, 1] + offset_xy[i, 1],
                    float(desired_heights[i]) + 0.3,
                ])
                dx, dy, dz = _quat_to_axes(desired_quats[i].astype(np.float64))
                _frame_geoms[i][0] = sim_utils.render_vector(
                    viewer, dx, frame_pos, scale=0.25,
                    color=np.array([1.0, 0.15, 0.15, 0.85]),
                    geom_id=_frame_geoms[i][0],
                )
                _frame_geoms[i][1] = sim_utils.render_vector(
                    viewer, dy, frame_pos, scale=0.25,
                    color=np.array([0.15, 0.9, 0.15, 0.85]),
                    geom_id=_frame_geoms[i][1],
                )
                _frame_geoms[i][2] = sim_utils.render_vector(
                    viewer, dz, frame_pos, scale=0.25,
                    color=np.array([0.15, 0.15, 1.0, 0.85]),
                    geom_id=_frame_geoms[i][2],
                )

            counter += 1
            toc = timer()
            if toc - tic < model.opt.timestep:
                time.sleep(model.opt.timestep - (toc - tic))
            viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-environment Go2 4-contact balance."
    )
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of simulation steps (headless only).")
    parser.add_argument("--scene", type=str,
                        choices=["flat", "rough", "perlin", "stairs", "ramp", "slippery"],
                        default="flat")
    parser.add_argument("--robot", type=str, choices=["go2"], default="go2")
    parser.add_argument("--n-env", type=int, default=8,
                        help="Number of parallel environments.")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
        robot=args.robot,
        n_env=args.n_env,
    )
