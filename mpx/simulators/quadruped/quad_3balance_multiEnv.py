"""Multi-environment 3-contact (tripod) balance simulator for the Go2 quadruped.

N robots run in parallel on the GPU via JAX/MJX.  Each robot has its own:
  - MPC warm-start (``batch_mpc_data``)
  - Desired body pose from ``DesiredPoseSampler`` (disabled by default — flat upright)
  - Per-robot tripod foot anchor sampled via ``tripod_foot_reference_world``
    (swing foot pushed forward by SWING_INIT_OFFSET at each reset)
  - Swing goal sphere rendered per robot in the viewer

All robots are displayed simultaneously as ghost overlays on a single CPU
MjModel, arranged in a square grid.

Usage::

    python -m mpx.simulators.quadruped.quad_3balance_multiEnv --n-env 8
    python -m mpx.simulators.quadruped.quad_3balance_multiEnv --n-env 16 --scene flat
    python -m mpx.simulators.quadruped.quad_3balance_multiEnv --headless --n-env 64
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
from mpx.utils.quad_utils_balance.mpc_wrapper_3balance import MPCWrapper

from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation

from mpx.utils.quad_utils_balance.desired_pose_sampler import (
    DesiredPoseSampler, DesiredPoseConfig,
)
# 3-balance: desired pose sampler disabled by default (flat upright)
desired_pose_config = DesiredPoseConfig(enabled=False)

from mpx.utils.quad_utils_balance.foot_reference import (
    FootReferenceManager,
    RandomSwingFootSampler,
    swing_foot_anchor_from_target,
    foot_target_foot_local_to_world,
)
from mpx.config.sim_config.config_foot_ref_config import foot_ref_config, random_swing_foot_config
import glfw

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
    # ── Config ──────────────────────────────────────────────────────────────
    config = go2_config(Go2Mode.BALANCE, balance_stance=BalanceStance.TRIPOD_SWING_FL)

    # Swing leg = the entry in balance_fixed_contact_mask that is 0.
    swing_leg_idx = int(np.where(np.array(config.balance_fixed_contact_mask) < 0.5)[0][0])
    SWING_INIT_OFFSET = np.array([0.25, -0.1, -0.3])   # forward+up offset in body frame

    print(
        f"\n[3balance_multiEnv] {n_env} robots | "
        f"swing leg: {swing_leg_idx} ({['FL','FR','RL','RR'][swing_leg_idx]}) | "
        f"scene: {scene}\n"
        "  R  — randomise all swing foot targets NOW (within bounds)\n"
        "  N  — toggle random-on-respawn mode ON/OFF\n",
        flush=True,
    )

    # ── CPU model ────────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../../data/{robot}/scene_{scene}.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1.0 / sim_frequency

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
    cpu_contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    scratch_data = mujoco.MjData(model)

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

    # ── Per-robot desired poses and foot anchors ──────────────────────────────
    desired_pose_sampler = DesiredPoseSampler.from_config(config.robot_height, desired_pose_config)
    foot_ref_mgr = FootReferenceManager(foot_ref_config)

    # Random swing-foot sampler — R: resample all robots now, N: toggle auto-resample on respawn.
    # Edit random_swing_foot_config in config_foot_ref_config.py to change bounds / enable at start.
    random_swing_sampler = RandomSwingFootSampler(random_swing_foot_config)
    print(
        f"[random_swing] mode={'ON' if random_swing_sampler.enabled else 'OFF'}  "
        f"bounds: {random_swing_sampler.bounds_summary()}",
        flush=True,
    )

    desired_heights = np.full(n_env, config.robot_height, dtype=np.float64)
    desired_quats   = np.tile(np.asarray(config.quat0, dtype=np.float32), (n_env, 1))
    foot_anchors    = np.zeros((n_env, 3 * config.n_contact), dtype=np.float32)
    arrival_cooldown_steps = np.zeros(n_env, dtype=np.int32)
    arrival_hold_steps = np.zeros(n_env, dtype=np.int32)

    def _check_arrival_resample_at_mpc(batch_data_mjx) -> None:
        """Resample swing targets (runs at MPC rate, before batched_solve)."""
        nonlocal foot_anchors, arrival_cooldown_steps, arrival_hold_steps
        if not random_swing_sampler.resample_on_arrival:
            return
        sim_dt = model.opt.timestep
        qpos_cur = np.array(batch_data_mjx.qpos)
        for i in range(n_env):
            foot_flat = np.concatenate([
                np.asarray(batch_data_mjx.geom_xpos[mjx_contact_ids[k]][i], dtype=np.float64)
                for k in range(config.n_contact)
            ])
            measured_swing = foot_flat[3 * swing_leg_idx : 3 * swing_leg_idx + 3]
            updated, arrival_cooldown_steps[i], arrival_hold_steps[i], did = (
                random_swing_sampler.try_resample_on_arrival(
                    foot_anchors[i],
                    swing_leg_idx,
                    measured_swing,
                    qpos_cur[i, :3],
                    qpos_cur[i, 3:7],
                    sim_dt=sim_dt,
                    cooldown_steps=int(arrival_cooldown_steps[i]),
                    hold_steps=int(arrival_hold_steps[i]),
                )
            )
            if did:
                foot_anchors[i] = updated.astype(np.float32)
                sw = foot_anchors[i, 3 * swing_leg_idx : 3 * swing_leg_idx + 3]
                print(
                    f"  [random_swing] robot {i} arrival → "
                    f"[{sw[0]:.3f}, {sw[1]:.3f}, {sw[2]:.3f}]",
                    flush=True,
                )

    def _sample_foot_anchor(
        i: int, qpos_i: np.ndarray, measured_swing_world: np.ndarray,
    ) -> np.ndarray:
        """Compute world-frame tripod foot anchor for robot i (at its current pose)."""
        rng_key = jax.random.PRNGKey(int(time.time() * 1000 + i) & 0x7FFFFFFF)
        anchor = np.asarray(
            foot_ref_mgr.tripod_foot_reference_world(
                key=rng_key,
                p=jnp.asarray(qpos_i[:3]),
                quat=jnp.asarray(qpos_i[3:7]),
                foot0=jnp.asarray(config.p_legs0),
                n_contact=config.n_contact,
                sigma=np.array([0.04, 0.04, 0.0]),
            ),
            dtype=np.float64,
        )
        origin = np.asarray(measured_swing_world, dtype=np.float64).reshape(3)
        if random_swing_sampler.resample_on_respawn:
            new_swing = random_swing_sampler.sample_swing_world(qpos_i[:3], qpos_i[3:7])
        else:
            new_swing = foot_target_foot_local_to_world(origin, qpos_i[3:7], SWING_INIT_OFFSET)
        return swing_foot_anchor_from_target(anchor, swing_leg_idx, new_swing).astype(np.float32)

    def _reset_robot(i: int, qpos_np: np.ndarray) -> None:
        """Reset robot i in-place: pose back to nominal, resample desired pose + foot anchor."""
        qpos_np[i] = qpos0_single
        spawn_quat = qpos_np[i, 3:7].astype(np.float64)

        h, delta_quat = desired_pose_sampler.sample()
        desired_heights[i] = h
        desired_quats[i]   = quat_normalize_wxyz(
            quat_mul_wxyz(spawn_quat, delta_quat)
        ).astype(np.float32)
        scratch_data.qpos[:len(qpos0_single)] = qpos_np[i]
        mujoco.mj_forward(model, scratch_data)
        measured_swing = sim_utils.geom_positions(scratch_data, cpu_contact_ids)[
            3 * swing_leg_idx : 3 * swing_leg_idx + 3
        ]
        foot_anchors[i] = _sample_foot_anchor(i, qpos_np[i], measured_swing)
        arrival_cooldown_steps[i] = 0
        arrival_hold_steps[i] = 0
        perturbers[i].reset()

    # ── Per-robot base-force perturbations ────────────────────────────────────
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

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"[3balance_multiEnv] Warming up {n_env} environments …", flush=True)

    qpos_np = np.array(batch_data.qpos)
    for i in range(n_env):
        _reset_robot(i, qpos_np)

    batch_data = jax.vmap(
        lambda qp: mjx_data_template.replace(qpos=qp)
    )(jnp.asarray(qpos_np))

    batch_x0, batch_foot = build_x0_batch(batch_data)
    batch_mpc_data = batched_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, batch_foot)

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
        f"[3balance_multiEnv] {n_env} robots | period {period} steps | scene={scene}",
        flush=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Shared crash-reset logic (used in both headless and viewer loops)
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_crashes(qpos_np: np.ndarray):
        """Reset crashed robots and rebuild batch_data / batch_mpc_data."""
        nonlocal batch_data, batch_mpc_data, foot_anchors
        crashed = _is_crashed_batch(qpos_np, CRASH_HEIGHT, CRASH_TILT)
        if not crashed.any():
            return qpos_np
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
        _, bf = build_x0_batch(batch_data)
        batch_mpc_data = batched_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, bf)
        return np.array(batch_data.qpos)

    # ── Headless loop ─────────────────────────────────────────────────────────
    if headless:
        for _ in range(steps):
            if counter % period == 0:
                _check_arrival_resample_at_mpc(batch_data)
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
            qpos_np = _handle_crashes(np.array(batch_data.qpos))
            counter += 1
        return

    # ── Viewer loop ───────────────────────────────────────────────────────────
    ghost_geoms  = [None] * n_env

    # Swing goal spheres and random-bounds boxes per robot.
    _swing_goal_ids = [-1] * n_env
    _swing_bounds_markers: list = [None] * n_env

    def key_callback(key: int) -> None:
        nonlocal foot_anchors
        if key == glfw.KEY_R:
            # Resample all robots' swing foot targets in base-frame bounds.
            qpos_cur = np.array(batch_data.qpos)
            for i in range(n_env):
                new_swing = random_swing_sampler.sample_swing_world(
                    qpos_cur[i, :3], qpos_cur[i, 3:7]
                )
                foot_anchors[i] = swing_foot_anchor_from_target(
                    foot_anchors[i], swing_leg_idx, new_swing
                ).astype(np.float32)
                arrival_cooldown_steps[i] = 0
                arrival_hold_steps[i] = 0
            print(f"[random_swing] resampled {n_env} swing targets", flush=True)
        elif key == glfw.KEY_N:
            state = random_swing_sampler.toggle()
            print(
                f"[random_swing] random-on-respawn {'ON ' if state else 'OFF'}  "
                f"({random_swing_sampler.bounds_summary()})",
                flush=True,
            )

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
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
                _check_arrival_resample_at_mpc(batch_data)
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

            qpos_np = _handle_crashes(np.array(batch_data.qpos))

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

            # ── Render per-robot swing goal spheres & sampling bounds boxes ───
            qpos_vis = np.array(batch_data.qpos)
            for i in range(n_env):
                base_vis = qpos_vis[i, :3].copy()
                base_vis[0] += offset_xy[i, 0]
                base_vis[1] += offset_xy[i, 1]

                if _swing_bounds_markers[i] is None:
                    _swing_bounds_markers[i] = random_swing_sampler.attach_bounds_box_marker(
                        viewer
                    )
                random_swing_sampler.update_bounds_box_marker(
                    _swing_bounds_markers[i],
                    viewer,
                    base_vis,
                    qpos_vis[i, 3:7],
                    sync=False,
                )

                swing_world = foot_anchors[i, 3*swing_leg_idx : 3*swing_leg_idx + 3].astype(np.float64)
                goal_pos = np.array([
                    swing_world[0] + offset_xy[i, 0],
                    swing_world[1] + offset_xy[i, 1],
                    swing_world[2],
                ])
                goal_diameter = 2.0 * foot_ref_config.swing_goal_radius
                _swing_goal_ids[i] = sim_utils.render_sphere(
                    viewer,
                    position=goal_pos,
                    diameter=goal_diameter,
                    color=np.array([1.0, 0.5, 0.0, 0.8]),   # orange
                    geom_id=_swing_goal_ids[i],
                )

            counter += 1
            toc = timer()
            if toc - tic < model.opt.timestep:
                time.sleep(model.opt.timestep - (toc - tic))
            viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-environment Go2 3-contact (tripod) balance."
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
