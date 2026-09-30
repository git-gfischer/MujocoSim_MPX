"""Multi-environment 4-contact balance simulator for the Go2 quadruped.

N robots run in parallel on the GPU via JAX/MJX.  Each robot has its own:
  - MPC warm-start (``batch_mpc_data``)
  - Desired pose sampled from ``DesiredPoseSampler`` (random height + orientation)
  - Foot anchor locked to its spawn foot positions
  - Per-robot desired-orientation frame rendered as RGB arrows in the viewer

Robot 0 is the live viewer body, so MuJoCo contact visualization matches
that mesh. The other robots are ghost overlays on a square grid.

``--collect`` switches to N CPU MuJoCo worlds (dataset ``sim_hz``), one batched
MPC solve, and one run directory — the same collector layout as
``quad_locomotion_multiEnv``. Without ``--collect`` the MJX ghost demo is unchanged.

Usage::

    python -m mpx.simulators.quadruped.quad_4balance_multiEnv --n-env 8
    python -m mpx.simulators.quadruped.quad_4balance_multiEnv --n-env 16 --scene rough
    python -m mpx.simulators.quadruped.quad_4balance_multiEnv --headless --n-env 64
    python -m mpx.simulators.quadruped.quad_4balance_multiEnv --collect --n-env 8 --headless
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
from mpx.config.sim_config.config_base_weight import base_weight_config
from mpx.utils.simulation_utils.base_weight import BaseWeightForce
from mpx.config.sim_config.config_reset_randomization import balance_reset_randomization_config
from mpx.utils.simulation_utils.reset_randomizer import ResetRandomizer, ResetTargets
from mpx.config.sim_config.config_quad_spawn import spawn_config
from mpx.utils.spawner.spawner import RobotMapSpawner

from mpx.utils.quad_utils_balance.desired_pose_sampler import (
    DesiredPoseSampler, desired_pose_config,
)

from mpx.utils.dataset_collection.episode_recorder import (
    ControlSample,
    read_episode_conditions,
    setup_multi_env_collection,
)
from mpx.utils.dataset_collection.dataset_bucket_system import GaitType
from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config

import mpx.utils.simulation_utils.sim_utils as sim_utils
from mpx.utils.math_utils.quad_math import (
    yaw_from_quat, _quat_to_axes, quat_normalize_wxyz, quat_mul_wxyz,
)


def _robot_config(robot: str):
    if robot == "go2":
        return go2_config(Go2Mode.BALANCE, balance_stance=BalanceStance.FOUR)
    if robot == "b2":
        from mpx.config.robot_config.config_b2 import b2_config, B2Mode
        return b2_config(B2Mode.BALANCE, balance_stance=BalanceStance.FOUR)
    raise ValueError(f"Unsupported robot: {robot}")


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


def _tree_set_index(tree, index, value):
    """Replace row ``index`` of a batched pytree with a single-env pytree."""
    return jax.tree.map(lambda batch, leaf: batch.at[index].set(leaf), tree, value)


def _env_scalar(value, i: int) -> float:
    """Batched ``array[i]``, or a shared mjx PyTreeNode metadata scalar."""
    arr = np.asarray(value)
    return float(arr[i] if arr.ndim else arr)


def _resolve_headless_steps(
    *,
    collect: bool,
    steps: int | None,
    episode_duration_s: float,
    sim_hz: float,
) -> int:
    """Default live runs stay short; collection must outlive the episode gate."""
    if steps is not None:
        return int(steps)
    if collect:
        return int(round(float(episode_duration_s) * float(sim_hz)))
    return 2000


def _scene_xml_path(robot: str, scene: str) -> str:
    return os.path.abspath(os.path.join(dir_path, "..", "..", "data", robot, f"scene_{scene}.xml"))


# Shorter than SpawnConfig.settle_duration_s (0.8 s). Four-contact balance is
# already on its feet; the shared budget is the rough-terrain locomotion cap.
# The loop still exits early once joint and vertical speed are under tolerance.
_BALANCE_SETTLE_S = 0.4
# Consecutive time inside the desired pose before that environment respawns.
_POSE_HOLD_S = 1.0
# Wide enough that a robot which has settled near the pose counts as arrived.
# 3 cm / 8 deg left them outside the window, so the hold timer never finished.
_POSE_HEIGHT_TOL = 0.06       # [m]
_POSE_ORIENT_TOL_DEG = 15.0   # [deg]


def _settle_world(model, data, config, sim_frequency: float) -> None:
    if not spawn_config.settle_after_spawn:
        return
    n_steps = int(round(_BALANCE_SETTLE_S * sim_frequency))
    if n_steps <= 0:
        return
    n_j = config.n_joints
    q_hold = np.asarray(config.q0, dtype=np.float64).reshape(n_j)
    kp = float(spawn_config.settle_kp)
    kd = float(spawn_config.settle_kd)
    lo, hi = float(config.min_torque), float(config.max_torque)
    tol = float(spawn_config.settle_qvel_tol)
    data.qfrc_applied[:] = 0.0
    for _ in range(n_steps):
        q = np.asarray(data.qpos[7 : 7 + n_j], dtype=np.float64)
        dq = np.asarray(data.qvel[6 : 6 + n_j], dtype=np.float64)
        data.ctrl = np.clip(kp * (q_hold - q) - kd * dq, lo, hi)
        mujoco.mj_step(model, data)
        if float(np.max(np.abs(dq))) < tol and abs(float(data.qvel[2])) < tol:
            break
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def _holds_desired_pose(qpos, desired_height: float, desired_quat) -> bool:
    """True while the base is inside the goal-pose window."""
    if abs(float(qpos[2]) - float(desired_height)) > _POSE_HEIGHT_TOL:
        return False
    quat_now = quat_normalize_wxyz(np.asarray(qpos[3:7], dtype=np.float64))
    quat_ref = quat_normalize_wxyz(np.asarray(desired_quat, dtype=np.float64))
    cos_half = min(1.0, abs(float(np.dot(quat_now, quat_ref))))
    orient_err = 2.0 * np.arccos(cos_half)
    return orient_err <= np.deg2rad(_POSE_ORIENT_TOL_DEG)


def _advance_pose_hold(hold_s: float, holding: bool, dt: float) -> tuple[float, bool]:
    """Accumulate time inside the goal pose. Leaving it clears the timer."""
    hold_s = hold_s + float(dt) if holding else 0.0
    return hold_s, hold_s >= _POSE_HOLD_S


def _main_collect(
    *,
    headless: bool,
    steps: int,
    scene: str,
    robot: str,
    n_env: int,
    collect_out,
    episode_duration_s,
):
    """N CPU MuJoCo worlds, batched 4-contact MPC, one dataset run directory."""
    config = _robot_config(robot)
    xml_path = _scene_xml_path(robot, scene)
    sim_frequency = float(dataset_collection_config.rates.sim_hz)
    models = []
    datas = []
    contact_ids = []
    for _ in range(n_env):
        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.timestep = 1.0 / sim_frequency
        data = mujoco.MjData(model)
        models.append(model)
        datas.append(data)
        contact_ids.append(sim_utils.geom_ids(model, config.contact_frame))

    mpc = MPCWrapper(config, limited_memory=True)
    fixed_contact = jnp.asarray(config.balance_fixed_contact_mask, dtype=jnp.float32)

    def _run_one(mpc_data_i, x0_i, command_i,
                 base_quat_ref_i, use_base_quat_ref_i,
                 foot_ref_anchor_i, use_foot_ref_anchor_i):
        return mpc.run(
            mpc_data_i, x0_i, command_i, fixed_contact,
            base_quat_ref_i, use_base_quat_ref_i,
            foot_ref_anchor_i, use_foot_ref_anchor_i,
        )

    batched_solve = jax.jit(jax.vmap(_run_one))
    collectors = setup_multi_env_collection(
        True,
        n_env,
        gait_type=GaitType.BALANCE,
        scene=scene,
        sim_hz=sim_frequency,
        robot=robot,
        episode_duration_s=episode_duration_s,
        collect_out=collect_out,
        name_prefix="4balance",
        cfg=dataset_collection_config,
    )

    desired_pose_sampler = DesiredPoseSampler.from_config(config.robot_height, desired_pose_config)
    desired_heights = np.full(n_env, config.robot_height, dtype=np.float64)
    desired_quats = np.tile(np.asarray(config.quat0, dtype=np.float32), (n_env, 1))
    foot_anchors = np.zeros((n_env, 3 * config.n_contact), dtype=np.float32)

    spawners = [
        RobotMapSpawner.from_config(
            cfg=spawn_config,
            foot_geom_names=config.contact_frame,
            check_collisions=True,
            robot_root_body_name=getattr(
                config, "base_body_name", spawn_config.robot_root_body_name
            ),
        )
        for _ in range(n_env)
    ]
    perturbers = [
        RandomBaseForcePerturbation.from_config(
            sim_dt=1.0 / sim_frequency, cfg=ext_base_force_config
        )
        for _ in range(n_env)
    ]
    weights = [BaseWeightForce.from_config(cfg=base_weight_config) for _ in range(n_env)]
    randomizers = [
        ResetRandomizer.from_config(balance_reset_randomization_config) for _ in range(n_env)
    ]

    episode_seeds = [0] * n_env
    tau_np = np.zeros((n_env, config.n_joints), dtype=np.float64)
    crash_height = config.robot_height * 0.5
    crash_tilt = np.deg2rad(60.0)

    def _pack_env(i: int):
        foot = jnp.ravel(jnp.asarray(sim_utils.geom_positions(datas[i], contact_ids[i])))
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(jnp.asarray(datas[i].qpos))
            .at[mpc.qvel_slice].set(jnp.asarray(datas[i].qvel))
            .at[mpc.foot_slice].set(foot)
        )
        return x0, foot

    def _sample_pose(i: int) -> None:
        height, delta_quat = desired_pose_sampler.sample()
        desired_heights[i] = height
        spawn_quat = np.asarray(datas[i].qpos[3:7], dtype=np.float64)
        desired_quats[i] = quat_normalize_wxyz(
            quat_mul_wxyz(spawn_quat, delta_quat)
        ).astype(np.float32)
        foot_anchors[i] = np.asarray(
            sim_utils.geom_positions(datas[i], contact_ids[i]), dtype=np.float32
        ).reshape(-1)

    def _randomize(i: int, mpc_data_i):
        episode_seeds[i] += 1
        sample, mpc_data_i = randomizers[i].sample_and_apply(
            ResetTargets(
                model=models[i],
                foot_geom_ids=contact_ids[i],
                base_weight=weights[i],
                navigator=None,
                mpc_data=mpc_data_i,
            )
        )
        collectors.env[i].set_episode_conditions(
            randomization=sample.to_metadata(),
            seed=episode_seeds[i],
            mode="stand_4leg",
            **read_episode_conditions(models[i], contact_ids[i], weights[i]),
        )
        return mpc_data_i

    pose_hold_s = np.zeros(n_env, dtype=np.float64)

    def _respawn(i: int, batch_mpc, *, crashed: bool = False):
        pose_hold_s[i] = 0.0
        collectors.env[i].on_respawn(crashed=crashed)
        spawners[i].apply_to_data(models[i], datas[i], config.p0, config.quat0, config.q0)
        _settle_world(models[i], datas[i], config, sim_frequency)
        _sample_pose(i)
        x0, foot = _pack_env(i)
        single = mpc.reset(mpc.make_data(), datas[i].qpos.copy(), datas[i].qvel.copy(), foot)
        single = _randomize(i, single)
        perturbers[i].reset()
        tau_np[i] = 0.0
        return _tree_set_index(batch_mpc, i, single), x0, foot

    batch_mpc = jax.vmap(lambda _: mpc.make_data())(jnp.arange(n_env))
    for i in range(n_env):
        batch_mpc, _, _ = _respawn(i, batch_mpc)

    x0_list, cmd_list, quat_list, anchor_list = [], [], [], []
    for i in range(n_env):
        x0, _ = _pack_env(i)
        cmd = np.zeros(7, dtype=np.float64)
        cmd[6] = desired_heights[i]
        x0_list.append(x0)
        cmd_list.append(jnp.asarray(cmd))
        quat_list.append(jnp.asarray(desired_quats[i]))
        anchor_list.append(jnp.asarray(foot_anchors[i]))
    batch_mpc, tau_warm = batched_solve(
        batch_mpc,
        jnp.stack(x0_list),
        jnp.stack(cmd_list),
        jnp.stack(quat_list),
        jnp.ones(n_env, dtype=bool),
        jnp.stack(anchor_list),
        jnp.ones(n_env, dtype=bool),
    )
    tau_warm.block_until_ready()
    for i in range(n_env):
        batch_mpc, _, _ = _respawn(i, batch_mpc)
    collectors.on_ready()

    period = int(sim_frequency / config.mpc_frequency)
    print(
        f"[4balance collect] {n_env} CPU worlds | {period} steps/MPC | "
        f"{sim_frequency:.0f} Hz | scene={scene}",
        flush=True,
    )
    counter = 0

    def step_all():
        nonlocal counter, batch_mpc
        if counter % period == 0:
            x0_list, cmd_list, quat_list, anchor_list = [], [], [], []
            for i in range(n_env):
                x0, _ = _pack_env(i)
                cmd = np.zeros(7, dtype=np.float64)
                cmd[6] = desired_heights[i]
                x0_list.append(x0)
                cmd_list.append(jnp.asarray(cmd))
                quat_list.append(jnp.asarray(desired_quats[i]))
                anchor_list.append(jnp.asarray(foot_anchors[i]))
            start = timer()
            batch_mpc, tau_batch = batched_solve(
                batch_mpc,
                jnp.stack(x0_list),
                jnp.stack(cmd_list),
                jnp.stack(quat_list),
                jnp.ones(n_env, dtype=bool),
                jnp.stack(anchor_list),
                jnp.ones(n_env, dtype=bool),
            )
            tau_batch.block_until_ready()
            tau_np[:] = np.asarray(tau_batch)
            if counter == 0 or counter % (period * 50) == 0:
                print(f"  step {counter:6d}  batched MPC {1e3 * (timer() - start):.1f} ms", flush=True)

        for i in range(n_env):
            datas[i].ctrl = tau_np[i]
            perturbers[i].tick_and_apply(datas[i])
            weights[i].apply(datas[i])
            mujoco.mj_step(models[i], datas[i])

            holding = _holds_desired_pose(
                datas[i].qpos, desired_heights[i], desired_quats[i]
            )
            pose_closed = collectors.env[i].set_recording(holding, reason="pose_lost")
            if pose_closed:
                md_i = jax.tree.map(lambda x: x[i], batch_mpc)
                md_i = _randomize(i, md_i)
                batch_mpc = _tree_set_index(batch_mpc, i, md_i)

            duration_closed = collectors.env[i].after_physics_step(
                models[i],
                datas[i],
                tau_np[i],
                contact_ids[i],
                perturbers[i],
                config.n_joints,
                control=ControlSample(
                    tau_cmd=tau_np[i],
                    leg_phase=np.asarray(batch_mpc.contact_time[i]),
                    duty_factor=_env_scalar(batch_mpc.duty_factor, i),
                ),
            )
            crashed = bool(
                _is_crashed_batch(np.asarray(datas[i].qpos)[None], crash_height, crash_tilt)[0]
            )
            pose_hold_s[i], held = _advance_pose_hold(
                float(pose_hold_s[i]),
                holding and not crashed,
                float(models[i].opt.timestep),
            )
            if duration_closed and not pose_closed and not crashed and not held:
                md_i = jax.tree.map(lambda x: x[i], batch_mpc)
                md_i = _randomize(i, md_i)
                batch_mpc = _tree_set_index(batch_mpc, i, md_i)
            if held:
                collectors.env[i].end_episode(reason="pose_held")
                collectors.env[i].set_recording(False, reason="pose_held")
                batch_mpc, _, _ = _respawn(i, batch_mpc)
                print(
                    f"  [goal] robot {i} held the pose for {_POSE_HOLD_S:.0f}s — respawned",
                    flush=True,
                )
            elif crashed:
                batch_mpc, _, _ = _respawn(i, batch_mpc, crashed=True)
                print(f"  [crash] robot {i} respawned", flush=True)

        counter += 1

    if headless:
        for _ in range(steps):
            step_all()
        collectors.finish("")
        return

    scratch = mujoco.MjData(models[0])
    ghost_geoms = [None] * n_env
    frame_geoms = [[-1, -1, -1] for _ in range(n_env)]
    robots_per_row = math.ceil(math.sqrt(n_env))
    grid = np.arange(robots_per_row ** 2)
    offset_xy = np.stack(
        [
            (grid % robots_per_row)[:n_env] * 1.5,
            (grid // robots_per_row)[:n_env] * 1.5,
        ],
        axis=1,
    )
    with mujoco.viewer.launch_passive(models[0], datas[0]) as viewer:
        viewer.sync()
        while viewer.is_running():
            tic = timer()
            step_all()
            for i in range(n_env):
                qp_vis = np.asarray(datas[i].qpos).copy()
                qp_vis[0] += offset_xy[i, 0]
                qp_vis[1] += offset_xy[i, 1]
                # Env 0 is the live body. Contact viz ('c') reads its contacts.
                if i != 0:
                    scratch.qpos[: qp_vis.size] = qp_vis
                    mujoco.mj_forward(models[0], scratch)
                    ghost_geoms[i] = sim_utils.render_ghost_robot(
                        viewer, models[0], scratch, alpha=0.9, ghost_geoms=ghost_geoms[i]
                    )
                frame_pos = np.array([
                    qp_vis[0], qp_vis[1], float(desired_heights[i]) + 0.3,
                ])
                dx, dy, dz = _quat_to_axes(desired_quats[i].astype(np.float64))
                frame_geoms[i][0] = sim_utils.render_vector(
                    viewer, dx, frame_pos, scale=0.25,
                    color=np.array([1.0, 0.15, 0.15, 0.85]),
                    geom_id=frame_geoms[i][0],
                )
                frame_geoms[i][1] = sim_utils.render_vector(
                    viewer, dy, frame_pos, scale=0.25,
                    color=np.array([0.15, 0.9, 0.15, 0.85]),
                    geom_id=frame_geoms[i][1],
                )
                frame_geoms[i][2] = sim_utils.render_vector(
                    viewer, dz, frame_pos, scale=0.25,
                    color=np.array([0.15, 0.15, 1.0, 0.85]),
                    geom_id=frame_geoms[i][2],
                )
            toc = timer()
            if toc - tic < models[0].opt.timestep:
                time.sleep(models[0].opt.timestep - (toc - tic))
            viewer.sync()
    collectors.finish("")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    headless: bool = False,
    steps: int | None = None,
    scene: str = "flat",
    robot: str = "go2",
    n_env: int = 8,
    collect: bool = False,
    collect_out=None,
    episode_duration_s=None,
):
    duration_s = (
        episode_duration_s
        if episode_duration_s is not None
        else dataset_collection_config.episode.episode_duration_s
    )
    steps = _resolve_headless_steps(
        collect=collect,
        steps=steps,
        episode_duration_s=duration_s,
        sim_hz=dataset_collection_config.rates.sim_hz,
    )
    if collect:
        sim_s = steps / dataset_collection_config.rates.sim_hz
        if sim_s < 30.0:
            print(
                f"[collect] warning: --steps {steps} is only {sim_s:.1f}s of sim "
                f"(want >=30s median episode). Pass a larger --steps or omit it.",
                flush=True,
            )
        return _main_collect(
            headless=headless,
            steps=steps,
            scene=scene,
            robot=robot,
            n_env=n_env,
            collect_out=collect_out,
            episode_duration_s=episode_duration_s,
        )

    # ── Config & CPU model ──────────────────────────────────────────────────
    config = _robot_config(robot)

    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../../data/{robot}/scene_{scene}.xml"
    )
    data = mujoco.MjData(model)
    # 500 Hz by default; see config_dataset_bucket.SimRateConfig. At 200 Hz
    # the 0.01 s foot solref is only two substeps and the contact bounces.
    sim_frequency = float(dataset_collection_config.rates.sim_hz)
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
    pose_hold_s = np.zeros(n_env, dtype=np.float64)
    print(
        f"[4balance_multiEnv] {n_env} robots | period {period} steps | scene={scene} | "
        f"respawn after {_POSE_HOLD_S:.0f}s on the goal pose",
        flush=True,
    )

    def _handle_resets() -> None:
        """Respawn any env that crashed or has held the goal pose for 1 s."""
        nonlocal batch_data, batch_mpc_data, foot_anchors
        qpos_np = np.array(batch_data.qpos)
        dt = float(model.opt.timestep)
        crashed = _is_crashed_batch(qpos_np, CRASH_HEIGHT, CRASH_TILT)
        held = np.zeros(n_env, dtype=bool)
        for i in range(n_env):
            pose_hold_s[i], held[i] = _advance_pose_hold(
                float(pose_hold_s[i]),
                (not crashed[i]) and _holds_desired_pose(
                    qpos_np[i], desired_heights[i], desired_quats[i]
                ),
                dt,
            )
        reset = crashed | held
        if not reset.any():
            return
        for i in np.where(reset)[0]:
            _reset_robot(i, qpos_np)
            pose_hold_s[i] = 0.0
            label = "crash" if crashed[i] else "goal"
            print(f"  [{label}] robot {i} respawned", flush=True)
        batch_data = jax.vmap(
            lambda qp: mjx_data_template.replace(
                qpos=qp,
                qvel=jnp.zeros(6 + config.n_joints),
                ctrl=jnp.zeros(config.n_joints),
            )
        )(jnp.asarray(qpos_np))
        _, bf = build_x0_batch(batch_data)
        batch_mpc_data = batched_reset(
            batch_mpc_data, batch_data.qpos, batch_data.qvel, bf
        )
        foot_anchors = np.array(bf, dtype=np.float32)

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
            _handle_resets()

            counter += 1
        return

    # ── Viewer loop ───────────────────────────────────────────────────────────
    scratch_data = mujoco.MjData(model)
    ghost_geoms  = [None] * n_env

    # Desired-orientation frame arrows per robot: [x, y, z] geom IDs each.
    _frame_geoms = [[-1, -1, -1] for _ in range(n_env)]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        # Env 0 is the live viewer body. Cache ghosts for the other envs.
        qpos_np = np.array(batch_data.qpos)
        for i in range(1, n_env):
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
            _handle_resets()
            qpos_np = np.array(batch_data.qpos)

            # Env 0 is the body the viewer uses for contact visualization.
            data.qpos[:] = qpos_np[0]
            data.qvel[:] = np.asarray(batch_data.qvel[0])
            mujoco.mj_forward(model, data)

            for i in range(1, n_env):
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
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Simulation steps (headless). Default 2000 for live runs; "
            "for --collect, one episode_duration at sim rate "
            f"(currently {int(dataset_collection_config.episode.episode_duration_s * dataset_collection_config.rates.sim_hz)})."
        ),
    )
    parser.add_argument("--scene", type=str,
                        choices=["flat", "rough", "perlin", "stairs", "ramp", "slippery"],
                        default="flat")
    parser.add_argument("--robot", type=str, choices=["go2", "b2"], default="go2")
    parser.add_argument("--n-env", type=int, default=8,
                        help="Number of parallel environments.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect v4 episodes from N CPU worlds (batched MPC) into one run directory.",
    )
    parser.add_argument(
        "--collect-out",
        type=str,
        default=None,
        help="Run directory name for --collect.",
    )
    parser.add_argument(
        "--episode-duration",
        type=float,
        default=None,
        help="Episode length [s] for --collect.",
    )
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
        robot=args.robot,
        n_env=args.n_env,
        collect=args.collect,
        collect_out=args.collect_out,
        episode_duration_s=args.episode_duration,
    )
