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
    python -m mpx.simulators.quadruped.quad_locomotion_multiEnv --collect --n-env 8 --nav random --headless
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

from mpx.utils.quad_utils_locomotion.mpc_wrapper_inverse import make_locomotion_mpc

from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation
from mpx.config.sim_config.config_base_weight import base_weight_config
from mpx.utils.simulation_utils.base_weight import BaseWeightForce
from mpx.config.sim_config.config_reset_randomization import loco_reset_randomization_config
from mpx.utils.simulation_utils.reset_randomizer import ResetRandomizer, ResetTargets
from mpx.config.sim_config.config_quad_spawn import spawn_config
from mpx.utils.spawner.spawner import RobotMapSpawner

from mpx.utils.simulation_utils.console import KeyboardVelocityCommand
import mpx.utils.simulation_utils.sim_utils as sim_utils

from mpx.navigation.pointNav import PointNavigator
from mpx.estimators.quad_contact_estimation import estimate_contacts
from mpx.utils.simulation_utils.velocity_command import (
    VelocityCommandSampler,
    command_from_mpc_input,
)
from mpx.utils.dataset_collection.episode_recorder import (
    ControlSample,
    read_episode_conditions,
    setup_multi_env_collection,
)
from mpx.utils.dataset_collection.dataset_bucket_system import gait_from_phase_offsets
from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _robot_config(robot: str, mpc_model: str = "whole_body", gait=None):
    if robot == "go2":
        from mpx.config.robot_config.config_go2 import go2_config, Go2Mode
        return go2_config(Go2Mode.LOCOMOTION, gait=gait, mpc_model=mpc_model)
    if mpc_model not in (None, "whole_body"):
        raise ValueError(
            f"--mpc-model {mpc_model!r} is only supported for the go2 "
            f"(got robot {robot!r})."
        )
    if robot == "b2":
        from mpx.config.robot_config.config_b2 import b2_config, B2Mode
        return b2_config(B2Mode.LOCOMOTION, gait=gait)
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


def _tree_set_index(tree, index, value):
    """Replace row ``index`` of a batched pytree with a single-env pytree."""
    return jax.tree.map(lambda batch, leaf: batch.at[index].set(leaf), tree, value)


def _env_scalar(value, i: int) -> float:
    """Batched ``array[i]``, or a shared mjx PyTreeNode metadata scalar."""
    arr = np.asarray(value)
    return float(arr[i] if arr.ndim else arr)


def _collect_command_source(nav: str, segmented_commands: bool) -> tuple[bool, bool]:
    """``--nav random`` always follows goals; segments fill in when there is no nav."""
    use_navigation = nav in ("random", "pointuser")
    use_command_sampler = bool(segmented_commands) and not use_navigation
    return use_navigation, use_command_sampler


def _tick_navigators(navigators, qpos_batch, viewer=None) -> None:
    """Resample random goals on arrival. Viewer is optional (headless-safe)."""
    qpos_np = np.asarray(qpos_batch)
    for i, nav_i in enumerate(navigators):
        nav_i.update(qpos_np[i], viewer=viewer)


def _resolve_headless_steps(
    *,
    collect: bool,
    steps: int | None,
    episode_duration_s: float,
    sim_hz: float,
) -> int:
    """Default live runs stay short; collection must outlive the 30 s gate."""
    if steps is not None:
        return int(steps)
    if collect:
        return int(round(float(episode_duration_s) * float(sim_hz)))
    return 2000


def _scene_xml_path(robot: str, scene: str) -> str:
    return os.path.abspath(os.path.join(dir_path, "..", "..", "data", robot, f"scene_{scene}.xml"))


def _settle_world(model, data, config, sim_frequency: float) -> None:
    if not spawn_config.settle_after_spawn:
        return
    n_steps = int(round(spawn_config.settle_duration_s * sim_frequency))
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


def _is_crashed_one(qpos, collapse_counter: int, config, sim_frequency: float) -> tuple[bool, int]:
    crash_h = config.robot_height * 0.5
    collapse_h = config.robot_height * 0.65
    collapse_dwell = int(0.5 * sim_frequency)
    tilt = np.deg2rad(60.0)
    z = float(qpos[2])
    if z < collapse_h:
        collapse_counter += 1
    else:
        collapse_counter = 0
    if collapse_counter >= collapse_dwell or z < crash_h:
        return True, collapse_counter
    w, x, y, zq = (float(qpos[i]) for i in range(3, 7))
    roll = np.arctan2(2.0 * (w * x + y * zq), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - zq * x), -1.0, 1.0))
    if abs(roll) > tilt or abs(pitch) > tilt:
        return True, collapse_counter
    return False, collapse_counter


def _main_collect(
    *,
    headless: bool,
    steps: int,
    scene: str,
    robot: str,
    nav: str,
    n_env: int,
    mpc_model: str,
    collect_out,
    episode_duration_s,
    gait,
):
    """N CPU MuJoCo worlds, batched MPC, one dataset run directory."""
    config = _robot_config(robot, mpc_model=mpc_model, gait=gait)
    if hasattr(config, "gait"):
        print(f"[gait] {config.gait.summary()}", flush=True)

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

    mpc = make_locomotion_mpc(config, limited_memory=True)

    def _run_one(mpc_data_i, x0_i, command_i, contact_i):
        return mpc.run(mpc_data_i, x0_i, command_i, contact_i)

    batched_solve = jax.jit(jax.vmap(_run_one))

    collectors = setup_multi_env_collection(
        True,
        n_env,
        gait_type=gait_from_phase_offsets(config.timer_t),
        scene=scene,
        sim_hz=sim_frequency,
        robot=robot,
        episode_duration_s=episode_duration_s,
        collect_out=collect_out,
        cfg=dataset_collection_config,
    )

    use_navigation, use_command_sampler = _collect_command_source(
        nav, dataset_collection_config.episode.segmented_commands
    )
    command_handle = KeyboardVelocityCommand()
    navigators = [
        PointNavigator(
            robot_height=config.robot_height,
            auto_resample=(nav == "random"),
            control_dt=1.0 / config.mpc_frequency,
        )
        for _ in range(n_env)
    ]
    samplers = [
        VelocityCommandSampler(dt=1.0 / dataset_collection_config.episode.control_hz)
        for _ in range(n_env)
    ]
    if use_command_sampler:
        print(
            "[command] --collect without a nav mode: random velocity segments.",
            flush=True,
        )
    elif use_navigation:
        print(f"[command] --nav {nav} --collect: driving to navigation goals.", flush=True)
    else:
        print(f"[command] driving from --nav {nav}", flush=True)

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
        ResetRandomizer.from_config(loco_reset_randomization_config) for _ in range(n_env)
    ]

    episode_seeds = [0] * n_env
    collapse_counters = [0] * n_env
    commands = [None] * n_env
    tau_np = np.zeros((n_env, config.n_joints), dtype=np.float64)

    def _pack_env(i: int):
        foot = jnp.asarray(sim_utils.geom_positions(datas[i], contact_ids[i]))
        x0 = mpc.pack_state(datas[i].qpos.copy(), datas[i].qvel.copy(), foot)
        return x0, foot

    def _randomize(i: int, mpc_data_i, *, label: str = "episode"):
        episode_seeds[i] += 1
        sample, mpc_data_i = randomizers[i].sample_and_apply(
            ResetTargets(
                model=models[i],
                foot_geom_ids=contact_ids[i],
                base_weight=weights[i],
                navigator=navigators[i] if use_navigation else None,
                command_sampler=samplers[i] if use_command_sampler else None,
                mpc_data=mpc_data_i,
            )
        )
        collectors.env[i].set_episode_conditions(
            randomization=sample.to_metadata(),
            seed=episode_seeds[i],
            mode="locomotion",
            **read_episode_conditions(models[i], contact_ids[i], weights[i]),
        )
        return mpc_data_i

    def _respawn(i: int, batch_mpc, *, crashed: bool = False):
        collectors.env[i].on_respawn(crashed=crashed)
        collapse_counters[i] = 0
        spawners[i].apply_to_data(
            models[i], datas[i], config.p0, config.quat0, config.q0
        )
        _settle_world(models[i], datas[i], config, sim_frequency)
        x0, foot = _pack_env(i)
        single = mpc.reset(mpc.make_data(), datas[i].qpos.copy(), datas[i].qvel.copy(), foot)
        single = _randomize(i, single, label="respawn")
        perturbers[i].reset()
        samplers[i].reset()
        if nav == "random":
            navigators[i].reset(np.asarray(datas[i].qpos))
        tau_np[i] = 0.0
        return _tree_set_index(batch_mpc, i, single), x0, foot

    batch_mpc = jax.vmap(lambda _: mpc.make_data())(jnp.arange(n_env))
    for i in range(n_env):
        batch_mpc, _, _ = _respawn(i, batch_mpc)

    x0_list = []
    cmd_list = []
    contact_list = []
    for i in range(n_env):
        x0, _ = _pack_env(i)
        x0_list.append(x0)
        cmd_list.append(jnp.asarray(command_handle.mpc_input(config.robot_height)))
        contact_list.append(jnp.asarray(estimate_contacts(datas[i], contact_ids[i])))
    batch_mpc, tau_warm = batched_solve(
        batch_mpc,
        jnp.stack(x0_list),
        jnp.stack(cmd_list),
        jnp.stack(contact_list) * (1.0 if getattr(config, "mpc_model", "") == "srbd" else 0.0),
    )
    tau_warm.block_until_ready()

    for i in range(n_env):
        batch_mpc, _, _ = _respawn(i, batch_mpc)
    collectors.on_ready()

    period = int(sim_frequency / config.mpc_frequency)
    print(
        f"[multiEnv collect] {n_env} CPU worlds | {period} steps/MPC | "
        f"{sim_frequency:.0f} Hz | nav={nav} | mpc={mpc_model}",
        flush=True,
    )
    counter = 0
    episode_cfg = dataset_collection_config.episode

    def step_all():
        nonlocal counter, batch_mpc
        if counter % period == 0:
            x0_list = []
            cmd_list = []
            contact_list = []
            for i in range(n_env):
                x0, _ = _pack_env(i)
                x0_list.append(x0)
                if use_command_sampler:
                    cmd = jnp.asarray(samplers[i].mpc_input(config.robot_height))
                elif use_navigation:
                    cmd = jnp.asarray(
                        navigators[i].mpc_input(datas[i].qpos, config.robot_height)
                    )
                else:
                    cmd = jnp.asarray(command_handle.mpc_input(config.robot_height))
                commands[i] = cmd
                contact = jnp.asarray(estimate_contacts(datas[i], contact_ids[i]))
                if getattr(config, "mpc_model", "whole_body") != "srbd":
                    contact = contact * 0.0
                cmd_list.append(cmd)
                contact_list.append(contact)
            start = timer()
            batch_mpc, tau_batch = batched_solve(
                batch_mpc,
                jnp.stack(x0_list),
                jnp.stack(cmd_list),
                jnp.stack(contact_list),
            )
            tau_batch.block_until_ready()
            tau_np[:] = np.asarray(tau_batch)
            if counter == 0 or counter % (period * 50) == 0:
                print(f"  step {counter:6d}  batched MPC {1e3 * (timer() - start):.1f} ms", flush=True)

        if use_command_sampler:
            for s in samplers:
                s.step()

        for i in range(n_env):
            datas[i].ctrl = tau_np[i]
            perturbers[i].tick_and_apply(datas[i])
            weights[i].apply(datas[i])
            mujoco.mj_step(models[i], datas[i])

            closed = collectors.env[i].after_physics_step(
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
                    cmd_base_vel=(
                        command_from_mpc_input(np.asarray(commands[i]))
                        if commands[i] is not None
                        else None
                    ),
                    cmd_segment_id=samplers[i].segment_id if use_command_sampler else 0,
                ),
            )
            if closed:
                md_i = jax.tree.map(lambda x: x[i], batch_mpc)
                md_i = _randomize(i, md_i)
                batch_mpc = _tree_set_index(batch_mpc, i, md_i)

            if use_navigation and navigators[i].reached(datas[i].qpos):
                long_enough = (
                    collectors.env[i].episode_seconds
                    >= episode_cfg.goal_closes_episode_after_s
                )
                if episode_cfg.end_episode_on_goal and long_enough:
                    collectors.env[i].end_episode(reason="goal_reached")
                    md_i = jax.tree.map(lambda x: x[i], batch_mpc)
                    md_i = _randomize(i, md_i)
                    batch_mpc = _tree_set_index(batch_mpc, i, md_i)
                if navigators[i].auto_resample:
                    navigators[i].sample_goal(np.asarray(datas[i].qpos))

            crashed, collapse_counters[i] = _is_crashed_one(
                datas[i].qpos, collapse_counters[i], config, sim_frequency
            )
            if crashed:
                batch_mpc, _, _ = _respawn(i, batch_mpc, crashed=True)

        counter += 1

    if headless:
        for _ in range(steps):
            step_all()
        collectors.finish("")
        return

    scratch = mujoco.MjData(models[0])
    ghost_geoms = [None] * n_env
    robots_per_row = math.ceil(math.sqrt(n_env))
    grid = np.arange(robots_per_row ** 2)
    offset_xy = np.stack(
        [
            (grid % robots_per_row)[:n_env] * 1.5,
            (grid // robots_per_row)[:n_env] * 1.5,
        ],
        axis=1,
    )

    def key_callback(key: int):
        command_handle.key_callback(key)

    with mujoco.viewer.launch_passive(models[0], datas[0], key_callback=key_callback) as viewer:
        viewer.sync()
        while viewer.is_running():
            tic = timer()
            step_all()
            for i in range(n_env):
                qp_vis = np.asarray(datas[i].qpos).copy()
                qp_vis[0] += offset_xy[i, 0]
                qp_vis[1] += offset_xy[i, 1]
                scratch.qpos[: qp_vis.size] = qp_vis
                mujoco.mj_forward(models[0], scratch)
                ghost_geoms[i] = sim_utils.render_ghost_robot(
                    viewer, models[0], scratch, alpha=0.9, ghost_geoms=ghost_geoms[i]
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
    nav: str = "random",
    n_env: int = 8,
    mpc_model: str = "whole_body",
    collect: bool = False,
    collect_out=None,
    episode_duration_s=None,
    gait=None,
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
            nav=nav,
            n_env=n_env,
            mpc_model=mpc_model,
            collect_out=collect_out,
            episode_duration_s=episode_duration_s,
            gait=gait,
        )

    # ── Config & CPU model ──────────────────────────────────────────────────
    config = _robot_config(robot, mpc_model=mpc_model, gait=gait)

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
    mpc = make_locomotion_mpc(config, limited_memory=True)

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
        return mpc.pack_state(mjx_d.qpos, mjx_d.qvel, foot_pos), foot_pos

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

            if use_navigation:
                _tick_navigators(navigators, batch_data.qpos)

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
                _tick_navigators(navigators, batch_data.qpos)

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
    parser.add_argument("--robot", type=str,
                        choices=["go2", "b2"], default="go2")
    parser.add_argument("--nav", type=str,
                        choices=["random", "vel"], default="random",
                        help="random: each robot gets its own random goal "
                             "(works headless); vel: keyboard command broadcast "
                             "to all robots.")
    parser.add_argument("--n-env", type=int, default=8,
                        help="Number of parallel environments.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--mpc-model",
        type=str,
        choices=["whole_body", "inverse_dynamics", "srbd"],
        default="whole_body",
        help="Go2 MPC transcription: whole-body, inverse-dynamics, or SRBD.",
    )
    parser.add_argument(
        "--gait",
        type=str,
        choices=["trot", "pace", "crawl", "bound"],
        default=None,
        help="Locomotion gait (go2 / b2).",
    )
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
        nav=args.nav,
        n_env=args.n_env,
        mpc_model=args.mpc_model,
        collect=args.collect,
        collect_out=args.collect_out,
        episode_duration_s=args.episode_duration,
        gait=args.gait,
    )
