"""Multi-environment 3-contact (tripod) balance simulator for the Go2 quadruped.

N robots run in parallel on the GPU via JAX/MJX.  Each robot has its own:
  - MPC warm-start (``batch_mpc_data``)
  - Desired body pose from ``DesiredPoseSampler`` (disabled by default — flat upright)
  - Per-robot tripod foot anchor sampled via ``tripod_foot_reference_world``
    (swing foot pushed forward by SWING_INIT_OFFSET at each reset)
  - Swing goal sphere rendered per robot in the viewer

Robot 0 is the live viewer body, so MuJoCo contact visualization matches
that mesh. The other robots are ghost overlays on a square grid.

``--collect`` switches to N CPU MuJoCo worlds (dataset ``sim_hz``), one batched
MPC solve, and one run directory — the same collector layout as
``quad_locomotion_multiEnv``. Without ``--collect`` the MJX ghost demo is unchanged.

Usage::

    python -m mpx.simulators.quadruped.quad_3balance_multiEnv --n-env 8
    python -m mpx.simulators.quadruped.quad_3balance_multiEnv --n-env 16 --scene flat
    python -m mpx.simulators.quadruped.quad_3balance_multiEnv --headless --n-env 64
    python -m mpx.simulators.quadruped.quad_3balance_multiEnv --collect --n-env 8 --headless
"""

import argparse
import math
import os
import sys
import time
from dataclasses import replace
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
from mpx.config.sim_config.config_base_weight import base_weight_config
from mpx.utils.simulation_utils.base_weight import BaseWeightForce
from mpx.config.sim_config.config_reset_randomization import balance_reset_randomization_config
from mpx.utils.simulation_utils.reset_randomizer import ResetRandomizer, ResetTargets
from mpx.config.sim_config.config_quad_spawn import spawn_config
from mpx.utils.spawner.spawner import RobotMapSpawner

from mpx.utils.dataset_collection.episode_recorder import (
    ControlSample,
    read_episode_conditions,
    setup_multi_env_collection,
)
from mpx.utils.dataset_collection.dataset_bucket_system import GaitType
from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config

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
    swing_foot_at_goal,
    base_yaw_offset_to_world,
)
from mpx.config.sim_config.config_foot_ref_config import foot_ref_config, random_swing_foot_config
import glfw

import mpx.utils.simulation_utils.sim_utils as sim_utils
from mpx.utils.math_utils.quad_math import (
    yaw_from_quat, _quat_to_axes, quat_normalize_wxyz, quat_mul_wxyz,
)


def _robot_config(robot: str):
    if robot == "go2":
        return go2_config(Go2Mode.BALANCE, balance_stance=BalanceStance.TRIPOD_SWING_FL)
    if robot == "b2":
        from mpx.config.robot_config.config_b2 import b2_config, B2Mode
        return b2_config(B2Mode.BALANCE, balance_stance=BalanceStance.TRIPOD_SWING_FL)
    raise ValueError(f"Unsupported robot: {robot}")


# Hold the swing target this long, then respawn and pick a new swing foot.
_GOAL_HOLD_S = 3.0
# Right after a reset the robot is already still and the foot is not on the new target.
_STUCK_GRACE_S = 0.6
# Settled outside the arrival tolerance for this long → respawn.
_STUCK_STILL_S = 0.4
_STUCK_BASE_SPEED = 0.08   # m/s
_STUCK_JOINT_SPEED = 0.5   # rad/s


def _is_standing_still(qvel: np.ndarray) -> bool:
    """True when the base and the joints have settled."""
    qvel = np.asarray(qvel, dtype=np.float64).reshape(-1)
    if float(np.linalg.norm(qvel[:3])) >= _STUCK_BASE_SPEED:
        return False
    joints = qvel[6:]
    if joints.size == 0:
        return True
    return float(np.max(np.abs(joints))) < _STUCK_JOINT_SPEED


def _update_stuck(
    *,
    arrived: bool,
    qvel: np.ndarray,
    dt: float,
    since_reset_s: float,
    still_outside_s: float,
) -> tuple[float, float, bool]:
    """Advance the settled-miss timer.

    Returns ``(since_reset_s, still_outside_s, stuck)``. A miss only counts
    after ``_STUCK_GRACE_S``, so a fresh spawn is not reset before the robot
    starts walking to the target.
    """
    since_reset_s += float(dt)
    if arrived or since_reset_s < _STUCK_GRACE_S or not _is_standing_still(qvel):
        return since_reset_s, 0.0, False
    still_outside_s += float(dt)
    return since_reset_s, still_outside_s, still_outside_s >= _STUCK_STILL_S


def _swing_contact_mask(n_contact: int, swing_leg: int) -> np.ndarray:
    """1 on stance feet, 0 on the swing foot. That is the MPC contact mask."""
    mask = np.ones(n_contact, dtype=np.float32)
    mask[int(swing_leg)] = 0.0
    return mask


def _signed_interval(bounds, sign: float) -> tuple[float, float]:
    lo, hi = sorted((abs(float(bounds[0])), abs(float(bounds[1]))))
    if float(bounds[0]) < 0.0 < float(bounds[1]):
        lo = 0.0
    if sign < 0.0:
        return (-hi, -lo)
    return (lo, hi)


def _mirror_swing_offset(p_legs0, leg_idx: int, xyz: np.ndarray) -> np.ndarray:
    """Keep a base-frame sample in the nominal quadrant of ``leg_idx``."""
    nom = np.asarray(p_legs0, dtype=np.float64).reshape(-1)
    nom = nom[3 * int(leg_idx) : 3 * int(leg_idx) + 3]
    return np.array(
        [
            np.copysign(abs(float(xyz[0])), float(nom[0])),
            np.copysign(abs(float(xyz[1])), float(nom[1])),
            float(xyz[2]),
        ],
        dtype=np.float64,
    )


def _bounds_for_leg(cfg, p_legs0, leg_idx: int):
    nom = np.asarray(p_legs0, dtype=np.float64).reshape(-1)
    nom = nom[3 * int(leg_idx) : 3 * int(leg_idx) + 3]
    return replace(
        cfg,
        x_bounds=_signed_interval(cfg.x_bounds, float(nom[0])),
        y_bounds=_signed_interval(cfg.y_bounds, float(nom[1])),
    )


def _random_swing_world(sampler, p_legs0, leg_idx: int, base_pos, base_quat) -> np.ndarray:
    xyz = _mirror_swing_offset(p_legs0, leg_idx, sampler.sample_offset_base())
    return base_yaw_offset_to_world(base_pos, base_quat, xyz)


def _draw_swing_bounds(marker, viewer, sampler, p_legs0, leg_idx, base_pos, base_quat) -> None:
    if marker is None or viewer is None:
        return
    if sampler.enabled and sampler.cfg.show_bounds_box:
        marker.set_frame(base_pos, base_quat)
    else:
        marker.clear()
    marker.draw(viewer, _bounds_for_leg(sampler.cfg, p_legs0, leg_idx), sync=False)


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
    """N CPU MuJoCo worlds, batched tripod MPC, one dataset run directory."""
    config = _robot_config(robot)
    swing_init_offset = np.array([0.25, -0.1, -0.3])
    swing_legs = np.zeros(n_env, dtype=np.int32)
    contact_masks = np.ones((n_env, config.n_contact), dtype=np.float32)
    goal_hold_s = np.zeros(n_env, dtype=np.float64)
    since_reset_s = np.zeros(n_env, dtype=np.float64)
    still_outside_s = np.zeros(n_env, dtype=np.float64)
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

    def _run_one(mpc_data_i, x0_i, command_i, contact_i,
                 base_quat_ref_i, use_base_quat_ref_i,
                 foot_ref_anchor_i, use_foot_ref_anchor_i):
        return mpc.run(
            mpc_data_i, x0_i, command_i, contact_i,
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
        name_prefix="balance",
        cfg=dataset_collection_config,
    )

    desired_pose_sampler = DesiredPoseSampler.from_config(config.robot_height, desired_pose_config)
    foot_ref_mgr = FootReferenceManager(foot_ref_config)
    random_swing_sampler = RandomSwingFootSampler(random_swing_foot_config)
    print(
        f"[random_swing] mode={'ON' if random_swing_sampler.enabled else 'OFF'}  "
        f"bounds: {random_swing_sampler.bounds_summary()}",
        flush=True,
    )
    desired_heights = np.full(n_env, config.robot_height, dtype=np.float64)
    desired_quats = np.tile(np.asarray(config.quat0, dtype=np.float32), (n_env, 1))
    foot_anchors = np.zeros((n_env, 3 * config.n_contact), dtype=np.float32)
    arrival_cooldown_steps = np.zeros(n_env, dtype=np.int32)
    arrival_hold_steps = np.zeros(n_env, dtype=np.int32)

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

    def _feet(i: int) -> np.ndarray:
        return np.ravel(np.asarray(
            sim_utils.geom_positions(datas[i], contact_ids[i]), dtype=np.float64
        ))

    def _pack_env(i: int):
        foot = jnp.asarray(_feet(i))
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(jnp.asarray(datas[i].qpos))
            .at[mpc.qvel_slice].set(jnp.asarray(datas[i].qvel))
            .at[mpc.foot_slice].set(foot)
        )
        return x0, foot

    def _choose_swing_leg(i: int) -> int:
        leg = int(np.random.randint(0, config.n_contact))
        swing_legs[i] = leg
        contact_masks[i] = _swing_contact_mask(config.n_contact, leg)
        goal_hold_s[i] = 0.0
        since_reset_s[i] = 0.0
        still_outside_s[i] = 0.0
        return leg

    def _sample_pose_and_anchor(i: int) -> None:
        leg = int(swing_legs[i])
        height, delta_quat = desired_pose_sampler.sample()
        desired_heights[i] = height
        spawn_quat = np.asarray(datas[i].qpos[3:7], dtype=np.float64)
        desired_quats[i] = quat_normalize_wxyz(
            quat_mul_wxyz(spawn_quat, delta_quat)
        ).astype(np.float32)
        rng_key = jax.random.PRNGKey(int(time.time() * 1000 + i) & 0x7FFFFFFF)
        anchor = np.asarray(
            foot_ref_mgr.tripod_foot_reference_world(
                key=rng_key,
                p=jnp.asarray(datas[i].qpos[:3]),
                quat=jnp.asarray(datas[i].qpos[3:7]),
                foot0=jnp.asarray(config.p_legs0),
                n_contact=config.n_contact,
                sigma=np.array([0.04, 0.04, 0.0]),
            ),
            dtype=np.float64,
        )
        measured_swing = _feet(i)[3 * leg : 3 * leg + 3]
        if random_swing_sampler.resample_on_respawn:
            new_swing = _random_swing_world(
                random_swing_sampler, config.p_legs0, leg,
                datas[i].qpos[:3], datas[i].qpos[3:7],
            )
        else:
            new_swing = foot_target_foot_local_to_world(
                measured_swing, datas[i].qpos[3:7], swing_init_offset
            )
        foot_anchors[i] = swing_foot_anchor_from_target(
            anchor, leg, new_swing
        ).astype(np.float32)
        arrival_cooldown_steps[i] = 0
        arrival_hold_steps[i] = 0

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
            mode=f"stand_3leg_{config.contact_frame[int(swing_legs[i])]}",
            **read_episode_conditions(models[i], contact_ids[i], weights[i]),
        )
        return mpc_data_i

    def _respawn(i: int, batch_mpc, *, crashed: bool = False, reason: str = "respawn"):
        collectors.env[i].on_respawn(crashed=crashed)
        leg = _choose_swing_leg(i)
        spawners[i].apply_to_data(models[i], datas[i], config.p0, config.quat0, config.q0)
        _settle_world(models[i], datas[i], config, sim_frequency)
        _sample_pose_and_anchor(i)
        print(
            f"  [{reason}] robot {i} swing={config.contact_frame[leg]}",
            flush=True,
        )
        x0, foot = _pack_env(i)
        single = mpc.reset(mpc.make_data(), datas[i].qpos.copy(), datas[i].qvel.copy(), foot)
        single = _randomize(i, single)
        perturbers[i].reset()
        tau_np[i] = 0.0
        return _tree_set_index(batch_mpc, i, single), x0, foot

    batch_mpc = jax.vmap(lambda _: mpc.make_data())(jnp.arange(n_env))
    for i in range(n_env):
        batch_mpc, _, _ = _respawn(i, batch_mpc)

    def _solve_batch():
        x0_list, cmd_list, quat_list, anchor_list = [], [], [], []
        for i in range(n_env):
            x0, _ = _pack_env(i)
            cmd = np.zeros(7, dtype=np.float64)
            cmd[6] = desired_heights[i]
            x0_list.append(x0)
            cmd_list.append(jnp.asarray(cmd))
            quat_list.append(jnp.asarray(desired_quats[i]))
            anchor_list.append(jnp.asarray(foot_anchors[i]))
        return batched_solve(
            batch_mpc,
            jnp.stack(x0_list),
            jnp.stack(cmd_list),
            jnp.asarray(contact_masks),
            jnp.stack(quat_list),
            jnp.ones(n_env, dtype=bool),
            jnp.stack(anchor_list),
            jnp.ones(n_env, dtype=bool),
        )

    batch_mpc, tau_warm = _solve_batch()
    tau_warm.block_until_ready()
    for i in range(n_env):
        batch_mpc, _, _ = _respawn(i, batch_mpc)
    collectors.on_ready()

    period = int(sim_frequency / config.mpc_frequency)
    print(
        f"[3balance collect] {n_env} CPU worlds | {period} steps/MPC | "
        f"{sim_frequency:.0f} Hz | swing foot chosen on each respawn | scene={scene}",
        flush=True,
    )
    counter = 0

    def step_all():
        nonlocal counter, batch_mpc
        if counter % period == 0:
            start = timer()
            batch_mpc, tau_batch = _solve_batch()
            tau_batch.block_until_ready()
            tau_np[:] = np.asarray(tau_batch)
            if counter == 0 or counter % (period * 50) == 0:
                print(f"  step {counter:6d}  batched MPC {1e3 * (timer() - start):.1f} ms", flush=True)

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
                    # The gait timer is bypassed (duty 1). The plan is the
                    # swing-foot mask, not phase < duty_factor.
                    planned_contact=np.asarray(contact_masks[i], dtype=np.uint8),
                ),
            )
            crashed = bool(
                _is_crashed_batch(np.asarray(datas[i].qpos)[None], crash_height, crash_tilt)[0]
            )
            leg = int(swing_legs[i])
            measured = _feet(i)[3 * leg : 3 * leg + 3]
            target = foot_anchors[i, 3 * leg : 3 * leg + 3]
            dt = float(models[i].opt.timestep)
            arrived, _, _ = swing_foot_at_goal(
                measured, target, random_swing_sampler.cfg
            )
            goal_hold_s[i] = goal_hold_s[i] + dt if arrived else 0.0
            held = goal_hold_s[i] >= _GOAL_HOLD_S
            since_reset_s[i], still_outside_s[i], stuck = _update_stuck(
                arrived=arrived,
                qvel=datas[i].qvel,
                dt=dt,
                since_reset_s=float(since_reset_s[i]),
                still_outside_s=float(still_outside_s[i]),
            )
            if closed and not crashed and not held and not stuck:
                md_i = jax.tree.map(lambda x: x[i], batch_mpc)
                md_i = _randomize(i, md_i)
                batch_mpc = _tree_set_index(batch_mpc, i, md_i)
            if crashed or held or stuck:
                if crashed:
                    reason = "crash"
                elif held:
                    reason = "goal"
                else:
                    reason = "stuck"
                batch_mpc, _, _ = _respawn(i, batch_mpc, crashed=crashed, reason=reason)

        counter += 1

    if headless:
        for _ in range(steps):
            step_all()
        collectors.finish("")
        return

    scratch = mujoco.MjData(models[0])
    ghost_geoms = [None] * n_env
    swing_goal_ids = [-1] * n_env
    swing_bounds_markers = [None] * n_env
    robots_per_row = math.ceil(math.sqrt(n_env))
    grid = np.arange(robots_per_row ** 2)
    offset_xy = np.stack(
        [
            (grid % robots_per_row)[:n_env] * 1.5,
            (grid // robots_per_row)[:n_env] * 1.5,
        ],
        axis=1,
    )
    def key_callback(key: int) -> None:
        nonlocal batch_mpc
        if key in spawn_config.respawn_keycodes:
            for i in range(n_env):
                batch_mpc, _, _ = _respawn(i, batch_mpc, reason="key")
        elif key == glfw.KEY_R:
            for i in range(n_env):
                leg = int(swing_legs[i])
                new_swing = _random_swing_world(
                    random_swing_sampler, config.p_legs0, leg,
                    datas[i].qpos[:3], datas[i].qpos[3:7],
                )
                foot_anchors[i] = swing_foot_anchor_from_target(
                    foot_anchors[i], leg, new_swing
                ).astype(np.float32)
                goal_hold_s[i] = 0.0
                since_reset_s[i] = 0.0
                still_outside_s[i] = 0.0
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

    with mujoco.viewer.launch_passive(models[0], datas[0], key_callback=key_callback) as viewer:
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
                base_vis = qp_vis[:3].copy()
                if swing_bounds_markers[i] is None:
                    swing_bounds_markers[i] = random_swing_sampler.attach_bounds_box_marker(viewer)
                leg = int(swing_legs[i])
                _draw_swing_bounds(
                    swing_bounds_markers[i], viewer, random_swing_sampler,
                    config.p_legs0, leg, base_vis, qp_vis[3:7],
                )
                swing_world = foot_anchors[i, 3 * leg : 3 * leg + 3]
                goal_pos = np.array([
                    float(swing_world[0]) + offset_xy[i, 0],
                    float(swing_world[1]) + offset_xy[i, 1],
                    float(swing_world[2]),
                ])
                swing_goal_ids[i] = sim_utils.render_sphere(
                    viewer,
                    position=goal_pos,
                    diameter=2.0 * foot_ref_config.swing_goal_radius,
                    color=np.array([1.0, 0.5, 0.0, 0.8]),
                    geom_id=swing_goal_ids[i],
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

    # ── Config ──────────────────────────────────────────────────────────────
    config = _robot_config(robot)

    SWING_INIT_OFFSET = np.array([0.25, -0.1, -0.3])   # forward+up offset in body frame

    print(
        f"\n[3balance_multiEnv] {n_env} robots | swing foot chosen on each respawn | "
        f"scene: {scene}\n"
        "  B / Backspace — respawn all robots and pick a new swing foot\n"
        "  R             — randomise all swing foot targets NOW (within bounds)\n"
        "  N             — toggle random-on-respawn mode ON/OFF\n"
        f"  goal hold     — respawn after {_GOAL_HOLD_S:.0f}s inside arrival tolerance\n"
        f"  stuck         — respawn if settled outside tolerance for {_STUCK_STILL_S:.1f}s\n",
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

    def _run_one(mpc_data_i, x0_i, command_i, contact_i,
                 base_quat_ref_i, use_base_quat_ref_i,
                 foot_ref_anchor_i, use_foot_ref_anchor_i):
        return mpc.run(
            mpc_data_i, x0_i, command_i, contact_i,
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
    swing_legs = np.zeros(n_env, dtype=np.int32)
    contact_masks = np.ones((n_env, config.n_contact), dtype=np.float32)
    goal_hold_s = np.zeros(n_env, dtype=np.float64)
    since_reset_s = np.zeros(n_env, dtype=np.float64)
    still_outside_s = np.zeros(n_env, dtype=np.float64)
    pending_reset = np.zeros(n_env, dtype=bool)

    def _sample_foot_anchor(
        i: int, qpos_i: np.ndarray, measured_swing_world: np.ndarray, leg: int,
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
            new_swing = _random_swing_world(
                random_swing_sampler, config.p_legs0, leg, qpos_i[:3], qpos_i[3:7],
            )
        else:
            new_swing = foot_target_foot_local_to_world(origin, qpos_i[3:7], SWING_INIT_OFFSET)
        return swing_foot_anchor_from_target(anchor, leg, new_swing).astype(np.float32)

    def _reset_robot(i: int, qpos_np: np.ndarray, *, reason: str = "respawn") -> None:
        """Reset robot i, then pick a new swing foot and its target."""
        leg = int(np.random.randint(0, config.n_contact))
        swing_legs[i] = leg
        contact_masks[i] = _swing_contact_mask(config.n_contact, leg)
        goal_hold_s[i] = 0.0
        since_reset_s[i] = 0.0
        still_outside_s[i] = 0.0
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
            3 * leg : 3 * leg + 3
        ]
        foot_anchors[i] = _sample_foot_anchor(i, qpos_np[i], measured_swing, leg)
        perturbers[i].reset()
        print(f"  [{reason}] robot {i} swing={config.contact_frame[leg]}", flush=True)

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
        jnp.asarray(contact_masks),
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
    def _handle_resets() -> np.ndarray:
        """Respawn on crash, goal hold, a settled miss, or the manual reset key."""
        nonlocal batch_data, batch_mpc_data, pending_reset
        qpos_np = np.array(batch_data.qpos)
        qvel_np = np.array(batch_data.qvel)
        crashed = _is_crashed_batch(qpos_np, CRASH_HEIGHT, CRASH_TILT)
        feet = np.concatenate(
            [
                np.asarray(batch_data.geom_xpos[gid], dtype=np.float64)
                for gid in mjx_contact_ids
            ],
            axis=1,
        )
        held = np.zeros(n_env, dtype=bool)
        stuck = np.zeros(n_env, dtype=bool)
        dt = float(model.opt.timestep)
        for i in range(n_env):
            if crashed[i] or pending_reset[i]:
                continue
            leg = int(swing_legs[i])
            measured = feet[i, 3 * leg : 3 * leg + 3]
            target = np.asarray(foot_anchors[i, 3 * leg : 3 * leg + 3], dtype=np.float64)
            arrived, _, _ = swing_foot_at_goal(measured, target, random_swing_sampler.cfg)
            goal_hold_s[i] = goal_hold_s[i] + dt if arrived else 0.0
            held[i] = goal_hold_s[i] >= _GOAL_HOLD_S
            since_reset_s[i], still_outside_s[i], stuck[i] = _update_stuck(
                arrived=arrived,
                qvel=qvel_np[i],
                dt=dt,
                since_reset_s=float(since_reset_s[i]),
                still_outside_s=float(still_outside_s[i]),
            )
        reset = crashed | held | stuck | pending_reset
        pending_reset[:] = False
        if not reset.any():
            return qpos_np
        for i in np.where(reset)[0]:
            if crashed[i]:
                reason = "crash"
            elif held[i]:
                reason = "goal"
            elif stuck[i]:
                reason = "stuck"
            else:
                reason = "key"
            _reset_robot(i, qpos_np, reason=reason)
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
                batch_x0, _ = build_x0_batch(batch_data)
                batch_cmd = jnp.zeros((n_env, 7)).at[:, 6].set(
                    jnp.asarray(desired_heights, dtype=jnp.float32)
                )
                start = timer()
                batch_mpc_data, tau_batch = batched_solve(
                    batch_mpc_data, batch_x0, batch_cmd,
                    jnp.asarray(contact_masks),
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
    ghost_geoms  = [None] * n_env

    # Swing goal spheres and random-bounds boxes per robot.
    _swing_goal_ids = [-1] * n_env
    _swing_bounds_markers: list = [None] * n_env

    def key_callback(key: int) -> None:
        nonlocal foot_anchors
        if key in spawn_config.respawn_keycodes:
            pending_reset[:] = True
        elif key == glfw.KEY_R:
            qpos_cur = np.array(batch_data.qpos)
            for i in range(n_env):
                leg = int(swing_legs[i])
                new_swing = _random_swing_world(
                    random_swing_sampler, config.p_legs0, leg,
                    qpos_cur[i, :3], qpos_cur[i, 3:7],
                )
                foot_anchors[i] = swing_foot_anchor_from_target(
                    foot_anchors[i], leg, new_swing
                ).astype(np.float32)
                goal_hold_s[i] = 0.0
                since_reset_s[i] = 0.0
                still_outside_s[i] = 0.0
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

        # Pre-build ghost geom caches. Env 0 is the live viewer body.
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
                    jnp.asarray(contact_masks),
                    jnp.asarray(desired_quats),
                    jnp.ones(n_env, dtype=bool),
                    jnp.asarray(foot_anchors),
                    jnp.ones(n_env, dtype=bool),
                )
                tau_batch.block_until_ready()
                print(f"  step {counter:5d}  batched MPC {1e3*(timer()-start):.1f} ms", flush=True)

            # ── Physics step ─────────────────────────────────────────────────
            batch_data = batched_step(batch_data, tau_batch)

            qpos_np = _handle_resets()

            # Env 0 is the body the viewer steps for contact visualization.
            data.qpos[:] = qpos_np[0]
            data.qvel[:] = np.asarray(batch_data.qvel[0])
            mujoco.mj_forward(model, data)

            # ── Render ghost robots (envs 1..N-1) ─────────────────────────────
            for i in range(1, n_env):
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
                leg = int(swing_legs[i])
                _draw_swing_bounds(
                    _swing_bounds_markers[i], viewer, random_swing_sampler,
                    config.p_legs0, leg, base_vis, qpos_vis[i, 3:7],
                )

                swing_world = foot_anchors[i, 3 * leg : 3 * leg + 3].astype(np.float64)
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
