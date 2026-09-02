#Usage: python quad_4balance.py --collect --collect-out 4balance_flat_001.npz --episode-duration 60.0
# tag: dataset collection   collect_out: 4balance_flat_001.npz, episode_duration: 60.0

import argparse
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

import types

# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from mpx.config.robot_config.config_go2 import go2_config, Go2Mode, BalanceStance
from mpx.utils.quad_utils_balance.mpc_wrapper_4balance import MPCWrapper

from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config, ExtBaseForceConfig
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation

from mpx.config.sim_config.config_quad_spawn import spawn_config, SpawnConfig
from mpx.utils.spawner.spawner import RobotMapSpawner
from mpx.utils.quad_utils_balance.desired_pose_sampler import (
    DesiredPoseSampler, desired_pose_config,
)

import mpx.utils.simulation_utils.sim_utils as sim_utils
from mpx.utils.math_utils.quad_math import yaw_from_quat, _quat_to_axes, quat_normalize_wxyz, quat_mul_wxyz

from mpx.utils.dataset_collection.episode_recorder import setup_sim_collection
from mpx.utils.dataset_collection.dataset_bucket_system import GaitType
from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config

from mpx.estimators.quad_contact_estimation import estimate_contacts, print_contact_friction

# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

#region ================Helper functions================
def robot_config(robot):
    if(robot == "go2"):
        GO2_CONTROLLER_MODE = Go2Mode.BALANCE
        config = go2_config(GO2_CONTROLLER_MODE,balance_stance=BalanceStance.FOUR)
    elif(robot == "aliengo"):
        import mpx.config.robot_config.config_aliengo as config
    return config
#------------------------------------------------
def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact,
                  base_quat_ref, use_base_quat_ref,
                  foot_ref_anchor, use_foot_ref_anchor):
        x0 = (mpc.initial_state
            .at[mpc.qpos_slice].set(qpos)
            .at[mpc.qvel_slice].set(qvel)
            .at[mpc.foot_slice].set(foot)
        )
        return mpc.run(mpc_data, x0, command, contact,
                       base_quat_ref, use_base_quat_ref,
                       foot_ref_anchor, use_foot_ref_anchor)
    return solve_mpc
#endregion------------------------------------------------

def main(
    headless=False,
    steps=500,
    scene="flat",
    robot="go2",
    collect=False,
    collect_out=None,
    episode_duration_s=None,
):
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../../data/{robot}/scene_{scene}.xml"
    )

    # robot configuration
    config = robot_config(robot)

    #print_contact_friction(model, config.contact_frame, "floor")
    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    mpc = MPCWrapper(config, limited_memory=True)
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    #region dataset collection hooks------------------------------------
    collect_hooks = setup_sim_collection(
        collect,
        gait_type=GaitType.BALANCE,
        scene=scene,
        sim_hz=sim_frequency,
        robot=robot,
        episode_duration_s=episode_duration_s,
        collect_out=collect_out,
        name_prefix="4balance",
        cfg=dataset_collection_config,
    )
    #endregion------------------------------------------------

    # region Spawner configuration---------------------------
    spawner = RobotMapSpawner.from_config(
        cfg=spawn_config,
        foot_geom_names=config.contact_frame,
        check_collisions=True,     # set True for rough/stairs/ramp
    )
    RESPAWN_KEYCODES = spawn_config.respawn_keycodes
    #endregion------------------------------------------------

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    # Base force perturbation configuration---------------------------------------
    base_force_pert = RandomBaseForcePerturbation.from_config( 
        sim_dt=1.0 / sim_frequency,
        cfg=ext_base_force_config,
    )
    #------------------------------------------------

    # region desired pose sampler configuration---------------------------------------
    desired_pose_sampler = DesiredPoseSampler.from_config(config.robot_height, desired_pose_config)
    desired_height = float(config.robot_height)
    foot_anchor = np.zeros(3 * config.n_contact, dtype=np.float64)
    desired_quat   = np.asarray(config.quat0, dtype=np.float64)
    #endregion
    #------------------------------------------------

    # region respawn helper -------------------------------------
    def _respawn(*, manual: bool = False, crashed: bool = False):
        nonlocal mpc_data, tau, q_ref, counter, desired_height, desired_quat

        collect_hooks.on_respawn(manual=manual, crashed=crashed)

        # 1. Place robot at a random XY/yaw position on the map
        spawner.apply_to_data(model, data, config.p0, config.quat0, config.q0)

        # 2. Sample a DELTA pose (roll/pitch/yaw relative to spawn orientation)
        desired_height, delta_quat = desired_pose_sampler.sample()

        # 3. Compose: desired_quat = spawn_quat * delta_quat
        #    This keeps the desired orientation within sampler bounds of the spawn yaw,
        #    avoiding large independent jumps between spawn and desired orientation.
        spawn_quat   = np.asarray(data.qpos[3:7], dtype=np.float64)
        desired_quat = quat_normalize_wxyz(
            quat_mul_wxyz(spawn_quat, delta_quat)
        )

        # 4. Reset MPC warm-start from the new spawn state
        foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
        foot_anchor = np.asarray(foot, dtype=np.float64)   # ← lock feet to spawn positions
        mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)
        tau      = jnp.zeros(config.n_joints)
        q_ref    = config.q0.copy()
        counter  = 0

        # 5. Reset auxiliary systems
        base_force_pert.reset()

        print(
            f"[respawn] height={desired_height:.3f} m  "
            f"spawn_yaw={np.rad2deg(yaw_from_quat(spawn_quat)):.1f}°  "
            f"desired_yaw={np.rad2deg(yaw_from_quat(desired_quat)):.1f}°",
            flush=True,
        )
    # endregion
    #------------------------------------------------

    # region robot crash reset-----------------------
    CRASH_HEIGHT_THRESHOLD = config.robot_height * 0.5  # [m]
    CRASH_TILT_DEG = 60.0                               # [deg]
    def _is_crashed() -> bool:
        # Height check
        if float(data.qpos[2]) < CRASH_HEIGHT_THRESHOLD:
            return True
        # Tilt check — qpos[3:7] is [w, x, y, z] in MuJoCo
        w, x, y, z = (float(data.qpos[i]) for i in range(3, 7))
        roll  = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2.0 * (w*y - z*x), -1.0, 1.0))
        if abs(roll) > np.deg2rad(CRASH_TILT_DEG):
            return True
        if abs(pitch) > np.deg2rad(CRASH_TILT_DEG):
            return True
        return False
    # endregion
    #------------------------------------------------

    # region desired-pose tracking (dataset episode boundary)-----------
    # A collected episode spans exactly the stretch where the robot holds the
    # sampled balance pose, so widen these tolerances to accept looser tracking.
    POSE_HEIGHT_TOL = 0.03      # [m]
    POSE_ORIENT_TOL_DEG = 8.0   # [deg]

    def _holds_desired_pose() -> bool:
        if abs(float(data.qpos[2]) - desired_height) > POSE_HEIGHT_TOL:
            return False
        quat_now = quat_normalize_wxyz(np.asarray(data.qpos[3:7], dtype=np.float64))
        quat_ref = quat_normalize_wxyz(np.asarray(desired_quat, dtype=np.float64))
        # Geodesic angle between the two orientations; sign of the dot is irrelevant
        # because q and -q describe the same rotation.
        cos_half = min(1.0, abs(float(np.dot(quat_now, quat_ref))))
        orient_err = 2.0 * np.arccos(cos_half)
        return orient_err <= np.deg2rad(POSE_ORIENT_TOL_DEG)
    # endregion
    #------------------------------------------------

    
    _respawn()
    warm_command = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, desired_height])
    warm_contact = jnp.asarray(config.balance_fixed_contact_mask, dtype=jnp.float32)
    mpc_data, tau = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        foot,
        warm_command,
        warm_contact,
        jnp.asarray(desired_quat,  dtype=jnp.float32),   # base_quat_ref
        jnp.array(True),                                   # use_base_quat_ref
        jnp.asarray(foot_anchor,   dtype=jnp.float32),    # foot_ref_anchor  ← add
        jnp.array(True),                                   # use_foot_ref_anchor  ← add
    )
    tau.block_until_ready()
    _respawn()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot)
    collect_hooks.on_ready()

    period = int(sim_frequency / config.mpc_frequency)
    print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    tau = jnp.zeros(config.n_joints)
    q_ref = config.q0.copy()

    def step_controller():
        nonlocal counter, tau, q_ref, mpc_data, desired_height, desired_quat # nonlocal variables are used to modify the variables in the outer scope

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        
        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
           
            command = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,desired_height])
            contact = jnp.asarray(estimate_contacts(data, contact_ids))
            # print(f"Contact: {contact}")
            # print(foot)
            # print(f"Command: {command}")
            
            start = timer()
            mpc_data, tau = solve_mpc(
                mpc_data, qpos, qvel, foot, command,
                jnp.asarray(config.balance_fixed_contact_mask, dtype=jnp.float32),
                jnp.asarray(desired_quat,  dtype=jnp.float32),
                jnp.array(True),
                jnp.asarray(foot_anchor,   dtype=jnp.float32),  # ← fixed anchor
                jnp.array(True),                                  
            )
            tau.block_until_ready()
            stop = timer()

            # tau = jnp.clip(tau, config.min_torque, config.max_torque)
            # The shifted warm start is the next joint target used by the PD stabilizer.
            q_ref = mpc_data.X0[0, 7 : 7 + config.n_joints]
            #print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        data.ctrl = np.asarray(tau)

        base_force_pert.tick_and_apply(data) # apply random base force perturbation
 
        mujoco.mj_step(model, data)
        counter += 1

        # Record only while the robot holds the sampled pose; losing it closes the
        # episode, so each stored episode is one continuous "on target" stretch.
        collect_hooks.set_recording(_holds_desired_pose(), reason="pose_lost")
        collect_hooks.after_physics_step(
            model, data, np.asarray(tau), contact_ids, base_force_pert, config.n_joints,
        )

        if _is_crashed():
            _respawn(crashed=True)
         

    if headless:
        for _ in range(steps):
            step_controller()
        collect_hooks.finish("")
        return

    def key_callback(key: int):
        if key in RESPAWN_KEYCODES:
            _respawn(manual=True)
        else:
            pass

    _spawn_region_visual = None
    _force_geom_id = -1

    _desired_x_visual = -1   # forward (red)
    _desired_y_visual = -1   # left    (green)
    _desired_z_visual = -1   # up      (blue)

     
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        viewer.sync()
        sim_utils.setup_tracking_camera(viewer, model, body_name="base")
        while viewer.is_running():
            tic = timer()
            
            # region render spawn region-----------------------
            if spawn_config.show_spawn_region:
                _spawn_region_visual = spawner.render_spawn_region(
                    viewer,
                    z=spawn_config.region_z,
                    visual=_spawn_region_visual,
                )
            # endregion
            #---------------------------------------------------

            # region render external force ------------------
            base_pos = np.asarray(data.qpos[:3], dtype=np.float64)
            if base_force_pert.is_active:
                force_vec   = base_force_pert.force
                force_color = np.array([1.0, 0.15, 0.15, 0.85])
                force_scale = float(np.linalg.norm(force_vec)) * 0.005
            else:
                force_vec   = np.array([0.0, 0.0, 1e-3])  # non-zero dummy
                force_color = np.array([0.0, 0.0, 0.0, 0.0])  # invisible
                force_scale = 1e-3
            _force_geom_id = sim_utils.render_vector(
                viewer,
                vector=force_vec,
                pos=base_pos + np.array([0.0, 0.0, 0.25]),
                scale=force_scale,
                color=force_color,
                geom_id=_force_geom_id,
            )
            # endregion -------------------------------------
            #---------------------------------------------------

            # region render desired balance pose (coordinate frame) -----------
            _frame_pos = np.array([data.qpos[0], data.qpos[1], desired_height + 0.3])
            _dx, _dy, _dz = _quat_to_axes(np.asarray(desired_quat))
            _desired_x_visual = sim_utils.render_vector(
                viewer, _dx, _frame_pos, scale=0.3,
                color=np.array([1.0, 0.15, 0.15, 0.9]),   # red  → forward / X
                geom_id=_desired_x_visual,
            )
            _desired_y_visual = sim_utils.render_vector(
                viewer, _dy, _frame_pos, scale=0.3,
                color=np.array([0.15, 0.9, 0.15, 0.9]),   # green → left / Y
                geom_id=_desired_y_visual,
            )
            _desired_z_visual = sim_utils.render_vector(
                viewer, _dz, _frame_pos, scale=0.3,
                color=np.array([0.15, 0.15, 1.0, 0.9]),   # blue  → up / Z
                geom_id=_desired_z_visual,
            )
            # endregion -------------------------------------------------------
            #---------------------------------------------------

            step_controller()
            toc = timer()
            if toc - tic < model.opt.timestep:
                sleep_time = model.opt.timestep - (toc - tic)
                time.sleep(sleep_time)
            viewer.sync()

    collect_hooks.finish("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, choices=["flat", "rough", "perlin","stairs","ramp", "slippery"], default="rough")
    parser.add_argument("--robot", type=str, choices=["aliengo", "mini_cheetah", "go2", "hyqreal"], default="go2")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect proprioceptive dataset into contact-state buckets.",
    )
    parser.add_argument(
        "--collect-out",
        type=str,
        default=None,
        help="Optional .npz filename (relative paths go inside the auto-created run folder).",
    )
    parser.add_argument(
        "--episode-duration",
        type=float,
        default=None,
        help=(
            "Episode length [s] for --collect "
            f"(default: {dataset_collection_config.episode.episode_duration_s} from config)."
        ),
    )
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
        robot=args.robot,
        collect=args.collect,
        collect_out=args.collect_out,
        episode_duration_s=args.episode_duration,
    )
