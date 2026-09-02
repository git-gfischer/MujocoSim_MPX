
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
from mpx.utils.quad_utils_balance.mpc_wrapper_3balance import MPCWrapper

from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config, ExtBaseForceConfig
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation

from mpx.config.sim_config.config_quad_spawn import spawn_config, SpawnConfig
from mpx.utils.spawner.spawner import RobotMapSpawner
from mpx.utils.quad_utils_balance.desired_pose_sampler import (
    DesiredPoseSampler, DesiredPoseConfig,
)
desired_pose_config = DesiredPoseConfig(enabled=False)

from mpx.utils.quad_utils_balance.foot_reference import (
    FootReferenceManager,
    RandomSwingFootSampler,
    swing_foot_anchor_from_target,
    foot_target_foot_local_to_world,
)
from mpx.config.sim_config.config_foot_ref_config import foot_ref_config, random_swing_foot_config
import glfw

from mpx.utils.simulation_utils.console import SwingFootCommand
import mpx.utils.simulation_utils.sim_utils as sim_utils
from mpx.utils.math_utils.quad_math import yaw_from_quat, _quat_to_axes, quat_normalize_wxyz, quat_mul_wxyz

from mpx.estimators.quad_contact_estimation import estimate_contacts, estimate_foot_grf

from mpx.utils.simulation_utils.live_plotter import ProprioceptivePlotter

from mpx.utils.dataset_collection.episode_recorder import setup_sim_collection
from mpx.utils.dataset_collection.dataset_bucket_system import GaitType
from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config

from timeit import default_timer as timer
# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

#region ================Helper functions================
def robot_config(robot):
    if(robot == "go2"):
        GO2_CONTROLLER_MODE = Go2Mode.BALANCE
        config = go2_config(GO2_CONTROLLER_MODE,balance_stance=BalanceStance.TRIPOD_SWING_FL)
    elif(robot == "aliengo"):
        import mpx.config.robot_config.config_aliengo as config
    return config
#------------------------------------------------
def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact,
                  base_quat_ref, use_base_quat_ref,
                  foot_ref_anchor, use_foot_ref_anchor):
        x0 = (
            mpc.initial_state
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
    plot=False,
    collect=False,
    collect_out=None,
    episode_duration_s=None,
):
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../../data/{robot}/scene_{scene}.xml"
    )

    # robot configuration
    config = robot_config(robot)

    # Swing leg: the contact mask entry that is 0 (the nominal swing leg)
    swing_leg_idx = int(np.where(np.array(config.balance_fixed_contact_mask) < 0.5)[0][0])

    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    mpc = MPCWrapper(config, limited_memory=True)
    swing_foot_cmd = SwingFootCommand(swing_leg_idx=swing_leg_idx, step=0.025)
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)
    counter = 0

    # Live proprioception plotter (toggle signals via the "Signal Selector" window).
    plotter = ProprioceptivePlotter(window_size=200) if plot and not collect else None
    collect_hooks = setup_sim_collection(
        collect,
        gait_type=GaitType.BALANCE,
        scene=scene,
        sim_hz=sim_frequency,
        robot=robot,
        episode_duration_s=episode_duration_s,
        collect_out=collect_out,
        name_prefix="balance",
        cfg=dataset_collection_config,
    )

    # region print controls---------------------------------------
    print(
        "\n"
        "========== quad_3balance controls ==========\n"
        "  B / Backspace : respawn (random pose)\n"
        "  I / K         : swing foot forward / backward\n"
        "  J / L         : swing foot left / right\n"
        "  U / O         : swing foot up / down\n"
        "  R             : randomise swing foot NOW (within bounds)\n"
        "  N             : toggle random-on-respawn mode ON/OFF\n"
        f"  swing leg     : {swing_leg_idx} "
        f"({['FL','FR','RL','RR'][swing_leg_idx]})\n"
        f"  scene         : {scene}   robot : {robot}\n"
        "============================================\n",
        flush=True,
    )
    #endregion
    #------------------------------------------------

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
    desired_quat   = np.asarray(config.quat0, dtype=np.float64)

    #endregion
    #------------------------------------------------


    # region foot reference configuration---------------------------------------
    foot_ref_mgr = FootReferenceManager(foot_ref_config)
    # Swing leg: the one with mask=0 in the config
    swing_leg_idx = int(np.where(np.array(config.balance_fixed_contact_mask) < 0.5)[0][0])
    # World-frame foot anchor: flat (12,) [FL, FR, RL, RR] × XYZ
    foot_anchor = np.zeros(3 * config.n_contact, dtype=np.float64)  # filled in _respawn
    SWING_STEP = 0.025   # metres per key press
    SWING_INIT_OFFSET = np.array([0.25, -0.1, -0.3])   # forward / inward / down in body frame

    # Random swing-foot sampler — R: resample now, N: toggle auto-resample on respawn.
    # Edit random_swing_foot_config in config_foot_ref_config.py to change bounds / enable at start.
    random_swing_sampler = RandomSwingFootSampler(random_swing_foot_config)
    print(
        f"[random_swing] mode={'ON' if random_swing_sampler.enabled else 'OFF'}  "
        f"bounds: {random_swing_sampler.bounds_summary()}",
        flush=True,
    )
    #endregion
    #------------------------------------------------


    # region respawn helper -------------------------------------
    def _respawn(*, manual: bool = False, crashed: bool = False):
        nonlocal mpc_data, tau, q_ref, counter, desired_height, desired_quat, foot_anchor

        collect_hooks.on_respawn(manual=manual, crashed=crashed)

        # 1. Place robot at a random XY/yaw position on the map
        spawner.apply_to_data(model, data, config.p0, config.quat0, config.q0)

        # 2. Sample desired body pose (height + orientation delta relative to spawn)
        desired_height, delta_quat = desired_pose_sampler.sample()
        spawn_quat   = np.asarray(data.qpos[3:7], dtype=np.float64)
        desired_quat = quat_normalize_wxyz(quat_mul_wxyz(spawn_quat, delta_quat))

        # 3. Sample nominal world-frame foot anchor for all 4 legs
        foot_anchor = np.asarray(
            foot_ref_mgr.tripod_foot_reference_world(
                key=jax.random.PRNGKey(int(time.time() * 1000) & 0x7FFFFFFF),
                p=jnp.asarray(data.qpos[:3]),
                quat=jnp.asarray(data.qpos[3:7]),
                foot0=jnp.asarray(config.p_legs0),
                n_contact=config.n_contact,
                sigma=np.array([0.04, 0.04, 0.0]),   # ±4 cm XY, fixed Z
            ),
            dtype=np.float64,
        )

        # 4. Place swing foot — random target if enabled, otherwise fixed offset
        measured_swing_world = sim_utils.geom_positions(data, contact_ids)[
            3 * swing_leg_idx : 3 * swing_leg_idx + 3
        ]
        if random_swing_sampler.resample_on_respawn:
            new_swing_world = random_swing_sampler.sample_swing_world(
                data.qpos[:3], data.qpos[3:7]
            )
        else:
            new_swing_world = foot_target_foot_local_to_world(
                measured_swing_world, data.qpos[3:7], SWING_INIT_OFFSET
            )
        foot_anchor = swing_foot_anchor_from_target(foot_anchor, swing_leg_idx, new_swing_world)

        # 5. Reset MPC warm-start from the new spawn state
        foot     = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
        mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)
        tau      = jnp.zeros(config.n_joints)
        q_ref    = config.q0.copy()
        counter  = 0

        # 6. Reset auxiliary systems
        base_force_pert.reset()
        swing_foot_cmd.reset()
        random_swing_sampler.reset_arrival_state()

        swing_target = foot_anchor[3*swing_leg_idx : 3*swing_leg_idx+3]
        print(
            f"[respawn] height={desired_height:.3f}m  "
            f"spawn_yaw={np.rad2deg(yaw_from_quat(spawn_quat)):.1f}°  "
            f"swing_leg={swing_leg_idx}  "
            f"swing_target=[{swing_target[0]:.2f}, {swing_target[1]:.2f}, {swing_target[2]:.2f}]",
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
        nonlocal counter, tau, q_ref, mpc_data, desired_height, desired_quat, foot_anchor
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if counter % period == 0:
            foot = sim_utils.geom_positions(data, contact_ids)

            if random_swing_sampler.resample_on_arrival:
                measured_swing = foot[3 * swing_leg_idx : 3 * swing_leg_idx + 3]
                foot_anchor, _, _, did_resample = random_swing_sampler.try_resample_on_arrival(
                    foot_anchor,
                    swing_leg_idx,
                    measured_swing,
                    data.qpos[:3],
                    data.qpos[3:7],
                    sim_dt=model.opt.timestep,
                )
                if did_resample:
                    sw = foot_anchor[3 * swing_leg_idx : 3 * swing_leg_idx + 3]
                    print(
                        f"[random_swing] arrival → new target "
                        f"[{sw[0]:.3f}, {sw[1]:.3f}, {sw[2]:.3f}]",
                        flush=True,
                    )

            foot    = jnp.asarray(foot)
            command = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, desired_height])
            mpc_data, tau = solve_mpc(
                mpc_data, qpos, qvel, foot, command,
                jnp.asarray(config.balance_fixed_contact_mask, dtype=jnp.float32),
                jnp.asarray(desired_quat,  dtype=jnp.float32),
                jnp.array(True),                                # use_base_quat_ref
                jnp.asarray(foot_anchor,   dtype=jnp.float32), # foot_ref_anchor
                jnp.array(True),                                # use_foot_ref_anchor
            )
            tau.block_until_ready()
            q_ref = mpc_data.X0[0, 7 : 7 + config.n_joints]
        data.ctrl = np.asarray(tau)
        base_force_pert.tick_and_apply(data)
        mujoco.mj_step(model, data)
        counter += 1

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
        nonlocal foot_anchor
        if key in RESPAWN_KEYCODES:
            _respawn(manual=True)
        elif key == glfw.KEY_R:
            # Manually draw a new random swing target in base-frame bounds.
            new_target = random_swing_sampler.sample_swing_world(
                data.qpos[:3], data.qpos[3:7]
            )
            foot_anchor = swing_foot_anchor_from_target(foot_anchor, swing_leg_idx, new_target)
            random_swing_sampler.reset_arrival_state()
            sw = foot_anchor[3*swing_leg_idx : 3*swing_leg_idx+3]
            print(
                f"[random_swing] resampled → [{sw[0]:.3f}, {sw[1]:.3f}, {sw[2]:.3f}]",
                flush=True,
            )
        elif key == glfw.KEY_N:
            # Toggle auto-randomise-on-respawn.
            state = random_swing_sampler.toggle()
            print(
                f"[random_swing] random-on-respawn {'ON ' if state else 'OFF'}  "
                f"({random_swing_sampler.bounds_summary()})",
                flush=True,
            )
        else:
            foot_anchor = swing_foot_cmd.key_callback(key, foot_anchor, data.qpos[3:7])

    # region render initializations---------------------------------------
    _spawn_region_visual = None
    _force_geom_id = -1
    #endregion
    #------------------------------------------------

    if plotter is not None:
        plotter.start()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        viewer.sync()
        sim_utils.setup_tracking_camera(viewer, model, body_name="base")

        # Allocate foot markers now that viewer exists# Allocate foot markers now that viewer exists
        _desired_foot_markers   = foot_ref_mgr.attach_desired_foot_markers(viewer, n_contact=config.n_contact)
        _swing_goal_marker      = foot_ref_mgr.attach_swing_foot_goal_marker(viewer)
        _swing_workspace_marker = foot_ref_mgr.attach_swing_workspace_marker(viewer)
        _swing_bounds_box       = random_swing_sampler.attach_bounds_box_marker(viewer)
        
        while viewer.is_running():
            overlay_text = swing_foot_cmd.consume_overlay_text()
            tic = timer()
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
            
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

            # region Desired foot positions (swing only)
            if _desired_foot_markers is not None:
                _desired_foot_markers.draw(
                    viewer, foot_anchor,
                    contact_mask=np.array(config.balance_fixed_contact_mask),
                    swing_only=True, sync=False,
                )
            # Fixed swing goal sphere
            if _swing_goal_marker is not None:
                _swing_goal_marker.set_goal(swing_leg_idx, foot_anchor[3*swing_leg_idx:3*swing_leg_idx+3])
                _swing_goal_marker.draw(viewer, sync=False)
            # Reachable workspace ellipsoid
            if _swing_workspace_marker is not None:
                _swing_workspace_marker.set_center(swing_leg_idx, foot_anchor[3*swing_leg_idx:3*swing_leg_idx+3])
                _swing_workspace_marker.draw(viewer, sync=False)
            # Random swing sampling region (base-frame XYZ bounds box)
            if _swing_bounds_box is not None:
                random_swing_sampler.update_bounds_box_marker(
                    _swing_bounds_box,
                    viewer,
                    data.qpos[:3],
                    data.qpos[3:7],
                    sync=False,
                )
            #endregion
            #---------------------------------------------------

            step_controller()

            if plotter is not None:
                n = config.n_joints
                foot_xyz = sim_utils.geom_positions(data, contact_ids, flatten=False)
                plotter.update(
                    torque=np.asarray(tau),
                    joint_pos=np.asarray(data.qpos[7 : 7 + n]),
                    joint_vel=np.asarray(data.qvel[6 : 6 + n]),
                    contacts=estimate_contacts(
                        data, contact_ids, foot_positions=foot_xyz,
                    ),
                    contact_nominal=np.asarray(
                        config.balance_fixed_contact_mask, dtype=np.float32,
                    ),
                    grf=estimate_foot_grf(model, data, contact_ids),
                    ang_vel=np.asarray(data.qvel[3:6]),
                    lin_acc=np.asarray(data.qacc[:3]),
                )

            toc = timer()
            if toc - tic < model.opt.timestep:
                sleep_time = model.opt.timestep - (toc - tic)
                time.sleep(sleep_time)
            viewer.sync()

    if plotter is not None:
        plotter.stop()

    collect_hooks.finish("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, choices=["flat", "rough", "perlin","stairs","ramp", "slippery"], default="flat")
    parser.add_argument("--robot", type=str, choices=["aliengo", "mini_cheetah", "go2", "hyqreal"], default="go2")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open the live proprioception plotter with a signal-toggle window.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect proprioceptive dataset into contact-state buckets.",
    )
    parser.add_argument(
        "--collect-out",
        type=str,
        default=None,
        help="Output .npz path for --collect (default: dataset_balance_<robot>_<scene>.npz).",
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
        plot=args.plot,
        collect=args.collect,
        collect_out=args.collect_out,
        episode_duration_s=args.episode_duration,
    )
