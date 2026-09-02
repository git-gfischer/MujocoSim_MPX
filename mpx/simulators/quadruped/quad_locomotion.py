# Usage: python quad_locomotion.py --headless --steps 2000 --scene flat --robot go2 --nav random --n-env 8 
# dataset_collection: python quad_locomotion.py --collect --scene flat --robot go2 --nav random
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


#from gym_quadruped.quadruped_env import QuadrupedEnv
#from gym_quadruped.utils.mujoco.visual import render_vector

from mpx.utils.quad_utils_locomotion.mpc_wrapper_locomotion import MPCWrapper
#from mpx.utils.quad_utils_locomotion.mpc_wrapper_locomotion import LocomotionMPCControllerWrapper



from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config, ExtBaseForceConfig
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation

from mpx.config.sim_config.config_quad_spawn import spawn_config, SpawnConfig
from mpx.utils.spawner.spawner import RobotMapSpawner

from mpx.utils.simulation_utils.console import KeyboardVelocityCommand
import mpx.utils.simulation_utils.sim_utils as sim_utils

from mpx.navigation.pointNav import PointNavigator

from mpx.utils.simulation_utils.live_plotter import ProprioceptivePlotter
from mpx.utils.dataset_collection.episode_recorder import setup_sim_collection
from mpx.utils.dataset_collection.dataset_bucket_system import GaitType
from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config

from mpx.estimators.quad_contact_estimation import estimate_contacts, estimate_foot_grf

# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

#region ================Helper functions================
def robot_config(robot):
    if(robot == "go2"):
        from mpx.config.robot_config.config_go2 import go2_config, Go2Mode, BalanceStance
        GO2_CONTROLLER_MODE = Go2Mode.LOCOMOTION
        config = go2_config(GO2_CONTROLLER_MODE)
    elif(robot == "aliengo"):
        import mpx.config.robot_config.config_aliengo as config
    return config
#------------------------------------------------
def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact):
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(qpos)
            .at[mpc.qvel_slice].set(qvel)
            .at[mpc.foot_slice].set(foot)
        )
        return mpc.run(mpc_data, x0, command, contact)

    return solve_mpc
#endregion------------------------------------------------

def main(
    headless=False,
    steps=500,
    scene="flat",
    robot="go2",
    nav="vel",
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

    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    mpc = MPCWrapper(config, limited_memory=True)
    command_handle = KeyboardVelocityCommand()
    # Navigation mode: "vel" = keyboard velocity, "random" = auto random goals,
    # "pointuser" = user-pointed goal (double-click ground + G in the viewer).
    use_navigation = nav in ("random", "pointuser")
    navigator = PointNavigator(
        robot_height=config.robot_height,
        auto_resample=(nav == "random"),
    )
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    plotter = ProprioceptivePlotter(window_size=200) if plot and not collect else None
    collect_hooks = setup_sim_collection(
        collect,
        gait_type=GaitType.TROT,
        scene=scene,
        sim_hz=sim_frequency,
        robot=robot,
        episode_duration_s=episode_duration_s,
        collect_out=collect_out,
        cfg=dataset_collection_config,
    )

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

    # region reset helper -------------------------------------
    def _respawn(*, manual: bool = False, crashed: bool = False):
        nonlocal mpc_data, tau, q_ref, counter
        collect_hooks.on_respawn(manual=manual, crashed=crashed)
        spawner.apply_to_data(model, data, config.p0, config.quat0, config.q0)
        foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
        mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)
        tau = jnp.zeros(config.n_joints)
        q_ref = config.q0.copy()
        counter = 0
        base_force_pert.reset()
        command_handle.reset()
        if nav == "random":
            navigator.reset(np.asarray(data.qpos))
        print("[respawn] new random yaw spawn", flush=True)
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
    warm_command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    warm_contact = jnp.asarray(estimate_contacts(data, contact_ids))
    mpc_data, tau = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        foot,
        warm_command,
        warm_contact,
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
        nonlocal counter, tau, q_ref, mpc_data # nonlocal variables are used to modify the variables in the outer scope

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        
        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
           
            if use_navigation:
                command = jnp.asarray(navigator.mpc_input(qpos, config.robot_height))
            else:
                command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            contact = jnp.asarray(estimate_contacts(data, contact_ids))
            if not collect_hooks.enabled:
                print(f"Contact: {contact}")
                print(foot)
                print(f"Command: {command}")
            
            start = timer()
            mpc_data, tau = solve_mpc(
                mpc_data,
                qpos,
                qvel,
                foot,
                command,
                contact*0.0,
            )
            tau.block_until_ready()
            stop = timer()

            # tau = jnp.clip(tau, config.min_torque, config.max_torque)
            # The shifted warm start is the next joint target used by the PD stabilizer.
            q_ref = mpc_data.X0[0, 7 : 7 + config.n_joints]
            if not collect_hooks.enabled:
                print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        data.ctrl = np.asarray(tau)

        base_force_pert.tick_and_apply(data) # apply random base force perturbation
 
        mujoco.mj_step(model, data)
        counter += 1

        collect_hooks.after_physics_step(
            model, data, np.asarray(tau), contact_ids, base_force_pert, config.n_joints,
        )

        # One episode = one traverse to the goal. Checked here (not in the render
        # loop) so headless collection runs get the same episode boundaries.
        if use_navigation and navigator.reached(data.qpos):
            collect_hooks.end_episode(reason="goal_reached")
            if navigator.auto_resample:
                navigator.sample_goal(np.asarray(data.qpos))

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
        elif use_navigation and navigator.handle_key(key):
            pass
        else:
            command_handle.key_callback(key)

    _spawn_region_visual = None
    _force_geom_id = -1 

    if plotter is not None:
        plotter.start()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
            sim_utils.setup_tracking_camera(viewer, model, body_name="base")
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

            # Update / render the navigation goal (handles user-pointed goals
            # via the camera lookat and auto-resamples random goals when reached).
            if use_navigation:
                navigator.update(np.asarray(data.qpos), viewer)

            step_controller()

            # Stream all proprioception signals; only toggled-on plots render.
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
    parser.add_argument(
        "--nav",
        type=str,
        choices=["random", "pointuser", "vel"],
        default="vel",
        help="Navigation mode: random goals, user-pointed goal, or keyboard velocity.",
    )
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
        help="Output .npz path for --collect (default: dataset_loco_<robot>_<scene>.npz).",
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
        nav=args.nav,
        plot=args.plot,
        collect=args.collect,
        collect_out=args.collect_out,
        episode_duration_s=args.episode_duration,
    )
