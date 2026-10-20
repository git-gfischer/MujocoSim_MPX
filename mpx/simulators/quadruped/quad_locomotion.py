# Usage: python quad_locomotion.py --headless --steps 2000 --scene flat --robot go2 --nav random --n-env 8 --gait trot
# dataset_collection: python quad_locomotion.py --collect --scene flat --robot go2 --nav random
# pickle collection: python quad_locomotion.py --collect pickle --scene flat --robot go2 --nav random
# Note for dataset collection: python -m mpx.utils.dataset_collection.make_signal_bounds --robot go2
# dataset_collection: python quad_locomotion.py --collect --scene flat --robot go2 --nav random
# pickle collection: python quad_locomotion.py --collect pickle --scene flat --robot go2 --nav random
# Note for dataset collection: python -m mpx.utils.dataset_collection.make_signal_bounds --robot go2

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

from mpx.utils.quad_utils_locomotion.mpc_wrapper_inverse import make_locomotion_mpc
from mpx.utils.simulation_utils.measured_kinematics import (
    apply_rough_terrain_mpc,
    build_terrain_height_grid,
    terrain_overlay_grid,
)
from mpx.utils.simulation_utils.pyramid_terrain import PyramidTerrain, opposite_goal
from mpx.utils.simulation_utils.ramp_pyramid import RampPyramid
#from mpx.utils.quad_utils_locomotion.mpc_wrapper_locomotion import LocomotionMPCControllerWrapper

from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config, ExtBaseForceConfig
from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation
from mpx.config.sim_config.config_base_weight import base_weight_config, BaseWeightConfig
from mpx.utils.simulation_utils.base_weight import BaseWeightForce
from mpx.utils.simulation_utils.motor_model import MotorModel
from mpx.config.sim_config.config_reset_randomization import loco_reset_randomization_config
from mpx.utils.simulation_utils.reset_randomizer import ResetRandomizer, ResetTargets

from mpx.config.sim_config.config_quad_spawn import spawn_config, SpawnConfig
from mpx.utils.spawner.spawner import RobotMapSpawner

from mpx.utils.simulation_utils.console import KeyboardVelocityCommand
import mpx.utils.simulation_utils.sim_utils as sim_utils

from mpx.navigation.extreme import ExtremeNavigator
from mpx.navigation.pointNav import PointNavigator

from mpx.utils.simulation_utils.live_plotter import ProprioceptivePlotter
from mpx.config.sim_config.config_live_plotter import live_plotter_config
from mpx.utils.simulation_utils.velocity_command import command_from_mpc_input
from mpx.utils.dataset_collection.dataset_pickle import (
    COLLECT_PARQUET,
    COLLECT_PICKLE,
    PickleSimCollector,
    add_collect_argument,
)
from mpx.utils.dataset_collection.episode_recorder import (
    ControlSample,
    read_episode_conditions,
    setup_sim_collection,
)
from mpx.utils.dataset_collection.dataset_bucket_system import (
    GaitType,
    gait_from_phase_offsets,
)
from mpx.config.sim_config.config_dataset_bucket import dataset_collection_config

from mpx.estimators.camurri_contact import camurri_from_sim, format_camurri
from mpx.estimators.quad_contact_estimation import estimate_contacts, estimate_foot_grf

# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

_ROBOT_DATA_DIR = {
    "go2": "go2",
    "go2_dls": "go2_dls",
    "aliengo": "aliengo",
    "spot": "boston_dynamics_spot",
    "b2": "b2",
}


#region ================Helper functions================
def robot_config(robot, gait=None, mpc_model="whole_body"):
    """Locomotion config for ``robot``, optionally on a named gait.

    ``gait=None`` keeps each robot's own default gait. Go2, Spot and B2 share
    the same gait registry layout (trot / pace / crawl / bound).
    """
    if robot in ("go2", "go2_dls"):
        from mpx.config.robot_config.config_go2 import go2_config, Go2Mode
        return go2_config(Go2Mode.LOCOMOTION, gait=gait, mpc_model=mpc_model, robot=robot)
    if mpc_model not in (None, "whole_body"):
        raise ValueError(
            f"--mpc-model {mpc_model!r} is only supported for the go2 "
            f"(got robot {robot!r})."
        )
    if robot == "spot":
        from mpx.config.robot_config.config_spot import spot_config, SpotMode
        return spot_config(SpotMode.LOCOMOTION, gait=gait)
    if robot == "b2":
        from mpx.config.robot_config.config_b2 import b2_config, B2Mode
        return b2_config(B2Mode.LOCOMOTION, gait=gait)
    if gait is not None:
        raise ValueError(
            f"--gait is not supported for robot {robot!r}; "
            "expected one of go2, spot, b2."
        )
    if robot == "aliengo":
        import mpx.config.robot_config.config_aliengo as config
        return config
    raise ValueError(f"Unknown robot {robot!r}; expected one of {sorted(_ROBOT_DATA_DIR)}")


def _scene_xml_path(robot: str, scene: str) -> str:
    folder = _ROBOT_DATA_DIR.get(robot, robot)
    return os.path.abspath(os.path.join(dir_path, "..", "..", "data", folder, f"scene_{scene}.xml"))
#------------------------------------------------
def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact, terrain=None):
        x0 = mpc.pack_state(qpos, qvel, foot)
        return mpc.run(mpc_data, x0, command, contact, terrain=terrain)

    return solve_mpc
#endregion------------------------------------------------

def main(
    headless=False,
    steps=500,
    scene="flat",
    robot="go2_dls",
    nav="vel",
    collect=None,
    collect_out=None,
    episode_duration_s=None,
    gait=None,
    mpc_model="whole_body",
    terrain_aware=True,
):
    collect_parquet = collect == COLLECT_PARQUET
    collect_pickle = collect == COLLECT_PICKLE
    collecting = collect_parquet or collect_pickle
    model = mujoco.MjModel.from_xml_path(_scene_xml_path(robot, scene))
    # Place staired-pyramid layers and the ramp stamp before the height grid
    # is raycast. Other scenes are a no-op and keep the XML as loaded.
    ramp_pyramid = RampPyramid.from_scene(scene, model)
    if ramp_pyramid.active:
        ramp_pyramid.apply(model)
    pyramid_terrain = PyramidTerrain.from_scene(scene, model)
    pyramid_sample = pyramid_terrain.apply(model)

    # robot configuration
    config = robot_config(robot, gait=gait, mpc_model=mpc_model)
    if hasattr(config, "gait"):
        print(f"[gait] {config.gait.summary()}", flush=True)

    data = mujoco.MjData(model)
    # 500 Hz by default: the contact solver's 10 ms time constant then spans 5
    # substeps, which is what stops the one-frame contact dropouts. Control and
    # logging stay at 50 Hz; the labels are reduced from the substeps between.
    sim_frequency = float(dataset_collection_config.rates.sim_hz)
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    apply_rough_terrain_mpc(config, model, scene=scene, terrain_aware=terrain_aware)
    if terrain_aware and (ramp_pyramid.active or pyramid_terrain.kind == "staired_pyramid"):
        config.terrain_grid_dynamic = True
    mpc = make_locomotion_mpc(config, limited_memory=True)
    command_handle = KeyboardVelocityCommand()
    # Navigation mode: "vel" = keyboard velocity, "random" = auto random goals,
    # "pointuser" = user-pointed goal (double-click ground + G in the viewer).
    use_navigation = nav in ("random", "random_fb", "pointuser")
    navigator = PointNavigator(
        robot_height=config.robot_height,
        auto_resample=nav in ("random", "random_fb"),
        random_fb=(nav == "random_fb"),
        # The navigator's command is refreshed once per MPC tick, which is what
        # its yaw slew limit integrates against.
        control_dt=1.0 / config.mpc_frequency,
    )
    if ramp_pyramid.active and nav in ("random", "random_fb"):
        navigator.goal_sampler = ramp_pyramid.opposite_goal
    elif pyramid_terrain.kind == "staired_pyramid" and nav in ("random", "random_fb"):
        # Across the pyramid: random nearby goals mostly walked off the region.
        navigator.goal_sampler = lambda qpos: opposite_goal(
            qpos, pyramid_sample, pyramid_terrain.cfg
        )
    # --nav vel is the keyboard, including during --collect. Reverse and
    # near-limit commands come from --nav random_fb and --nav extreme.
    use_extreme = nav == "extreme"
    extreme = (
        ExtremeNavigator(
            robot, getattr(config, "gait_name", "trot"), dt=1.0 / config.mpc_frequency
        )
        if use_extreme
        else None
    )
    if extreme is not None:
        extreme.reset()
        limits = extreme.limits
        print(
            f"[command] --nav extreme ({robot} {extreme.gait}): slewing to "
            f"±{limits.vx_mps:.2f} m/s, ±{limits.vy_mps:.2f} m/s, "
            f"±{limits.yaw_rate_rps:.2f} rad/s at "
            f"{limits.linear_accel_mps2:.2f} m/s² and "
            f"{limits.yaw_accel_rps2:.2f} rad/s².",
            flush=True,
        )
    elif collecting and use_navigation:
        reverse_note = (
            "Each new goal is approached forward or backward at random."
            if nav == "random_fb"
            else (
                "Forward goals only. Commanded reverse is --nav random_fb; "
                "near-limit linear and yaw commands are --nav extreme."
            )
        )
        print(
            f"[command] --nav {nav} --collect: driving to navigation goals. "
            f"{reverse_note}",
            flush=True,
        )
    else:
        print(f"[command] driving from --nav {nav}", flush=True)

    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    plotter = (
        ProprioceptivePlotter.from_config(cfg=live_plotter_config)
        if live_plotter_config.enabled and not collecting
        else None
    )
    collect_hooks = setup_sim_collection(
        collect_parquet,
        # Read the gait off the controller's phase offsets rather than
        # hardcoding it: config_go2.timer_t decides what the robot walks,
        # and a hardcoded label put "trot" on a crawling robot.
        gait_type=gait_from_phase_offsets(config.timer_t),
        scene=scene,
        sim_hz=sim_frequency,
        robot=robot,
        episode_duration_s=episode_duration_s,
        collect_out=collect_out,
        nav=nav,
        reset_randomization=loco_reset_randomization_config.for_scene(
            scene, gait=getattr(config, "gait_name", None), robot=robot
        ).to_metadata(),
        cfg=dataset_collection_config,
    )

    # region Spawner configuration---------------------------
    spawner = RobotMapSpawner.from_config(
        cfg=spawn_config,
        foot_geom_names=config.contact_frame,
        check_collisions=True,     # set True for rough/stairs/ramp
        robot_root_body_name=getattr(
            config, "base_body_name", spawn_config.robot_root_body_name
        ),
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
    # Constant extra-mass load on the base (added after pulse write).
    base_weight = BaseWeightForce.from_config(cfg=base_weight_config)
    reset_randomizer = ResetRandomizer.from_config(
        loco_reset_randomization_config.for_scene(
            scene, gait=getattr(config, "gait_name", None), robot=robot
        )
    )
    #------------------------------------------------

    # region spawn settle -------------------------------------
    def _settle_after_spawn() -> tuple[int, float]:
        """Step the physics with a joint PD to ``q0`` until the robot is at rest.

        Foot vertical relief spawns the base at the lowest collision-free ``z``,
        which is up to ``foot_relief_max`` (0.10 m) above nominal on rough terrain.
        Handing the MPC its first command mid-fall is what made the robot slam
        into the ground at episode start, so the drop is absorbed here instead,
        before the MPC is initialised and before any recording begins.

        Returns ``(steps_taken, settled_height)`` for logging.
        """
        if not spawn_config.settle_after_spawn:
            return 0, float(data.qpos[2])

        n_steps = int(round(spawn_config.settle_duration_s * sim_frequency))
        if n_steps <= 0:
            return 0, float(data.qpos[2])

        n_j = config.n_joints
        q_hold = np.asarray(config.q0, dtype=np.float64).reshape(n_j)
        kp = float(spawn_config.settle_kp)
        kd = float(spawn_config.settle_kd)
        lo, hi = float(config.min_torque), float(config.max_torque)
        tol = float(spawn_config.settle_qvel_tol)

        # Perturbation and payload forces are written fresh every control step in
        # the main loop; clear whatever the previous episode left behind so the
        # settle is not fighting a stale load.
        data.qfrc_applied[:] = 0.0

        taken = 0
        for step in range(n_steps):
            q = np.asarray(data.qpos[7 : 7 + n_j], dtype=np.float64)
            dq = np.asarray(data.qvel[6 : 6 + n_j], dtype=np.float64)
            data.ctrl = np.clip(kp * (q_hold - q) - kd * dq, lo, hi)
            mujoco.mj_step(model, data)
            taken = step + 1
            # Stop as soon as it is at rest: a flat spawn settles ~1 cm and does
            # not need the full duration.
            if float(np.max(np.abs(dq))) < tol and abs(float(data.qvel[2])) < tol:
                break

        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        return taken, float(data.qpos[2])
    # endregion

    motor = MotorModel.from_config(dt=1.0 / sim_frequency, n_joints=config.n_joints)
    pickle_collector = None
    if collect_pickle:
        duration = (
            dataset_collection_config.episode.episode_duration_s
            if episode_duration_s is None
            else episode_duration_s
        )
        pickle_collector = PickleSimCollector(
            model, contact_ids, config.n_joints, sim_frequency, duration,
        )

    # region terrain map for the MPC -----------------------------
    # solve_mpc is jitted, so the map goes in as an argument: one held by the
    # wrapper is frozen into the first trace, and every respawned pyramid was
    # planned on the old steps. A fixed extent keeps the grid shape, so a new
    # pyramid does not recompile the MPC.
    terrain = None
    grid_extent = (
        float(pyramid_terrain.cfg.base_half.high) + 0.5
        if pyramid_terrain.kind == "staired_pyramid"
        else None
    )

    def _rebuild_terrain_grid():
        nonlocal terrain
        if not getattr(config, "terrain_grid_dynamic", False):
            return
        built = build_terrain_height_grid(
            model,
            resolution=float(config.terrain_height_resolution),
            extent=grid_extent,
        )
        if built is None:
            return
        grid, origin, res = built
        config.terrain_height_grid = grid
        config.terrain_height_origin = origin
        config.terrain_height_resolution = res
        terrain = (jnp.asarray(grid), jnp.asarray(origin))
    # endregion

    # region reset helper -------------------------------------
    def _respawn(*, manual: bool = False, crashed: bool = False):
        nonlocal mpc_data, tau, q_ref, counter, pyramid_sample
        collect_hooks.on_respawn(manual=manual, crashed=crashed)
        if pickle_collector is not None:
            pickle_collector.end_episode()
        if ramp_pyramid.active:
            sample = ramp_pyramid.apply(model)
            _rebuild_terrain_grid()
            if sample is not None and not collecting:
                print(
                    f"[ramp] slope {sample.slope_deg:.1f} deg, "
                    f"height {sample.height:.2f} m",
                    flush=True,
                )
            ramp_pyramid.try_spawn(
                spawner, model, data, config.quat0, config.q0,
                float(config.robot_height),
            )
        elif pyramid_terrain.kind == "staired_pyramid":
            pyramid_sample = pyramid_terrain.apply(model)
            _rebuild_terrain_grid()
            if pyramid_sample is not None and not collecting:
                print(
                    f"[staired_pyramid] base {pyramid_sample.base_half:.2f} m, "
                    f"height {pyramid_sample.height:.2f} m, "
                    f"steps {pyramid_sample.n_steps} "
                    f"(rise {pyramid_sample.rise:.2f} m, tread {pyramid_sample.tread:.2f} m)",
                    flush=True,
                )
            pyramid_terrain.try_spawn(
                spawner, model, data, config.quat0, config.q0,
                pyramid_sample, float(config.robot_height),
            )
        elif pyramid_sample is not None:
            pyramid_terrain.try_spawn(
                spawner, model, data, config.quat0, config.q0,
                pyramid_sample, float(config.robot_height),
            )
        else:
            spawner.apply_to_data(model, data, config.p0, config.quat0, config.q0)
        motor.reset()
        z_spawned = float(data.qpos[2])
        # Absorb the vertical-relief drop BEFORE the MPC is initialised, so it
        # starts from a pose the robot is actually holding rather than mid-fall.
        settle_steps, z_settled = _settle_after_spawn()
        if settle_steps and not collecting:
            print(
                f"[spawn] settled {z_spawned:.3f} -> {z_settled:.3f} m "
                f"in {settle_steps / sim_frequency:.2f} s",
                flush=True,
            )
        foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
        mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)
        tau = jnp.zeros(config.n_joints)
        q_ref = config.q0.copy()
        counter = 0
        base_force_pert.reset()
        command_handle.reset()
        if nav in ("random", "random_fb"):
            navigator.reset(np.asarray(data.qpos))
        if extreme is not None:
            extreme.reset()
        _randomize_episode(label="respawn")
    # endregion

    # region per-episode domain randomization -------------------------------
    episode_seed = 0

    def _randomize_episode(*, label: str = "episode"):
        """Resample every reset knob and stamp it on the episode about to start.

        Called at every episode boundary, not only at a respawn. v3 only
        randomized in ``_respawn``, and an episode that ended without a fall did
        not respawn — so one parameter set covered up to 12 consecutive episodes
        and near-duplicate episodes landed in different splits.
        """
        nonlocal mpc_data, episode_seed
        episode_seed += 1
        sample, mpc_data = reset_randomizer.sample_and_apply(
            ResetTargets(
                model=model,
                foot_geom_ids=contact_ids,
                base_weight=base_weight,
                navigator=navigator if use_navigation else None,
                mpc_data=mpc_data,
            )
        )
        meta = sample.to_metadata()
        if extreme is not None:
            meta["max_speed"] = float(extreme.limits.vx_mps)
            meta["max_yaw_rate"] = float(extreme.limits.yaw_rate_rps)
        collect_hooks.set_episode_conditions(
            randomization=meta,
            seed=episode_seed,
            mode="locomotion",
            **read_episode_conditions(model, contact_ids, base_weight),
        )
        if meta:
            bits = "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in meta.items()
            )
            print(f"[{label}] randomized {bits}", flush=True)
        else:
            print(f"[{label}] new random yaw spawn", flush=True)
    # endregion
    #------------------------------------------------

    # region robot crash reset-----------------------
    CRASH_HEIGHT_THRESHOLD = config.robot_height * 0.5   # [m] hard floor
    CRASH_TILT_DEG = 60.0                                # [deg]
    # A robot that folds onto its belly sits just ABOVE the hard floor, level,
    # and stays there. Detect it by a sustained low stance instead of an
    # instantaneous one: below this height for longer than the dwell means the
    # robot is down, not squatting.
    COLLAPSE_HEIGHT_THRESHOLD = config.robot_height * 0.65   # [m]
    COLLAPSE_DWELL_STEPS = int(0.5 * sim_frequency)          # 0.5 s
    collapse_counter = 0

    def _is_crashed() -> bool:
        nonlocal collapse_counter
        # Sustained-collapse check: the robot is resting on something that is
        # not its feet.
        if float(data.qpos[2]) < COLLAPSE_HEIGHT_THRESHOLD:
            collapse_counter += 1
        else:
            collapse_counter = 0
        if collapse_counter >= COLLAPSE_DWELL_STEPS:
            return True
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

    # region out-of-bounds reset --------------------------
    # Ramp and staired pyramid: this far past the farthest spawn, the robot is
    # walking flat floor away from the terrain. Not a crash: the episode is
    # kept, and the next one starts on a new pyramid.
    OUT_OF_BOUNDS_MARGIN = 0.4   # [m]

    def _out_of_bounds() -> bool:
        if ramp_pyramid.active:
            extent = ramp_pyramid.spawn_reach
        elif pyramid_sample is not None:
            extent = pyramid_terrain.floor_ring_outer(pyramid_sample)
        else:
            return False
        return float(np.max(np.abs(data.qpos[:2]))) > extent + OUT_OF_BOUNDS_MARGIN
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
        terrain,
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

    command = None

    def step_controller():
        nonlocal counter, tau, q_ref, mpc_data, command # nonlocal variables are used to modify the variables in the outer scope

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        
        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
           
            if extreme is not None:
                extreme.step()
                command = jnp.asarray(extreme.mpc_input(config.robot_height))
            elif use_navigation:
                command = jnp.asarray(navigator.mpc_input(qpos, config.robot_height))
            else:
                command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            contact = jnp.asarray(estimate_contacts(data, contact_ids))
            camurri_prob, camurri_stance = camurri_from_sim(model, data, contact_ids)
            if not collecting and not headless:
                print(f"Contact: {contact}")
                print(format_camurri(camurri_prob, camurri_stance))
                print(foot)
                print(f"Command: {command}")
            
            start = timer()
            mpc_contact = (
                contact
                if getattr(config, "mpc_model", "whole_body") == "srbd"
                else contact * 0.0
            )
            mpc_data, tau = solve_mpc(
                mpc_data,
                qpos,
                qvel,
                foot,
                command,
                mpc_contact,
                terrain,
            )
            tau.block_until_ready()
            stop = timer()

            # tau = jnp.clip(tau, config.min_torque, config.max_torque)
            # The shifted warm start is the next joint target used by the PD stabilizer.
            if getattr(config, "mpc_model", "whole_body") == "srbd":
                q_ref = config.q0.copy()
            else:
                q_ref = mpc_data.X0[0, 7 : 7 + config.n_joints]
            if not headless and not collecting:
                print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        data.ctrl = motor(np.asarray(tau), data.qvel[6 : 6 + config.n_joints])

        base_force_pert.tick_and_apply(data) # apply random base force perturbation
        base_weight.apply(data)  # extra mass: F = m g, world-down or base-normal
 
        mujoco.mj_step(model, data)
        counter += 1

        closed = collect_hooks.after_physics_step(
            model, data, np.asarray(tau), contact_ids, base_force_pert, config.n_joints,
            control=ControlSample(
                tau_cmd=np.asarray(tau),
                # The MPC's own gait timer: phase per leg, and the contact pattern
                # it planned. Logged as privileged so the leak the commanded
                # torques carry is auditable instead of hidden.
                leg_phase=np.asarray(mpc_data.contact_time),
                duty_factor=float(mpc_data.duty_factor),
                # command_from_mpc_input, not command[:3]: the MPC vector is
                # [vx, vy, 0, 0, 0, yaw_rate, height], so slicing the first three
                # logged a structural zero for yaw across the whole of the
                # audited v4 run.
                cmd_base_vel=(
                    command_from_mpc_input(np.asarray(command))
                    if command is not None
                    else None
                ),
                cmd_segment_id=(
                    extreme.segment_id if extreme is not None else 0
                ),
            ),
        )

        if pickle_collector is not None and pickle_collector.step(data):
            _randomize_episode()

        # The duration cap closed an episode: the next one gets fresh knobs.
        if closed:
            _randomize_episode()

        # Reaching the goal is a task success, and the natural place to close an
        # episode and advance the domain randomization: otherwise a robot that
        # keeps reaching goals carries one friction and payload for the whole
        # episode. But not at EVERY goal — v3 did that and left a 9 s median
        # episode, far short of the steady state a temporal representation needs.
        # So goals before the threshold only resample the goal; the first one
        # after it closes the episode and redraws the knobs.
        if use_navigation and navigator.reached(data.qpos):
            episode_cfg = dataset_collection_config.episode
            episode_seconds = (
                pickle_collector.seconds if pickle_collector is not None
                else collect_hooks.episode_seconds
            )
            long_enough = episode_seconds >= episode_cfg.goal_closes_episode_after_s
            if episode_cfg.end_episode_on_goal and long_enough:
                if pickle_collector is not None:
                    pickle_collector.end_episode()
                else:
                    collect_hooks.end_episode(reason="goal_reached")
                _randomize_episode()
            if navigator.auto_resample:
                navigator.sample_goal(np.asarray(data.qpos))

        if _is_crashed():
            _respawn(crashed=True)
        elif _out_of_bounds():
            if not collecting:
                print("[reset] left the terrain region", flush=True)
            collect_hooks.end_episode(reason="out_of_bounds")
            _respawn()

    if headless:
        for _ in range(steps):
            step_controller()
        if pickle_collector is not None:
            pickle_collector.close()
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
    _weight_geom_id = -1
    _height_map_ids = None 

    if plotter is not None:
        plotter.start()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        viewer.sync()
        # Once, so the view starts on the robot. Repeating this every frame
        # overwrites azimuth, elevation, and distance and locks the camera.
        sim_utils.setup_tracking_camera(
            viewer, model, body_name=getattr(config, "base_body_name", "base")
        )
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
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
            if base_weight.enabled and base_weight.magnitude > 0.0:
                weight_vec = base_weight.force
                weight_color = np.array([0.15, 0.45, 1.0, 0.85])
                weight_scale = float(np.linalg.norm(weight_vec)) * 0.005
            else:
                weight_vec = np.array([0.0, 0.0, 1e-3])
                weight_color = np.array([0.0, 0.0, 0.0, 0.0])
                weight_scale = 1e-3
            _weight_geom_id = sim_utils.render_vector(
                viewer,
                vector=weight_vec,
                pos=base_pos + np.array([0.0, 0.0, 0.18]),
                scale=weight_scale,
                color=weight_color,
                geom_id=_weight_geom_id,
            )
            # endregion -------------------------------------
            #---------------------------------------------------

            # Update / render the navigation goal (handles user-pointed goals
            # via the camera lookat and auto-resamples random goals when reached).
            if use_navigation:
                navigator.update(np.asarray(data.qpos), viewer)

            overlay = terrain_overlay_grid(config)
            if overlay is not None:
                _height_map_ids = sim_utils.render_height_grid(
                    viewer,
                    overlay[0],
                    overlay[1],
                    overlay[2],
                    base_pos[:2],
                    _height_map_ids,
                )

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
                    foot_vel=sim_utils.geom_linear_velocities(model, data, contact_ids),
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

    if pickle_collector is not None:
        pickle_collector.close()
    collect_hooks.finish("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, choices=["flat", "rough", "perlin","stairs","ramp", "staired_pyramid", "slippery"], default="flat")
    parser.add_argument("--robot", type=str, choices=["aliengo", "mini_cheetah", "go2", "go2_dls", "hyqreal", "spot", "b2"], default="go2_dls")
    parser.add_argument(
        "--nav",
        type=str,
        choices=["random", "random_fb", "pointuser", "vel", "extreme"],
        default="vel",
        help=(
            "Navigation mode: random goals (always forward), random goals "
            "approached forward or backward, user-pointed goal, keyboard velocity, "
            "or extreme: slew to that robot's velocity limits "
            "(linear, yaw, both, or a full stop, either sign)."
        ),
    )
    parser.add_argument(
        "--gait",
        type=str,
        choices=["trot", "pace", "crawl", "bound"],
        default=None,
        help=(
            "Locomotion gait (go2 / spot / b2). Selects the matched timing / "
            "swing / weight set from that robot's gait registry."
        ),
    )
    parser.add_argument(
        "--mpc-model",
        type=str,
        choices=["whole_body", "inverse_dynamics", "srbd"],
        default="whole_body",
        help=(
            "Go2 MPC transcription: whole-body dynamics (default), "
            "inverse-dynamics with equality constraints, or centroidal SRBD."
        ),
    )
    parser.add_argument(
        "--terrain-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On --scene rough, stairs, ramp, or staired_pyramid, land swings on "
            "the raycast height grid and draw that grid. "
            "--no-terrain-aware keeps the blind foothold height and draws nothing. "
            "Ignored on other scenes."
        ),
    )
    parser.add_argument("--headless", action="store_true")
    add_collect_argument(parser)
    parser.add_argument(
        "--collect-out",
        type=str,
        default=None,
        help="Run directory name for --collect (relative names go under the configured dataset root; default: auto-named from robot/scene/gait/time).",
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
        collect=args.collect,
        collect_out=args.collect_out,
        episode_duration_s=args.episode_duration,
        gait=args.gait,
        mpc_model=args.mpc_model,
        terrain_aware=args.terrain_aware,
    )
