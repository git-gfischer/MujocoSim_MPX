
import types

import jax.numpy as jnp
import jax
import mujoco
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
 
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_vector
 
import mpx.utils.mpc_wrapper as mpc_wrapper
from mpx.config.robot_config.config_go2 import go2_config, Go2Mode, BalanceStance

from mpx.config.sim_config.config_quad_spawn import spawn_config
from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config

from mpx.utils.base_force_perturbation import RandomBaseForcePerturbation
from mpx.utils.quadruped_wb.base_pose import (
    DesiredBasePoseVisualizationConfig,
    DesiredBasePoseVisualizer,
    BasePoseRandomizer,
)

from mpx.utils.spawner import (
    RobotMapSpawner,
    SpawnCollisionError,
    _random_map_respawn,
    reset_robot_and_mpc,
)

from timeit import default_timer as timer
# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

# region Simulation configuration ------------------------------------------------

# Go2Mode.LOCOMOTION (walking/trot MPC) vs Go2Mode.BALANCE (static contact schedule)
#GO2_CONTROLLER_MODE = Go2Mode.LOCOMOTION # (will override the balance_stance)
GO2_CONTROLLER_MODE = Go2Mode.BALANCE

# Balance: BalanceStance.FOUR, TRIPOD_SWING_FL|FR|RL|RR, DIAG_FL_RR, DIAG_FR_RL.
# Pass nominal MPC contact flags from the stance so MPC isn't fed all-ones detector contact while a leg is commanded swing.
config = go2_config(GO2_CONTROLLER_MODE, balance_stance=BalanceStance.FOUR)

 # Define robot and scene parameters
robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "random_boxes" # "random_boxes", "rough", "perlin", "flat", "random_boxes_sparse", "random_boxes_dense"

# endregion

# region QuadrupedEnv configuration ----------------------------------------------
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = config.mpc_frequency

state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables
 
# Initialize simulation environment
sim_frequency = 200.0
env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=0.7,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )

# endregion

obs = env.reset(random=False)
# Define the MPC wrapper
mpc = mpc_wrapper.MPCControllerWrapper(config)

_base_pose_randomizer = FourLegBalanceBasePoseRandomizer(
    nominal_p0=np.asarray(config.p0, dtype=np.float64),
    nominal_quat0=np.asarray(config.quat0, dtype=np.float64),
    # Spawn pose randomization: keep legacy in-example tuning.
    cfg=FourLegBalanceBasePoseRandomizationConfig(
        enabled=True,
        z_offset_range=(-0.03, 0.06),
        roll_range_deg=(-7.5, 7.5),
        pitch_range_deg=(-7.5, 7.5),
        yaw_range_deg=(-20.0, 20.0),
    ),
    rng_seed=spawn_config.rng_seed,
)
_desired_pose_randomizer = FourLegBalanceBasePoseRandomizer(
    nominal_p0=np.asarray(config.p0, dtype=np.float64),
    nominal_quat0=np.asarray(config.quat0, dtype=np.float64),
    # Desired pose randomization: uses defaults from pose_ref_4leg_config.py.
    cfg=FourLegBalanceBasePoseRandomizationConfig(),
    rng_seed=None if spawn_config.rng_seed is None else (spawn_config.rng_seed + 1),
)
_desired_base_pose_visualizer = DesiredBasePoseVisualizer(
    DesiredBasePoseVisualizationConfig(enabled=True)
)




# region spawner configuration ---------------------------------------------------
spawn_region = spawn_config.spawn_region()
_spawn_region_visual = None
_spawner: RobotMapSpawner | None = None

if spawn_config.use_random_map_spawn:
    _refresh_config_reset_pose()
    _rng = np.random.default_rng(spawn_config.rng_seed)
    _spawner = RobotMapSpawner(
        spawn_region,
        rng=_rng,
        check_collisions=spawn_config.check_any_collision,
        check_self_collision=spawn_config.check_self_collision,
        check_env_collision=spawn_config.check_env_collision,
        foot_geom_names=tuple(robot_feet_geom_names.values())
        if spawn_config.check_any_collision
        else None,
        robot_root_body_name=spawn_config.robot_root_body_name,
        max_spawn_attempts=spawn_config.max_attempts,
        on_collision_exhausted=spawn_config.on_collision_exhausted,
        try_foot_vertical_relief=spawn_config.try_foot_vertical_relief,
        foot_relief_step=spawn_config.foot_relief_step,
        foot_relief_max=spawn_config.foot_relief_max,
        verbose=spawn_config.verbose,
    )
    try:
        _spawner.apply(env, config.p0, config.quat0, config.q0)
    except SpawnCollisionError as e:
        import traceback

        print(f"[mjx_quad] spawn failed: {e}", flush=True)
        traceback.print_exc()
        raise
else: # Spawn always in the same place
    _refresh_config_reset_pose()
    env.mjData.qpos[:] = np.asarray(
        jnp.concatenate([config.p0, config.quat0, config.q0]), dtype=np.float64
    )
    env.mjData.qvel[:] = 0.0
    mujoco.mj_forward(env.mjModel, env.mjData)
    # Region overlay still needs a spawner instance (same ``SpawnRegion`` bounds).
    if spawn_config.show_spawn_region:
        _spawner = RobotMapSpawner(spawn_region, check_collisions=False)
# endregion


env.render()  # creates passive viewer

if spawn_config.show_spawn_region and _spawner is not None:
    _spawn_region_visual = _spawner.render_spawn_region(env.viewer, z=spawn_config.region_z)
    env.viewer.sync()

# Random pulsed push on the base (world-frame force via qfrc_applied).
_base_force_pert = RandomBaseForcePerturbation.from_config(
    sim_dt=1.0 / sim_frequency,
    cfg=ext_base_force_config,
)
_ext_force_vec_id = -1


def _apply_random_base_force() -> None:
    _base_force_pert.tick_and_apply(env.mjData)

# region reset function for the QuadrupedEnv keys + R/Backspace for a fresh random map spawn
def _mjx_quad_key_callback(self, keycode: int) -> None:
    """Reset function QuadrupedEnv keys + R/Backspace for a fresh random map spawn."""
    global q, tau
    if (
        spawn_config.use_random_map_spawn
        and _spawner is not None
        and int(keycode) in spawn_config.respawn_keycodes
    ):
        print(f"[mjx_quad] random map respawn (key {keycode})", flush=True)
        _refresh_config_reset_pose()
        q, tau = reset_robot_and_mpc(env, config, mpc, _spawner)
        _refresh_mpc_desired_pose()
        _base_force_pert.reset()
        return
    QuadrupedEnv._key_callback(self, keycode)


env._key_callback = types.MethodType(_mjx_quad_key_callback, env)
# endregion

ids = []
# for i in range(8):
#      ids.append(render_vector(env.viewer,
#               np.zeros(3),
#               np.zeros(3),
#               0.1,
#               np.array([1, 0, 0, 1])))
counter = 0
# Main simulation loop
tau = jnp.zeros(config.n_joints)
tau_old = jnp.zeros(config.n_joints)
delay = int(0.007*sim_frequency)
print('Delay: ',delay)

q = config.q0.copy()
dq = jnp.zeros(config.n_joints)
mpc_time = 0
mpc.reset(np.asarray(env.mjData.qpos, dtype=np.float64).copy(),
          np.asarray(env.mjData.qvel, dtype=np.float64).copy())
_refresh_mpc_desired_pose()


def _mjx_quad_step(action):
    _apply_random_base_force()
    return env.step(action=action)


while env.viewer.is_running(): # Main simulation loop
 
    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()
    if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
    
        # reference velocity of the base ------------------------------------------
        ref_base_lin_vel = env._ref_base_lin_vel_H
        ref_base_ang_vel = np.array([0.0, 0.0, env._ref_base_ang_yaw_dot])
        # Balance mode assumes a stationary reference; locomotion follows human joystick.
        if config.behaviour == "balance":
            ref_base_lin_vel = np.zeros(3)
            ref_base_ang_vel = np.zeros(3)
        # --------------------------------------------------------------------------
        
        # input to the MPC ---------------------------------------------------------
        # [vx, vy, vz, wx, wy, wz, robot_height]
        input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           mpc.robot_height])
        # --------------------------------------------------------------------------

        # contact state ------------------------------------------------------------
        contact_temp, _ = env.feet_contact_state()
        contact_meas = np.array(
            [contact_temp[robot_feet_geom_names[leg]] for leg in ["FL", "FR", "RL", "RR"]],
            dtype=float,
        )
        if getattr(config, "use_balance_fixed_contact", False) and config.behaviour == "balance":
            contact = np.asarray(config.balance_fixed_contact_mask, dtype=float)
            contact = np.where(contact > 0.5, 1.0, 0.0)
        else:
            contact = contact_meas

        # --------------------------------------------------------------------------

        if counter != 0: # Feedback control loop
            for i in range(delay):
                qpos = env.mjData.qpos.copy()
                qvel = env.mjData.qvel.copy()
                # tau_fb = K@(x-np.concatenate([qpos,qvel]))

                tau_fb = 10*(q-qpos[7:7+config.n_joints])-2*(qvel[6:6+config.n_joints])# Feedback control law
                state, reward, is_terminated, is_truncated, info = _mjx_quad_step(tau + tau_fb)
                counter += 1
        start = timer()
        tau, q, dq = mpc.run(qpos,qvel,input,contact)
        stop = timer()

        stop = timer()
        # for i in range(4):
        #     render_sphere(env.viewer,
        #                   collision_point[3*i:3*i+3],
        #                   0.2,
        #                   np.array([1, 0, 0, 0.5]),
        #                   ids[i])

    tau_fb = 10*(q-qpos[7:7+config.n_joints])-2*(qvel[6:6+config.n_joints])
    state, reward, is_terminated, is_truncated, info = _mjx_quad_step(tau + tau_fb)

    if is_terminated or is_truncated:
        _refresh_config_reset_pose()
        if spawn_config.use_random_map_spawn and _spawner is not None:
            _, q, tau = _random_map_respawn(env, config, mpc, _spawner)
        else:
            q, tau = reset_robot_and_mpc(env, config, mpc, None)
        _refresh_mpc_desired_pose()
        _base_force_pert.reset()

    # time.sleep(0.1)
    counter += 1
    env.render()

    if spawn_config.show_spawn_region and _spawner is not None:
        _spawn_region_visual = _spawner.render_spawn_region(
            env.viewer, z=spawn_config.region_z, visual=_spawn_region_visual
        )

    _desired_base_pose_visualizer.update(
        env.viewer,
        desired_pos_xyz=np.asarray(mpc.last_base_ref_pos, dtype=np.float64),
        desired_quat_wxyz=np.asarray(mpc.last_base_ref_quat, dtype=np.float64),
    )

    if _base_force_pert.enabled and _base_force_pert.is_active:
        base_pos = np.asarray(env.mjData.qpos[:3], dtype=np.float64)
        _ext_force_vec_id = render_vector(
            env.viewer,
            _base_force_pert.force,
            pos=base_pos + np.array([0.0, 0.0, 0.2]),
            scale=0.02,
            color=np.array([1.0, 0.15, 0.15, 0.85]),
            geom_id=_ext_force_vec_id,
        )