"""Unitree G1 whole-body MPC locomotion in MuJoCo.

    python mpx/simulators/humanoid/mjx_g1.py                 # viewer, walk at 0.3 m/s
    python mpx/simulators/humanoid/mjx_g1.py --vx 0 --wz 0.3 # turn in place
    python mpx/simulators/humanoid/mjx_g1.py --headless --steps 5000
"""
import argparse
import os
import sys
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..', '..', '..')))

import jax
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.robot_config.config_g1 as config
from mpx.utils.quad_utils_locomotion.mpc_wrapper_locomotion import MPCWrapper
import mpx.utils.simulation_utils.sim_utils as sim_utils

SIM_FREQUENCY = 500.0
JOINT_DAMPING = 3.0
COMMAND_RAMP_S = 2.0


def main(headless=False, steps=5000, vx=0.3, vy=0.0, wz=0.0):
    model = mujoco.MjModel.from_xml_path(config.scene_path)
    data = mujoco.MjData(model)
    model.opt.timestep = 1 / SIM_FREQUENCY
    contact_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.contact_frame]

    mpc = MPCWrapper(config, limited_memory=True)

    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command):
        return mpc.run(mpc_data, mpc.pack_state(qpos, qvel, foot), command)

    reset_mpc = jax.jit(mpc.reset)

    data.qpos = np.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_forward(model, data)
    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    mpc_every = int(SIM_FREQUENCY / config.mpc_frequency)
    tau = np.zeros(config.n_joints)

    def step(counter):
        nonlocal mpc_data, tau
        if counter % mpc_every == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
            # A step change in velocity tips the stance foot onto one contact point, where it spins.
            ramp = min(1.0, data.time / COMMAND_RAMP_S)
            command = jnp.array([ramp * vx, ramp * vy, 0.0, 0.0, 0.0, ramp * wz, config.robot_height])
            start = timer()
            mpc_data, tau_j = solve_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot, command)
            tau = np.asarray(tau_j)
            if not headless:
                print(f"MPC time: {1e3 * (timer() - start):.2f} ms")
        data.ctrl = tau - JOINT_DAMPING * data.qvel[6:6 + config.n_joints]
        mujoco.mj_step(model, data)

    def status():
        w, x, y, z = data.qpos[3:7]
        yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return f"t={data.time:5.2f}s base=({data.qpos[0]:+.3f}, {data.qpos[1]:+.3f}, {data.qpos[2]:.3f}) yaw={yaw:+.1f}deg"

    if headless:
        for counter in range(steps):
            step(counter)
            if counter % 500 == 0:
                print(status())
            if data.qpos[2] < 0.5 * config.robot_height or not np.isfinite(data.qpos).all():
                print(f"G1 fell at t={data.time:.2f}s")
                return False
        print("Done:", status())
        return True

    with mujoco.viewer.launch_passive(model, data) as viewer:
        counter = 0
        while viewer.is_running():
            step(counter)
            counter += 1
            viewer.sync()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=5000, help="sim steps in headless mode")
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    args = parser.parse_args()
    ok = main(headless=args.headless, steps=args.steps, vx=args.vx, vy=args.vy, wz=args.wz)
    sys.exit(0 if ok else 1)
