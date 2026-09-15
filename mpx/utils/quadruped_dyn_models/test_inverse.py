import os

import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from mpx.utils.quadruped_dyn_models.inverse import (
    inv_dyn_control_slices,
    inv_dyn_dims,
    pack_inverse_state,
    quadruped_inv_dyn_dynamics,
    quadruped_inv_dyn_equality,
    quadruped_inv_dyn_obj,
)

_DIR = os.path.dirname(os.path.realpath(__file__))
_MODEL_PATH = os.path.abspath(os.path.join(_DIR, "..", "..", "data", "go2", "go2_mjx.xml"))

N_JOINTS = 12
N_CONTACT = 4
DT = 0.02
CONTACT_FRAME = ["FL", "FR", "RL", "RR"]
BODY_NAME = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]


def _go2_mjx():
    model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
    mjx_model = mjx.put_model(model)
    contact_id = [
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in CONTACT_FRAME
    ]
    body_id = [
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in BODY_NAME
    ]
    return model, mjx_model, contact_id, body_id


def _standing_x(model):
    nq, nv, n, _, _ = inv_dyn_dims(N_JOINTS, N_CONTACT)
    del n
    qpos = np.zeros(nq)
    qpos[2] = 0.27
    qpos[3] = 1.0
    qpos[7:] = np.array([0, 0.9, -1.8] * 4)
    qvel = np.zeros(nv)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return pack_inverse_state(jnp.asarray(data.qpos), jnp.asarray(data.qvel))


def test_inv_dyn_dims_go2():
    nq, nv, n, m, equality_dim = inv_dyn_dims(N_JOINTS, N_CONTACT)
    assert (nq, nv, n, m, equality_dim) == (19, 18, 37, 42, 18)
    qacc, tau, grf = inv_dyn_control_slices(N_JOINTS, N_CONTACT)
    assert qacc == slice(0, 18)
    assert tau == slice(18, 30)
    assert grf == slice(30, 42)


def test_dynamics_zero_acc_leaves_qpos():
    nq, nv, _, _, _ = inv_dyn_dims(N_JOINTS, N_CONTACT)
    x = pack_inverse_state(jnp.zeros(nq).at[2].set(0.27).at[3].set(1.0), jnp.zeros(nv))
    u = jnp.zeros(42)
    x_next = quadruped_inv_dyn_dynamics(
        nq, nv, N_JOINTS, N_CONTACT, DT, x, u, 0, jnp.ones((1, 4))
    )
    np.testing.assert_allclose(np.asarray(x_next[:nq]), np.asarray(x[:nq]), atol=1e-6)


def test_equality_small_when_consistent():
    model, mjx_model, contact_id, body_id = _go2_mjx()
    nq, nv, n, m, _ = inv_dyn_dims(N_JOINTS, N_CONTACT)
    x = _standing_x(model)
    qpos, qvel = x[:nq], x[nq : nq + nv]
    data = mjx.make_data(mjx_model)
    data = data.replace(qpos=qpos, qvel=qvel, qacc=jnp.zeros(nv))
    data = mjx.fwd_position(mjx_model, data)
    data = mjx.fwd_velocity(mjx_model, data)
    D = data.qfrc_bias
    jacobians = []
    for cid, bid in zip(contact_id, body_id):
        jac, _ = mjx.jac(mjx_model, data, data.geom_xpos[cid], bid)
        jacobians.append(jac)
    J = jnp.concatenate(jacobians, axis=1)
    actuation = jnp.concatenate([jnp.zeros((6, N_JOINTS)), jnp.eye(N_JOINTS)], axis=0)
    A = jnp.concatenate([J, actuation], axis=1)
    sol, *_ = jnp.linalg.lstsq(A, D, rcond=None)
    grf, tau = sol[: 3 * N_CONTACT], sol[3 * N_CONTACT :]
    u = jnp.concatenate([jnp.zeros(nv), tau, grf])
    parameter = jnp.ones((1, N_CONTACT))
    residual = quadruped_inv_dyn_equality(
        mjx_model, contact_id, body_id, nq, nv, N_JOINTS, N_CONTACT, x, u, 0, parameter
    )
    assert residual.shape == (nv,)
    assert float(jnp.linalg.norm(residual)) < 1e-3


def test_cost_finite_on_locomotion_reference_layout():
    model, mjx_model, contact_id, _ = _go2_mjx()
    nq, nv, n, m, _ = inv_dyn_dims(N_JOINTS, N_CONTACT)
    x = _standing_x(model)
    u = jnp.zeros(m)
    p_ref = x[:3]
    quat_ref = x[3:7]
    q_ref = x[7:nq]
    dp_ref = x[nq : nq + 3]
    omega_ref = x[nq + 3 : nq + 6]
    foot_ref = jnp.zeros(3 * N_CONTACT)
    contact = jnp.ones(N_CONTACT)
    grf_ref = jnp.zeros(3 * N_CONTACT)
    row = jnp.concatenate(
        [p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact, grf_ref]
    )
    reference = jnp.stack([row, row])
    W = {
        "pos": jnp.diag(jnp.array([0.0, 0.0, 1.0])),
        "rot": jnp.eye(3),
        "q": jnp.eye(N_JOINTS),
        "vel": jnp.eye(3),
        "omega": jnp.eye(3),
        "dq": jnp.eye(N_JOINTS),
        "contact": jnp.eye(3 * N_CONTACT),
        "acc": jnp.eye(nv),
        "tau": jnp.eye(N_JOINTS),
        "grf": jnp.eye(3 * N_CONTACT),
    }
    value = quadruped_inv_dyn_obj(
        mjx_model, contact_id, nq, nv, N_JOINTS, N_CONTACT, 1, W, reference, x, u, 0
    )
    assert jnp.isfinite(value)
    del n


def test_go2_inverse_config_dims_and_model_path():
    from mpx.config.robot_config.config_go2 import (
        Go2MpcModel,
        Go2Mode,
        go2_config,
    )

    cfg = go2_config(Go2Mode.LOCOMOTION, mpc_model=Go2MpcModel.INVERSE_DYNAMICS)
    assert cfg.n == 37
    assert cfg.m == 42
    assert cfg.solver_mode == "equality"
    assert cfg.mpc_model == Go2MpcModel.INVERSE_DYNAMICS
    assert cfg.model_path.endswith("data/go2/go2_mjx.xml")
    assert os.path.isfile(cfg.model_path)

    wb = go2_config(Go2Mode.LOCOMOTION)
    assert wb.mpc_model == Go2MpcModel.WHOLE_BODY
    assert wb.m == 12

