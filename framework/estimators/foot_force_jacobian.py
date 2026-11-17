"""Foot force from the leg Jacobian and the joint torque.

The solve matches ``physics_features.analytic_grf`` with ``sign=-1``, then
negates that result so a standing foot has negative world Z, the same sign
as ``estimate_foot_grf``. Returning ``solve(A, J @ tau)`` is that negation.
Do not negate again.
"""
from __future__ import annotations

import numpy as np
import mujoco

from framework.config.sim_config.config_foot_force_jacobian import foot_force_jacobian_config

_EPS = 1e-9
_FOOT_NAMES = ("FL", "FR", "RL", "RR")


def jacobian_foot_force(tau, J, tau_bias=None, damping=1e-2):
    """World force (..., 3) from torque (..., 3) and Jacobian (..., 3, 3)."""
    tau = np.asarray(tau, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    if tau_bias is not None:
        tau = tau - np.asarray(tau_bias, dtype=np.float64)
    jjt = J @ np.swapaxes(J, -1, -2)
    scale = np.maximum(np.trace(jjt, axis1=-2, axis2=-1), _EPS)
    regularized = jjt + damping * np.eye(3) * scale[..., None, None]
    return np.linalg.solve(regularized, J @ tau[..., None])[..., 0]


def format_jacobian_grf(solver, estimated) -> str:
    """``Jacobian GRF FL=(sx sy sz)/(jx jy jz) ...`` solver triple, then estimate."""
    solver = np.asarray(solver, dtype=np.float64).reshape(-1, 3)
    estimated = np.asarray(estimated, dtype=np.float64).reshape(-1, 3)
    parts = []
    n = min(solver.shape[0], estimated.shape[0])
    for i in range(n):
        name = _FOOT_NAMES[i] if i < len(_FOOT_NAMES) else str(i)
        s = solver[i]
        e = estimated[i]
        parts.append(
            f"{name}=({s[0]:.1f} {s[1]:.1f} {s[2]:.1f})/"
            f"({e[0]:.1f} {e[1]:.1f} {e[2]:.1f})"
        )
    return "Jacobian GRF " + " ".join(parts)


def _leg_dofs(model, geom_id):
    """Three nv indices for the calf joint and the two joints above it.

    None when the geom, the body joint, or any index is outside ``[6, nv)``.
    """
    gid = int(geom_id)
    if gid < 0 or gid >= int(model.ngeom):
        return None
    body = int(model.geom_bodyid[gid])
    joint = int(model.body_jntadr[body])
    if joint < 0:
        return None
    dofs = int(model.jnt_dofadr[joint]) + np.array([-2, -1, 0], dtype=np.int32)
    if np.any(dofs < 6) or np.any(dofs >= int(model.nv)):
        return None
    return dofs


def estimate_foot_force_jacobian(model, data, contact_geom_ids, cfg=None):
    """World XYZ force on each foot geom, solver sign, shape ``(n_feet, 3)``.

    ``cfg`` defaults to ``foot_force_jacobian_config``. Disabled configs return
    zeros and do not read ``model`` or ``data``.
    """
    cfg = foot_force_jacobian_config if cfg is None else cfg
    ids = [int(g) for g in contact_geom_ids]
    out = np.zeros((len(ids), 3), dtype=np.float64)
    if not cfg.enabled:
        return out

    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    J = np.zeros((len(ids), 3, 3), dtype=np.float64)
    tau = np.zeros((len(ids), 3), dtype=np.float64)
    bias = np.zeros((len(ids), 3), dtype=np.float64)
    valid = np.zeros(len(ids), dtype=bool)
    actuator = np.asarray(data.qfrc_actuator, dtype=np.float64).reshape(-1)
    bias_all = (
        np.asarray(data.qfrc_bias, dtype=np.float64).reshape(-1)
        if cfg.subtract_bias
        else None
    )
    for i, gid in enumerate(ids):
        dofs = _leg_dofs(model, gid)
        if dofs is None:
            continue
        mujoco.mj_jacGeom(model, data, jacp, None, gid)
        J[i] = jacp[:, dofs]
        tau[i] = actuator[dofs]
        if bias_all is not None:
            bias[i] = bias_all[dofs]
        valid[i] = True
    if not np.any(valid):
        return out
    solved = jacobian_foot_force(
        tau,
        J,
        tau_bias=bias if cfg.subtract_bias else None,
        damping=cfg.damping,
    )
    out[valid] = solved[valid]
    return out
