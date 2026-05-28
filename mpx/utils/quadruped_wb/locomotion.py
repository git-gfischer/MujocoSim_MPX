"""Locomotion / gait MPC: nominal contact solve tuning for cyclic standing & swing."""

from mpx.utils.quadruped_wb.core import quadruped_wb_step

# Milder ridge on vacant contact slots keeps less regularization overhead when gait
# opens and closes contacts every cycle.
_LOCOMOTION_BAUMGARTE_ALPHA = 12.0
_LOCOMOTION_RIDGE_STANCE = 5e-6
_LOCOMOTION_RIDGE_SWING = 3e-3


def quadruped_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    return quadruped_wb_step(
        baumgarte_alpha=_LOCOMOTION_BAUMGARTE_ALPHA,
        ridge_stance=_LOCOMOTION_RIDGE_STANCE,
        ridge_swing=_LOCOMOTION_RIDGE_SWING,
        model=model,
        mjx_model=mjx_model,
        contact_id=contact_id,
        body_id=body_id,
        n_joints=n_joints,
        dt=dt,
        x=x,
        u=u,
        t=t,
        parameter=parameter,
    )
