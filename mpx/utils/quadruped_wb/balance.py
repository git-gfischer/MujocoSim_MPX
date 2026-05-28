"""Reduced-support balance MPC: tripod / diagonal contacts with heavier regularization."""

from mpx.utils.quadruped_wb.core import quadruped_wb_step

# Stronger ridge on deactivated contact directions stabilizes JT M^-1 J when <=3 feet nominal.
_BALANCE_BAUMGARTE_ALPHA = 12.0
_BALANCE_RIDGE_STANCE = 5e-6
_BALANCE_RIDGE_SWING = 5e-3


def quadruped_wb_dynamics_balance(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    return quadruped_wb_step(
        baumgarte_alpha=_BALANCE_BAUMGARTE_ALPHA,
        ridge_stance=_BALANCE_RIDGE_STANCE,
        ridge_swing=_BALANCE_RIDGE_SWING,
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
