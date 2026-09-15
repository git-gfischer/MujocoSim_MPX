"""
Boston Dynamics Spot whole-body MPC configuration.

Pick a behaviour (same attribute layout ``MPCWrapper`` expects)::

    from mpx.config.robot_config.config_spot import spot_config, SpotMode, BalanceStance

    cfg = spot_config(SpotMode.LOCOMOTION)
    cfg = spot_config(SpotMode.BALANCE, balance_stance=BalanceStance.TRIPOD_SWING_FR)

Or construct directly: ``SpotLocomotion()``, ``SpotBalance(BalanceStance.TRIPOD_SWING_FR)``.

The module exposes ``config`` as locomotion defaults for backward compatibility.

Contact / body names follow the MuJoCo model in
``mpx/data/boston_dynamics_spot/spot.xml``: feet ``FL, FR, HL, HR`` and
lower-leg bodies ``fl_lleg, fr_lleg, hl_lleg, hr_lleg``. Balance-stance tags
still use Go2's ``RL`` / ``RR`` names; those map to the hind-left / hind-right
slots (indices 2 and 3).
"""
from __future__ import annotations

import os
from functools import partial
import jax
import jax.numpy as jnp

import mpx.utils.quadruped_dyn_models.models as mpc_dyn_model
import mpx.utils.quadruped_dyn_models.objectives as mpc_objectives
from mpx.config.robot_config.config_go2 import BalanceStance, balance_stance_to_mask

_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(_DIR, "..", "..", "data", "boston_dynamics_spot", "spot.xml")
)


class SpotMode:
    """String tags for ``spot_config(...)``."""

    LOCOMOTION = "locomotion"
    BALANCE = "balance"


class _SpotCommon:
    """MuJoCo model topology and MPC dimensions shared by all Spot behaviours."""

    behaviour: str = ""
    model_path: str = _DEFAULT_MODEL_PATH
    contact_frame = ["FL", "FR", "HL", "HR"]
    body_name = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
    base_body_name: str = "body"

    dt: float = 0.02
    N: int = 25
    mpc_frequency: int = 50

    quat0 = jnp.array([1, 0, 0, 0])
    q0 = jnp.array([0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8])
    q0_init = jnp.array([0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8])
    p_legs0 = jnp.array([
        0.34, 0.175, 0.0,
        0.34, -0.175, 0.0,
        -0.34, 0.175, 0.0,
        -0.34, -0.175, 0.0,
    ])

    grf_as_state: bool = True
    u_ref = jnp.zeros(12)

    use_balance_fixed_contact: bool = False
    balance_fixed_contact_mask = jnp.ones(4, dtype=jnp.float32)

    # Hip motors allow ±144.4 Nm; thigh/calf ±135.278 Nm. Use the tighter bound.
    max_torque: float = 135.278
    min_torque: float = -135.278

    @property
    def n_joints(self) -> int:
        return 12

    @property
    def n_contact(self) -> int:
        return len(self.contact_frame)

    @property
    def n(self) -> int:
        return 13 + 2 * self.n_joints + 6 * self.n_contact

    @property
    def m(self) -> int:
        return self.n_joints

    @property
    def initial_state(self) -> jnp.ndarray:
        """Nominal MPC state vector of length ``self.n`` (uses subclass ``p0``)."""
        core = jnp.concatenate(
            [self.p0, self.quat0, self.q0, jnp.zeros(6 + self.n_joints), self.p_legs0]
        )
        if self.grf_as_state:
            return jnp.concatenate([core, jnp.zeros(3 * self.n_contact)])
        return core


class SpotLocomotion(_SpotCommon):
    """Trot gait, terrain estimator on, lateral base position softly unconstrained."""

    behaviour: str = "locomotion"
    swing_tracking: bool = True

    robot_height: float = 0.46
    p0 = jnp.array([0, 0, robot_height])

    timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # trot
    duty_factor: float = 0.65
    step_freq: float = 1.35
    step_height: float = 0.12
    initial_height: float = 0.46
    clearance_speed: float = 0.2

    use_terrain_estimation: bool = True
    solver_mode = "primal_dual"

    @property
    def cost(self):
        return partial(mpc_objectives.quadruped_wb_obj, True, self.n_joints, self.n_contact, self.N)

    @property
    def dynamics(self):
        n_joints = self.n_joints
        dt = self.dt
        def _factory(model, mjx_model, contact_id, body_id):
            return partial(mpc_dyn_model.quadruped_wb_dynamics, model, mjx_model, contact_id, body_id, n_joints, dt)
        return _factory

    @property
    def hessian_approx(self):
        return None

    @property
    def W(self):
        Qp    = jnp.diag(jnp.array([0, 0, 1e4]))
        Qrot  = jnp.diag(jnp.array([1000, 1000, 0]))
        Qq    = jnp.diag(jnp.ones(self.n_joints)) * 1e-1
        Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 5e3
        Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e2
        Qdq   = jnp.diag(jnp.ones(self.n_joints)) * 1e-1
        Qtau  = jnp.diag(jnp.ones(self.n_joints)) * 1e-2
        Q_grf = jnp.diag(jnp.ones(3 * self.n_contact)) * 1e-3
        Qleg = jnp.diag(jnp.tile(jnp.array([1e4, 1e4, 1e5]), self.n_contact))
        return jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Q_grf)


class SpotBalance(_SpotCommon):
    """Reduced-support balance; gait timer is bypassed."""

    behaviour: str = "balance"
    swing_tracking: bool = True
    use_balance_fixed_contact: bool = True

    use_tripod_nominal_foot_ref: bool = True
    tripod_foot_ref_sigma = jnp.array([0.03, 0.03, 0.005])

    @property
    def dynamics(self):
        n_joints = self.n_joints
        dt = self.dt
        def _factory(model, mjx_model, contact_id, body_id):
            return partial(mpc_dyn_model.quadruped_wb_dynamics, model, mjx_model, contact_id, body_id, n_joints, dt)
        return _factory

    @property
    def cost(self):
        return partial(mpc_objectives.quadruped_wb_obj, True, self.n_joints, self.n_contact, self.N)

    @property
    def hessian_approx(self):
        return None

    robot_height: float = 0.46
    p0 = jnp.array([0, 0, robot_height])

    timer_t = jnp.zeros(4)
    duty_factor: float = 1.0
    step_freq: float = 1.0
    step_height: float = 0.0
    initial_height: float = 0.46

    use_terrain_estimation: bool = False

    @property
    def W(self):
        Qp    = jnp.diag(jnp.array([9e2, 9e2, 1.2e4]))
        Qrot  = jnp.diag(jnp.array([2200.0, 2200.0, 2200.0]))
        Qq    = jnp.diag(jnp.ones(self.n_joints)) * 1e2
        Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 8e3
        Qomega= jnp.diag(jnp.array([1, 1, 1])) * 3e2
        Qdq   = jnp.diag(jnp.ones(self.n_joints)) * 1e0
        Qtau  = jnp.diag(jnp.ones(self.n_joints)) * 1e-1
        Q_grf = jnp.diag(jnp.ones(3 * self.n_contact)) * 1e-2
        Qleg = jnp.diag(jnp.tile(jnp.array([7e3, 7e3, 6e4]), self.n_contact))
        return jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Q_grf)

    def __init__(self, balance_stance: str | None = None) -> None:
        stance = BalanceStance.FOUR if balance_stance is None else balance_stance
        self.balance_stance = stance
        self.balance_fixed_contact_mask = balance_stance_to_mask(stance)


def spot_config(
    mode: str = SpotMode.LOCOMOTION,
    *,
    balance_stance: str | None = None,
) -> _SpotCommon:
    """
    Parameters
    ----------
    mode
        ``SpotMode.LOCOMOTION`` / ``SpotMode.BALANCE`` or the equivalent strings.
    balance_stance
        Balance mode only. Same mask tags as Go2; hind slots are Spot ``HL`` / ``HR``.
        Ignored when ``mode`` is locomotion.
    """
    key = mode.lower().strip()
    if key == SpotMode.LOCOMOTION:
        return SpotLocomotion()
    if key == SpotMode.BALANCE:
        return SpotBalance(balance_stance=balance_stance)
    raise ValueError(
        f"Unknown Spot behaviour {mode!r}; expected {SpotMode.LOCOMOTION!r} or {SpotMode.BALANCE!r}."
    )


config = SpotLocomotion()
