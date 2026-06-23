"""
Go2 whole-body MPC configuration.

Pick a behaviour (same attribute layout ``MPCControllerWrapper`` expects)::

    from mpx.config.config_go2 import go2_config, Go2Mode, BalanceStance

    # Balance: pass ``balance_stance`` to set nominal MPC contact support:
    #   • ``BalanceStance.FOUR`` — all four feet in stance
    #   • ``BalanceStance.TRIPOD_SWING_<LEG>`` — three stance feet, one nominal swing (FL/FR/RL/RR)
    #   • ``BalanceStance.DIAG_FL_RR`` / ``DIAG_FR_RL`` — two diagonal stance feet (biped-style)
    cfg = go2_config(Go2Mode.BALANCE, balance_stance=BalanceStance.TRIPOD_SWING_FR)
    cfg = go2_config(Go2Mode.BALANCE, balance_stance=BalanceStance.DIAG_FL_RR)

Or construct directly: ``Go2Locomotion()``, ``Go2Balance(BalanceStance.TRIPOD_SWING_FR)``.

The module exposes ``config`` as locomotion defaults for backward compatibility.
"""
from __future__ import annotations

import os
from functools import partial
import jax
import jax.numpy as jnp

import mpx.utils.quadruped_dyn_models.models as mpc_dyn_model
import mpx.utils.quadruped_dyn_models.objectives as mpc_objectives  

_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(_DIR, "..", "..", "data", "go2", "go2_mjx.xml")
)

# Contact bitmask order follows ``contact_frame``: FL, FR, RL, RR (1 = in stance).


class Go2Mode:
    """String tags for ``go2_config(...)``."""

    LOCOMOTION = "locomotion"
    BALANCE = "balance"

#===========================================================
# region _Go2Common
class _Go2Common:
    """MuJoCo model topology and MPC dimensions shared by all Go2 behaviours."""

    behaviour: str = ""
    model_path: str = _DEFAULT_MODEL_PATH
    contact_frame = ["FL", "FR", "RL", "RR"]
    body_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

    dt: float = 0.02
    N: int = 25
    mpc_frequency: int = 50

    quat0 = jnp.array([1, 0, 0, 0])
    q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
    q0_init = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
    p_legs0 = jnp.array([ 0.192, 0.142, .0,  # Initial position of the front left leg
                          0.192, -0.142, .0, # Initial position of the front right leg
                         -0.195, 0.142, .0,  # Initial position of the rear left leg
                         -0.195, -0.142, .0  # Initial position of the rear right leg
                         ])

    #n_joints: int = 12
    grf_as_state: bool = True
    u_ref = jnp.zeros(12)

    use_balance_fixed_contact: bool = False
    balance_fixed_contact_mask = jnp.ones(4, dtype=jnp.float32)

    # Default dynamics = locomotion model; ``Go2Balance`` overrides with
    # ``quadruped_wb_dynamics_balance``.
    #dynamics = staticmethod(mpc_dyn_model.quadruped_wb_dynamics)
    max_torque: float = 25.0
    min_torque: float = -25.0

    #swing_tracking: bool = True
    
    @property
    def n_joints(self) -> int: # number of joints (12)
        return 12
    
    @property
    def n_contact(self) -> int: # number of contact points (FL, FR, RL, RR)
        return len(self.contact_frame)

    @property
    def n(self) -> int: # number of states (theta1, theta1_dot, theta2, theta2_dot)
        return 13 + 2 * self.n_joints + 6 * self.n_contact

    @property
    def m(self) -> int: # number of controls (F)
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
#endregion
#===========================================================
# region Go2Locomotion
class Go2Locomotion(_Go2Common):
    """Trot gait, terrain estimator on, lateral base position softly unconstrained."""

    behaviour: str = "locomotion"
    swing_tracking: bool = True

    robot_height: float = 0.27
    p0 = jnp.array([0, 0, robot_height])

    # change the gait pattern
    timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # trot; 
    #timer_t = jnp.array([0.5, 0.0, 0.5, 0.0])  # pace
    #timer_t = jnp.array([0.25, 0.75, 0.0, 0.5])  # crawl
    #timer_t = jnp.array([0.5, 0.5, 0.0, 0.0])  # bound (not reliable)

    duty_factor: float = 0.65
    step_freq: float = 1.35
    step_height: float = 0.18
    initial_height: float = 0.27

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
    def W(self):        # define weights for the MPC controller-------
        # Cost matrices (diagonal matrices created using jnp.diag)
        Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
        Qrot  = jnp.diag(jnp.array([1000, 1000, 0]))  # Cost matrix for rotation
        Qq    = jnp.diag(jnp.ones(self.n_joints)) * 1e-1  # Cost matrix for joint angles
        Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 5e3  # Cost matrix for position derivatives
        Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e2  # Cost matrix for angular velocity
        Qdq   = jnp.diag(jnp.ones(self.n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
        Qtau  = jnp.diag(jnp.ones(self.n_joints)) * 1e-1  # Cost matrix for torques
        Q_grf = jnp.diag(jnp.ones(3 * self.n_contact)) * 1e-2  # Cost matrix for ground reaction forces

        # For the leg contact cost, repeat the unit cost for each contact point.
        Qleg = jnp.diag(jnp.tile(jnp.array([1e4, 1e4, 1e5]), self.n_contact))

        # Combine all cost matrices into a block diagonal matrix
        return jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau,Q_grf)
    #------------------------------------------------

# region Go2Balance
class Go2Balance(_Go2Common):
    """
    Reduced-support balance: gait timer is bypassed; nominal contacts follow
    ``balance_fixed_contact_mask`` from ``balance_stance_to_mask`` (``BalanceStance``).

    Supported families (see ``BalanceStance`` and ``go2_config(..., balance_stance=...)``):

    * **Four-foot** — ``BalanceStance.FOUR`` (default if ``balance_stance`` is omitted).
    * **Tripod** — ``TRIPOD_SWING_FL`` … ``TRIPOD_SWING_RR`` (three stance feet, one nominal swing).
    * **Diagonal two-foot** — ``DIAG_FL_RR`` or ``DIAG_FR_RL``.

    ``swing_tracking=False`` disables foot-tracking cost on swinging feet.
    Use near-zero reference twist in MPC input for stationary balance.

    The ``W`` cost weights below are tuned with tripod-style support in mind; they are shared
    across all ``balance_stance`` presets unless you subclass and override ``W``.

    **Why ``n_contact`` stays 4 for every balance preset:** the Go2 whole-body model always
    stacks **four feet** (FL, FR, RL, RR) in the state and in ``_block_W`` — three XYZ blocks per
    foot for foot position cost and three per foot for GRF cost. Tripod / diagonal stances do
    **not** remove legs from the vector; ``balance_fixed_contact_mask`` (and measured contact)
    tells the dynamics and costs which legs are nominal **stance** vs **swing** (mask 0 turns off
    nominal contact/reaction for that foot). A smaller ``n_contact`` would require a different
    model layout (not supported here).
    """

    #n_joints: int = 12
    #n_contact: int = 4

    behaviour: str = "balance"
    swing_tracking: bool = True # this is the flag for the foot tracking cost
    use_balance_fixed_contact: bool = True

    # Tripod: world foot ref from random base-frame sample around ``p_legs0`` (resampled on MPC reset).
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

    robot_height: float = 0.27 # this is the nominal height of the robot
    p0 = jnp.array([0, 0, robot_height]) # this is the nominal position of the robot

    timer_t = jnp.zeros(4)
    duty_factor: float = 1.0
    step_freq: float = 1.0
    step_height: float = 0.0
    initial_height: float = 0.27


    use_terrain_estimation: bool = False

    # define weights for the MPC controller-------
    @property
    def W(self):
        # Cost matrices (diagonal matrices created using jnp.diag)
        Qp    = jnp.diag(jnp.array([9e2, 9e2, 1.2e4]))  # Cost matrix for position
        Qrot  = jnp.diag(jnp.array([2200.0, 2200.0, 2200.0]))  # Cost matrix for rotation
        Qq    = jnp.diag(jnp.ones(self.n_joints)) * 1e2  # Cost matrix for joint angles
        Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 8e3  # Cost matrix for position derivatives
        Qomega= jnp.diag(jnp.array([1, 1, 1])) * 3e2  # Cost matrix for angular velocity
        Qdq   = jnp.diag(jnp.ones(self.n_joints)) * 1e0  # Cost matrix for joint angle derivatives
        Qtau  = jnp.diag(jnp.ones(self.n_joints)) * 1e-1  # Cost matrix for torques
        Q_grf = jnp.diag(jnp.ones(3 * self.n_contact)) * 1e-2  # Cost matrix for ground reaction forces

        # For the leg contact cost, repeat the unit cost for each contact point.
        Qleg = jnp.diag(jnp.tile(jnp.array([7e3, 7e3, 6e4]), self.n_contact))

        # Combine all cost matrices into a block diagonal matrix
        W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau,Q_grf)
        return W
    #------------------------------------------------
    def __init__(self, balance_stance: str | None = None) -> None:
        """
        Parameters
        ----------
        balance_stance
            One of ``BalanceStance.FOUR``, ``TRIPOD_SWING_*``, ``THREE_MISSING_*`` (legacy tripod),
            or ``DIAG_FL_RR`` / ``DIAG_FR_RL``. If ``None``, defaults to ``FOUR``.
        """
        stance = BalanceStance.FOUR if balance_stance is None else balance_stance
        self.balance_stance = stance
        self.balance_fixed_contact_mask = balance_stance_to_mask(stance)

# endregion

# region BalanceStance
class BalanceStance:
    """
    Nominal MPC contact masks (bits in ``contact_frame`` order FL, FR, RL, RR).

    **Tripod presets** (`TRIPOD_SWING_<LEG>`): that leg is modeled with **zero** nominal
    contact / reaction; the **other three** are stance feet. Naming is intentional:
    `"three_missing_RL"` sounded like bulk legs detaching—in reality only that leg should
    be *nominal swing* in the QP; peers can still peel in MuJoCo if the stance polygon,
    friction, torque limits, or tracking make the trajectory roll.

    **Diagonal presets** (`DIAG_FL_RR`, `DIAG_FR_RL`): two diagonal feet carry nominal
    contact / reaction; the other two are modeled as swing (zero nominal reaction).

    Prefer ``TRIPOD_SWING_*`` for clarity; ``THREE_MISSING_*`` remain as legacy string tags.
    """

    FOUR = "four"

    TRIPOD_SWING_FL = "tripod_swing_fl"
    TRIPOD_SWING_FR = "tripod_swing_fr"
    TRIPOD_SWING_RL = "tripod_swing_rl"
    TRIPOD_SWING_RR = "tripod_swing_rr"

    THREE_MISSING_FL = "three_missing_fl"
    THREE_MISSING_FR = "three_missing_fr"
    THREE_MISSING_RL = "three_missing_rl"
    THREE_MISSING_RR = "three_missing_rr"

    DIAG_FL_RR = "diag_fl_rr"
    DIAG_FR_RL = "diag_fr_rl"


def balance_stance_to_mask(stance: str) -> jnp.ndarray:
    s = stance.lower().strip().replace("-", "_")
    fl, fr, rl, rr = 1.0, 1.0, 1.0, 1.0
    # tripod: index 0 in mask means FL has no MPC reaction, etc.
    tripod_fl = (0.0, fr, rl, rr)
    tripod_fr = (fl, 0.0, rl, rr)
    tripod_rl = (fl, fr, 0.0, rr)
    tripod_rr = (fl, fr, rl, 0.0)
    masks: dict[str, tuple[float, float, float, float]] = {
        BalanceStance.FOUR: (fl, fr, rl, rr),
        BalanceStance.TRIPOD_SWING_FL: tripod_fl,
        BalanceStance.TRIPOD_SWING_FR: tripod_fr,
        BalanceStance.TRIPOD_SWING_RL: tripod_rl,
        BalanceStance.TRIPOD_SWING_RR: tripod_rr,
        BalanceStance.THREE_MISSING_FL: tripod_fl,
        BalanceStance.THREE_MISSING_FR: tripod_fr,
        BalanceStance.THREE_MISSING_RL: tripod_rl,
        BalanceStance.THREE_MISSING_RR: tripod_rr,
        BalanceStance.DIAG_FL_RR: (fl, 0.0, 0.0, rr),
        BalanceStance.DIAG_FR_RL: (0.0, fr, rl, 0.0),
    }
    try:
        t = masks[s]
    except KeyError as e:
        known = ", ".join(sorted(masks.keys()))
        raise ValueError(f"Unknown balance stance {stance!r}; expected one of: {known}") from e
    return jnp.array(t, dtype=jnp.float32)

#endregion



def go2_config(
    mode: str = Go2Mode.LOCOMOTION,
    *,
    balance_stance: str | None = None,
) -> _Go2Common:
    """
    Parameters
    ----------
    mode
        ``Go2Mode.LOCOMOTION`` / ``Go2Mode.BALANCE`` or the equivalent strings.
    balance_stance
        Balance mode only. Nominal MPC contact mask (FL, FR, RL, RR order):

        * ``BalanceStance.FOUR`` — four feet in stance (default when omitted).
        * ``BalanceStance.TRIPOD_SWING_FL`` … ``TRIPOD_SWING_RR`` — three feet in stance, one nominal swing.
        * ``BalanceStance.THREE_MISSING_*`` — same masks as tripod (legacy names).
        * ``BalanceStance.DIAG_FL_RR`` / ``DIAG_FR_RL`` — two diagonal stance feet.

        Ignored when ``mode`` is locomotion.
    """
    key = mode.lower().strip()
    if key == Go2Mode.LOCOMOTION:
        return Go2Locomotion()
    if key == Go2Mode.BALANCE:
        return Go2Balance(balance_stance=balance_stance)
    raise ValueError(
        f"Unknown Go2 behaviour {mode!r}; expected {Go2Mode.LOCOMOTION!r} or {Go2Mode.BALANCE!r}."
    )


# Backward-compatible default (same content as historical module-level globals)
config = Go2Locomotion()
