"""
Boston Dynamics Spot whole-body MPC configuration.

Pick a behaviour (same attribute layout ``MPCControllerWrapper`` expects)::

    from framework.config.robot_config.config_spot import spot_config, SpotMode, SpotGait, BalanceStance

    # Locomotion: pass ``gait`` to pick the gait and its whole parameter set.
    #   • ``SpotGait.TROT``  — diagonal pairs (default)
    #   • ``SpotGait.PACE``  — lateral pairs
    #   • ``SpotGait.CRAWL`` — one leg at a time, three feet always down
    #   • ``SpotGait.BOUND`` — front pair / hind pair (experimental)
    cfg = spot_config(SpotMode.LOCOMOTION, gait=SpotGait.CRAWL)

    # Balance: pass ``balance_stance`` to set nominal MPC contact support:
    cfg = spot_config(SpotMode.BALANCE, balance_stance=BalanceStance.TRIPOD_SWING_FR)
    cfg = spot_config(SpotMode.BALANCE, balance_stance=BalanceStance.DIAG_FL_RR)

Or construct directly: ``SpotLocomotion(SpotGait.PACE)``, ``SpotBalance(BalanceStance.TRIPOD_SWING_FR)``.

Gait parameters live in one place, :data:`SPOT_GAITS`, so changing a gait means
editing one :class:`GaitParams` entry instead of uncommenting a ``timer_t`` line
and hand-matching ``duty_factor`` / ``step_freq`` / ``step_height`` to it. Each
gait carries its own MPC cost weights (:class:`LocomotionWeights`) too, because a
pace and a bound fail in different axes and want different penalties.

The module exposes ``config`` as locomotion defaults (trot) for backward compatibility.

Contact / body names follow the MuJoCo model in
``framework/data/boston_dynamics_spot/spot.xml``: feet ``FL, FR, HL, HR`` and
lower-leg bodies ``fl_lleg, fr_lleg, hl_lleg, hr_lleg``. Balance-stance tags
still use Go2's ``RL`` / ``RR`` names; those map to the hind-left / hind-right
slots (indices 2 and 3).
"""
from __future__ import annotations

import os
from functools import partial

import jax.numpy as jnp

import framework.utils.quadruped_dyn_models.models as mpc_dyn_model
import framework.utils.quadruped_dyn_models.objectives as mpc_objectives
from framework.config.robot_config.config_go2 import (
    BalanceStance,
    GaitParams,
    LocomotionWeights,
    balance_stance_to_mask,
)

_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(_DIR, "..", "..", "data", "boston_dynamics_spot", "spot.xml")
)

# Contact bitmask order follows ``contact_frame``: FL, FR, HL, HR (1 = in stance).


class SpotMode:
    """String tags for ``spot_config(...)``."""

    LOCOMOTION = "locomotion"
    BALANCE = "balance"


class SpotGait:
    """String tags for the locomotion gaits registered in :data:`SPOT_GAITS`."""

    TROT = "trot"
    PACE = "pace"
    CRAWL = "crawl"
    BOUND = "bound"


#===========================================================
# region gait parameter sets
# Spot-tuned defaults: same layout as Go2, but this robot is taller and the
# hip/thigh torque limits are higher, so the historical locomotion ``W`` used
# slightly lighter torque / GRF penalties than Go2.
_SPOT_LOCOMOTION_WEIGHTS = LocomotionWeights(
    torque=1e-2,
    grf=1e-3,
)


SPOT_GAITS: dict[str, GaitParams] = {
    # Diagonal pairs. Two feet down for 30% of the cycle on either side of each
    # swap, which is what makes it the robust default.
    SpotGait.TROT: GaitParams(
        name=SpotGait.TROT,
        phase_offsets=(0.5, 0.0, 0.0, 0.5),
        duty_factor=0.65,
        step_freq=1.35,
        step_height=0.12,
        robot_height=0.46,
        weights=_SPOT_LOCOMOTION_WEIGHTS,
        description="Diagonal pairs (FL+HR / FR+HL). Default; most robust.",
    ),
    # Lateral pairs. Both feet on one side leave the ground together, so the
    # roll axis is the one that loses the robot: step faster, stay down longer,
    # keep the feet low, and weight roll harder than trot does.
    SpotGait.PACE: GaitParams(
        name=SpotGait.PACE,
        phase_offsets=(0.5, 0.0, 0.5, 0.0),
        duty_factor=0.70,
        step_freq=1.55,
        step_height=0.08,
        clearance_speed=0.2,
        robot_height=0.46,
        weights=LocomotionWeights(
            rot=(2200.0, 1000.0, 0.0),
            ang_vel=2e2,
            torque=1e-2,
            grf=1e-3,
        ),
        description="Lateral pairs (FL+HL / FR+HR). Roll-unstable; low, quick steps.",
    ),
    # One leg at a time, evenly spaced a quarter cycle apart. duty >= 0.75 is
    # what keeps three feet down at all times and makes this statically stable;
    # 0.80 leaves margin so an early touchdown never drops support to two.
    SpotGait.CRAWL: GaitParams(
        name=SpotGait.CRAWL,
        phase_offsets=(0.25, 0.75, 0.0, 0.5),
        duty_factor=0.80,
        step_freq=1.00,
        step_height=0.11,
        robot_height=0.46,
        weights=LocomotionWeights(
            rot=(1500.0, 1500.0, 0.0),
            foot=(2e4, 2e4, 1e5),
            torque=1e-2,
            grf=1e-3,
        ),
        description="One leg at a time; three feet always down. Statically stable, slow.",
    ),
    # Front pair / hind pair. Pitch is the failure axis: in a two-pair gait the
    # support during swing is two feet on a single lateral line.
    SpotGait.BOUND: GaitParams(
        name=SpotGait.BOUND,
        phase_offsets=(0.5, 0.5, 0.0, 0.0),
        duty_factor=0.78,
        step_freq=1.55,
        step_height=0.10,
        clearance_speed=0.3,
        robot_height=0.46,
        weights=LocomotionWeights(
            pos=(0.0, 0.0, 2e4),
            rot=(1000.0, 4500.0, 0.0),
            ang_vel=3e2,
            lin_vel=4e3,
            torque=1e-2,
            grf=1e-3,
        ),
        description="Front pair / hind pair. Pitch-unstable; conservative duty.",
    ),
}


DEFAULT_GAIT: str = SpotGait.TROT


def spot_gait_params(gait: str | GaitParams | None = None) -> GaitParams:
    """Look up a :class:`GaitParams` by tag. ``None`` gives :data:`DEFAULT_GAIT`.

    A :class:`GaitParams` instance passes through, so a caller can hand in a
    one-off ``replace(SPOT_GAITS["trot"], step_freq=1.5)`` without registering it.
    """
    if gait is None:
        return SPOT_GAITS[DEFAULT_GAIT]
    if isinstance(gait, GaitParams):
        return gait
    key = str(gait).lower().strip().replace("-", "_")
    try:
        return SPOT_GAITS[key]
    except KeyError as e:
        known = ", ".join(sorted(SPOT_GAITS))
        raise ValueError(f"Unknown Spot gait {gait!r}; expected one of: {known}") from e
#endregion
#===========================================================
# region _SpotCommon
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
    solver_mode = "primal_dual"

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

    @property
    def W(self) -> jnp.ndarray:
        """MPC cost matrix, assembled from ``self.weights``."""
        return self.weights.matrix(self.n_joints, self.n_contact)
#endregion
#===========================================================
# region SpotLocomotion
class SpotLocomotion(_SpotCommon):
    """
    Terrain estimator on, lateral base position softly unconstrained.

    Gait timing, swing geometry, nominal height and cost weights all come from
    one :class:`GaitParams` entry — see :data:`SPOT_GAITS`::

        SpotLocomotion()                  # trot
        SpotLocomotion(SpotGait.CRAWL)    # crawl
        SpotLocomotion(replace(SPOT_GAITS["trot"], step_freq=1.5))   # one-off tweak

    ``self.gait`` keeps the full parameter set around, so a caller can read
    ``cfg.gait.swing_time`` or ``cfg.gait.summary()`` instead of recomputing it.
    """

    behaviour: str = "locomotion"
    swing_tracking: bool = True

    use_terrain_estimation: bool = True
    solver_mode = "primal_dual"

    def __init__(self, gait: str | GaitParams | None = None) -> None:
        """
        Parameters
        ----------
        gait
            ``SpotGait.TROT`` / ``PACE`` / ``CRAWL`` / ``BOUND``, a ready-made
            :class:`GaitParams`, or ``None`` for :data:`DEFAULT_GAIT`.
        """
        params = spot_gait_params(gait)
        self.gait = params
        self.gait_name: str = params.name

        # Flattened onto the config because ``MPCControllerWrapper`` reads these
        # names directly off it; ``self.gait`` stays the single source of truth.
        self.timer_t = params.timer_t
        self.duty_factor: float = params.duty_factor
        self.step_freq: float = params.step_freq
        self.step_height: float = params.step_height
        self.clearance_speed: float = params.clearance_speed
        self.weights: LocomotionWeights = params.weights

        self.robot_height: float = params.robot_height
        self.initial_height: float = params.robot_height
        self.p0 = jnp.array([0.0, 0.0, params.robot_height])

    def __repr__(self) -> str:
        return f"SpotLocomotion({self.gait.summary()})"
#endregion
#===========================================================
# region SpotBalance

BALANCE_WEIGHTS = LocomotionWeights(
    pos=(9e2, 9e2, 1.2e4),
    rot=(2200.0, 2200.0, 2200.0),
    joint_pos=1e2,
    lin_vel=8e3,
    ang_vel=3e2,
    joint_vel=1e0,
    torque=1e-1,
    grf=1e-2,
    foot=(7e3, 7e3, 6e4),
)


class SpotBalance(_SpotCommon):
    """
    Reduced-support balance: gait timer is bypassed; nominal contacts follow
    ``balance_fixed_contact_mask`` from ``balance_stance_to_mask`` (``BalanceStance``).

    Stance tags still use Go2 ``RL`` / ``RR`` names; those map to Spot hind slots
    (``HL`` / ``HR``) at indices 2 and 3.
    """

    behaviour: str = "balance"
    swing_tracking: bool = True
    use_balance_fixed_contact: bool = True

    use_tripod_nominal_foot_ref: bool = True
    tripod_foot_ref_sigma = jnp.array([0.03, 0.03, 0.005])

    robot_height: float = 0.46
    p0 = jnp.array([0, 0, robot_height])
    initial_height: float = 0.46

    timer_t = jnp.zeros(4)
    duty_factor: float = 1.0
    step_freq: float = 1.0
    step_height: float = 0.0
    clearance_speed: float = 0.2

    use_terrain_estimation: bool = False

    def __init__(
        self,
        balance_stance: str | None = None,
        *,
        weights: LocomotionWeights | None = None,
    ) -> None:
        stance = BalanceStance.FOUR if balance_stance is None else balance_stance
        self.balance_stance = stance
        self.balance_fixed_contact_mask = balance_stance_to_mask(stance)
        self.weights = BALANCE_WEIGHTS if weights is None else weights
#endregion


def spot_config(
    mode: str = SpotMode.LOCOMOTION,
    *,
    gait: str | GaitParams | None = None,
    balance_stance: str | None = None,
) -> _SpotCommon:
    """
    Parameters
    ----------
    mode
        ``SpotMode.LOCOMOTION`` / ``SpotMode.BALANCE`` or the equivalent strings.
    gait
        Locomotion mode only. Gait tag or :class:`GaitParams`.
    balance_stance
        Balance mode only. Same mask tags as Go2; hind slots are Spot ``HL`` / ``HR``.
    """
    key = mode.lower().strip()
    if key == SpotMode.LOCOMOTION:
        return SpotLocomotion(gait)
    if key == SpotMode.BALANCE:
        return SpotBalance(balance_stance=balance_stance)
    raise ValueError(
        f"Unknown Spot behaviour {mode!r}; expected {SpotMode.LOCOMOTION!r} or {SpotMode.BALANCE!r}."
    )


config = SpotLocomotion()
