"""
Unitree B2 whole-body MPC configuration.

Pick a behaviour (same attribute layout ``MPCControllerWrapper`` expects)::

    from mpx.config.robot_config.config_b2 import b2_config, B2Mode, B2Gait, BalanceStance

    # Locomotion: pass ``gait`` to pick the gait and its whole parameter set.
    #   • ``B2Gait.TROT``  — diagonal pairs (default)
    #   • ``B2Gait.PACE``  — lateral pairs
    #   • ``B2Gait.CRAWL`` — one leg at a time, three feet always down
    #   • ``B2Gait.BOUND`` — front pair / hind pair (experimental)
    cfg = b2_config(B2Mode.LOCOMOTION, gait=B2Gait.CRAWL)

    # Balance: pass ``balance_stance`` to set nominal MPC contact support:
    cfg = b2_config(B2Mode.BALANCE, balance_stance=BalanceStance.TRIPOD_SWING_FR)
    cfg = b2_config(B2Mode.BALANCE, balance_stance=BalanceStance.DIAG_FL_RR)

Or construct directly: ``B2Locomotion(B2Gait.PACE)``, ``B2Balance(BalanceStance.TRIPOD_SWING_FR)``.

Gait parameters live in one place, :data:`B2_GAITS`. The module exposes
``config`` as locomotion defaults (trot) for backward compatibility.

Contact / body names follow the MuJoCo model in ``mpx/data/b2/b2.xml``:
feet ``FL, FR, RL, RR`` and calf bodies ``FL_calf, FR_calf, RL_calf, RR_calf``.
"""
from __future__ import annotations

import os
from functools import partial

import jax.numpy as jnp

import mpx.utils.quadruped_dyn_models.models as mpc_dyn_model
import mpx.utils.quadruped_dyn_models.objectives as mpc_objectives
from mpx.config.robot_config.config_go2 import (
    BalanceStance,
    GaitParams,
    LocomotionWeights,
    balance_stance_to_mask,
)

_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(_DIR, "..", "..", "data", "b2", "b2.xml")
)

# Contact bitmask order follows ``contact_frame``: FL, FR, RL, RR (1 = in stance).
_B2_HEIGHT = 0.485  # home keyframe base height in ``b2.xml``


class B2Mode:
    """String tags for ``b2_config(...)``."""

    LOCOMOTION = "locomotion"
    BALANCE = "balance"


class B2Gait:
    """String tags for the locomotion gaits registered in :data:`B2_GAITS`."""

    TROT = "trot"
    PACE = "pace"
    CRAWL = "crawl"
    BOUND = "bound"


#===========================================================
# region gait parameter sets
# Same phase / duty / frequency layout as Go2. The B2 is ~83 kg with hip/thigh
# motors at ±200 Nm and calves at ±300 Nm, so torque / GRF penalties stay in
# the Go2 ballpark rather than Spot's lighter ones.
_B2_LOCOMOTION_WEIGHTS = LocomotionWeights()


B2_GAITS: dict[str, GaitParams] = {
    B2Gait.TROT: GaitParams(
        name=B2Gait.TROT,
        phase_offsets=(0.5, 0.0, 0.0, 0.5),
        duty_factor=0.65,
        step_freq=1.35,
        step_height=0.14,
        robot_height=_B2_HEIGHT,
        weights=_B2_LOCOMOTION_WEIGHTS,
        description="Diagonal pairs (FL+RR / FR+RL). Default; most robust.",
    ),
    B2Gait.PACE: GaitParams(
        name=B2Gait.PACE,
        phase_offsets=(0.5, 0.0, 0.5, 0.0),
        duty_factor=0.70,
        step_freq=1.55,
        step_height=0.10,
        clearance_speed=0.2,
        robot_height=_B2_HEIGHT,
        weights=LocomotionWeights(
            rot=(2200.0, 1000.0, 0.0),
            ang_vel=2e2,
        ),
        description="Lateral pairs (FL+RL / FR+RR). Roll-unstable; low, quick steps.",
    ),
    B2Gait.CRAWL: GaitParams(
        name=B2Gait.CRAWL,
        phase_offsets=(0.25, 0.75, 0.0, 0.5),
        duty_factor=0.80,
        step_freq=1.00,
        step_height=0.13,
        robot_height=_B2_HEIGHT,
        weights=LocomotionWeights(
            rot=(1500.0, 1500.0, 0.0),
            foot=(2e4, 2e4, 1e5),
        ),
        description="One leg at a time; three feet always down. Statically stable, slow.",
    ),
    B2Gait.BOUND: GaitParams(
        name=B2Gait.BOUND,
        phase_offsets=(0.5, 0.5, 0.0, 0.0),
        duty_factor=0.78,
        step_freq=1.55,
        step_height=0.12,
        clearance_speed=0.3,
        robot_height=_B2_HEIGHT,
        weights=LocomotionWeights(
            pos=(0.0, 0.0, 2e4),
            rot=(1000.0, 4500.0, 0.0),
            ang_vel=3e2,
            lin_vel=4e3,
        ),
        description="Front pair / hind pair. Pitch-unstable; conservative duty.",
    ),
}


DEFAULT_GAIT: str = B2Gait.TROT


def b2_gait_params(gait: str | GaitParams | None = None) -> GaitParams:
    """Look up a :class:`GaitParams` by tag. ``None`` gives :data:`DEFAULT_GAIT`.

    A :class:`GaitParams` instance passes through, so a caller can hand in a
    one-off ``replace(B2_GAITS["trot"], step_freq=1.5)`` without registering it.
    """
    if gait is None:
        return B2_GAITS[DEFAULT_GAIT]
    if isinstance(gait, GaitParams):
        return gait
    key = str(gait).lower().strip().replace("-", "_")
    try:
        return B2_GAITS[key]
    except KeyError as e:
        known = ", ".join(sorted(B2_GAITS))
        raise ValueError(f"Unknown B2 gait {gait!r}; expected one of: {known}") from e
#endregion
#===========================================================
# region _B2Common
class _B2Common:
    """MuJoCo model topology and MPC dimensions shared by all B2 behaviours."""

    behaviour: str = ""
    model_path: str = _DEFAULT_MODEL_PATH
    contact_frame = ["FL", "FR", "RL", "RR"]
    body_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    base_body_name: str = "base"

    dt: float = 0.02
    N: int = 25
    mpc_frequency: int = 50
    solver_mode = "primal_dual"

    quat0 = jnp.array([1, 0, 0, 0])
    q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
    q0_init = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
    p_legs0 = jnp.array([
        0.3285, 0.192, 0.0,
        0.3285, -0.192, 0.0,
        -0.3285, 0.192, 0.0,
        -0.3285, -0.192, 0.0,
    ])

    grf_as_state: bool = True
    u_ref = jnp.zeros(12)

    use_balance_fixed_contact: bool = False
    balance_fixed_contact_mask = jnp.ones(4, dtype=jnp.float32)

    # Hip / thigh motors ±200 Nm; calf ±300 Nm. Use the tighter bound.
    max_torque: float = 200.0
    min_torque: float = -200.0

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
# region B2Locomotion
class B2Locomotion(_B2Common):
    """
    Terrain estimator on, lateral base position softly unconstrained.

    Gait timing, swing geometry, nominal height and cost weights all come from
    one :class:`GaitParams` entry — see :data:`B2_GAITS`::

        B2Locomotion()                  # trot
        B2Locomotion(B2Gait.CRAWL)      # crawl
        B2Locomotion(replace(B2_GAITS["trot"], step_freq=1.5))   # one-off tweak
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
            ``B2Gait.TROT`` / ``PACE`` / ``CRAWL`` / ``BOUND``, a ready-made
            :class:`GaitParams`, or ``None`` for :data:`DEFAULT_GAIT`.
        """
        params = b2_gait_params(gait)
        self.gait = params
        self.gait_name: str = params.name

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
        return f"B2Locomotion({self.gait.summary()})"
#endregion
#===========================================================
# region B2Balance

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


class B2Balance(_B2Common):
    """
    Reduced-support balance: gait timer is bypassed; nominal contacts follow
    ``balance_fixed_contact_mask`` from ``balance_stance_to_mask`` (``BalanceStance``).
    """

    behaviour: str = "balance"
    swing_tracking: bool = True
    use_balance_fixed_contact: bool = True

    use_tripod_nominal_foot_ref: bool = True
    tripod_foot_ref_sigma = jnp.array([0.03, 0.03, 0.005])

    robot_height: float = _B2_HEIGHT
    p0 = jnp.array([0, 0, robot_height])
    initial_height: float = _B2_HEIGHT

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


def b2_config(
    mode: str = B2Mode.LOCOMOTION,
    *,
    gait: str | GaitParams | None = None,
    balance_stance: str | None = None,
) -> _B2Common:
    """
    Parameters
    ----------
    mode
        ``B2Mode.LOCOMOTION`` / ``B2Mode.BALANCE`` or the equivalent strings.
    gait
        Locomotion mode only. Gait tag or :class:`GaitParams`.
    balance_stance
        Balance mode only. Same mask tags as Go2.
    """
    key = mode.lower().strip()
    if key == B2Mode.LOCOMOTION:
        return B2Locomotion(gait)
    if key == B2Mode.BALANCE:
        return B2Balance(balance_stance=balance_stance)
    raise ValueError(
        f"Unknown B2 behaviour {mode!r}; expected {B2Mode.LOCOMOTION!r} or {B2Mode.BALANCE!r}."
    )


config = B2Locomotion()
