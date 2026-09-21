"""
Go2 whole-body MPC configuration.

Pick a behaviour (same attribute layout ``MPCControllerWrapper`` expects)::

    from mpx.config.robot_config.config_go2 import go2_config, Go2Mode, Go2Gait, BalanceStance

    # Locomotion: pass ``gait`` to pick the gait and its whole parameter set.
    #   • ``Go2Gait.TROT``  — diagonal pairs (default)
    #   • ``Go2Gait.PACE``  — lateral pairs
    #   • ``Go2Gait.CRAWL`` — one leg at a time, three feet always down
    #   • ``Go2Gait.BOUND`` — front pair / hind pair (experimental)
    cfg = go2_config(Go2Mode.LOCOMOTION, gait=Go2Gait.CRAWL)
    cfg = go2_config(Go2Mode.LOCOMOTION, mpc_model=Go2MpcModel.INVERSE_DYNAMICS)
    cfg = go2_config(Go2Mode.LOCOMOTION, mpc_model=Go2MpcModel.SRBD)

    # Balance: pass ``balance_stance`` to set nominal MPC contact support:
    #   • ``BalanceStance.FOUR`` — all four feet in stance
    #   • ``BalanceStance.TRIPOD_SWING_<LEG>`` — three stance feet, one nominal swing (FL/FR/RL/RR)
    #   • ``BalanceStance.DIAG_FL_RR`` / ``DIAG_FR_RL`` — two diagonal stance feet (biped-style)
    cfg = go2_config(Go2Mode.BALANCE, balance_stance=BalanceStance.TRIPOD_SWING_FR)
    cfg = go2_config(Go2Mode.BALANCE, balance_stance=BalanceStance.DIAG_FL_RR)

Or construct directly: ``Go2Locomotion(Go2Gait.PACE)``, ``Go2Balance(BalanceStance.TRIPOD_SWING_FR)``.

Gait parameters live in one place, :data:`GO2_GAITS`, so changing a gait means
editing one :class:`GaitParams` entry instead of uncommenting a ``timer_t`` line
and hand-matching ``duty_factor`` / ``step_freq`` / ``step_height`` to it. Each
gait carries its own MPC cost weights (:class:`LocomotionWeights`) too, because a
pace and a bound fail in different axes and want different penalties.

The module exposes ``config`` as locomotion defaults (trot) for backward compatibility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

import mpx.utils.quadruped_dyn_models.models as mpc_dyn_model
import mpx.utils.quadruped_dyn_models.objectives as mpc_objectives
from mpx.utils.quadruped_dyn_models.inverse import (
    inv_dyn_dims,
    inverse_dynamics_factories,
)

_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(_DIR, "..", "..", "data", "go2", "go2_mjx.xml")
)

# Contact bitmask order follows ``contact_frame``: FL, FR, RL, RR (1 = in stance).


class Go2Mode:
    """String tags for ``go2_config(...)``."""

    LOCOMOTION = "locomotion"
    BALANCE = "balance"


class Go2MpcModel:
    """Whole-body (default) vs inverse-dynamics vs centroidal SRBD transcription."""

    WHOLE_BODY = "whole_body"
    INVERSE_DYNAMICS = "inverse_dynamics"
    SRBD = "srbd"


class Go2Gait:
    """String tags for the locomotion gaits registered in :data:`GO2_GAITS`."""

    TROT = "trot"
    PACE = "pace"
    CRAWL = "crawl"
    BOUND = "bound"


#===========================================================
# region gait parameter sets
@dataclass(frozen=True)
class LocomotionWeights:
    """
    Diagonal MPC cost weights, per physical quantity instead of per matrix block.

    Assembled by :meth:`matrix` in the order ``quadruped_wb_obj`` expects:
    ``[p, rot, q, dp, omega, dq, foot, tau, grf]``. Defaults are the trot-tuned
    values this module has always used; other gaits override only what they need.
    """

    # Base position [x, y, z]. x/y are free in locomotion — the navigator owns
    # where the robot goes, the MPC only holds height.
    pos: tuple[float, float, float] = (0.0, 0.0, 1e4)
    # Base orientation [roll, pitch, yaw]. Yaw is free: the yaw rate command drives it.
    rot: tuple[float, float, float] = (1000.0, 1000.0, 0.0)
    joint_pos: float = 1e-1
    # Base linear velocity — this is what tracks the vx/vy command.
    lin_vel: float = 5e3
    ang_vel: float = 1e2
    joint_vel: float = 1e-1
    torque: float = 1e-1
    grf: float = 1e-2
    # Per-foot swing/stance tracking [x, y, z], repeated for each contact.
    foot: tuple[float, float, float] = (1e4, 1e4, 1e5)

    def matrix(self, n_joints: int, n_contact: int) -> jnp.ndarray:
        """Block-diagonal weight matrix ``W`` for ``n_joints`` / ``n_contact``."""
        Qp = jnp.diag(jnp.array(self.pos, dtype=jnp.float32))
        Qrot = jnp.diag(jnp.array(self.rot, dtype=jnp.float32))
        Qq = jnp.diag(jnp.ones(n_joints)) * self.joint_pos
        Qdp = jnp.diag(jnp.ones(3)) * self.lin_vel
        Qomega = jnp.diag(jnp.ones(3)) * self.ang_vel
        Qdq = jnp.diag(jnp.ones(n_joints)) * self.joint_vel
        Qleg = jnp.diag(jnp.tile(jnp.array(self.foot, dtype=jnp.float32), n_contact))
        Qtau = jnp.diag(jnp.ones(n_joints)) * self.torque
        Q_grf = jnp.diag(jnp.ones(3 * n_contact)) * self.grf
        return jax.scipy.linalg.block_diag(
            Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Q_grf
        )


@dataclass(frozen=True)
class GaitParams:
    """
    Everything that defines one locomotion gait.

    ``phase_offsets`` are the per-leg initial phases in ``contact_frame`` order
    (FL, FR, RL, RR), as fractions of one gait cycle. ``sim_utils.timer_run``
    advances every leg phase by ``step_freq * dt`` and calls a leg **stance**
    while its phase is below ``duty_factor`` — so two legs sharing an offset
    swing together, and ``duty_factor`` alone decides how long each leg is down.

    They must match ``dataset_bucket_system.GAIT_PHASE_OFFSETS`` within its 0.05
    tolerance, or episodes recorded with this gait get labelled ``TRANSITION``.
    """

    name: str
    phase_offsets: tuple[float, float, float, float]
    duty_factor: float
    step_freq: float          # [Hz], gait cycles per second
    step_height: float        # [m], apex of the swing arc above liftoff
    clearance_speed: float = 0.2   # [m/s], vertical liftoff bias in the swing spline
    robot_height: float = 0.27     # [m], commanded nominal base height
    weights: LocomotionWeights = field(default_factory=LocomotionWeights)
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.duty_factor < 1.0:
            raise ValueError(
                f"{self.name}: duty_factor must be in (0, 1), got {self.duty_factor}"
            )
        if self.step_freq <= 0.0:
            raise ValueError(
                f"{self.name}: step_freq must be > 0 Hz, got {self.step_freq}"
            )
        if self.step_height < 0.0:
            raise ValueError(
                f"{self.name}: step_height must be >= 0 m, got {self.step_height}"
            )
        if len(self.phase_offsets) != 4:
            raise ValueError(
                f"{self.name}: phase_offsets needs 4 entries (FL, FR, RL, RR), "
                f"got {len(self.phase_offsets)}"
            )
        if any(not 0.0 <= p < 1.0 for p in self.phase_offsets):
            raise ValueError(
                f"{self.name}: phase_offsets must lie in [0, 1), got {self.phase_offsets}"
            )

    # ------------------------------------------------------------ derived timing
    @property
    def timer_t(self) -> jnp.ndarray:
        """Initial per-leg phase vector consumed by the MPC gait timer."""
        return jnp.array(self.phase_offsets, dtype=jnp.float32)

    @property
    def cycle_time(self) -> float:
        """Duration of one full gait cycle [s]."""
        return 1.0 / self.step_freq

    @property
    def stance_time(self) -> float:
        """How long one leg stays down per cycle [s]."""
        return self.duty_factor / self.step_freq

    @property
    def swing_time(self) -> float:
        """How long one leg is in the air per cycle [s]. Below ~0.12 s the
        whole-body controller starts missing footholds on this robot."""
        return (1.0 - self.duty_factor) / self.step_freq

    def support_counts(self, samples: int = 400) -> tuple[int, int]:
        """``(min, max)`` number of stance feet over one cycle.

        A ``min`` of 0 means a flight phase: no foot on the ground, which the
        stance-force QP cannot hold the base with.
        """
        phases = (
            jnp.asarray(self.phase_offsets)[None, :]
            + jnp.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
        ) % 1.0
        counts = jnp.sum(phases < self.duty_factor, axis=1)
        return int(jnp.min(counts)), int(jnp.max(counts))

    def summary(self) -> str:
        """One-line human-readable description, handy in logs."""
        lo, hi = self.support_counts()
        return (
            f"{self.name}: duty={self.duty_factor:.2f} freq={self.step_freq:.2f}Hz "
            f"stance={self.stance_time*1e3:.0f}ms swing={self.swing_time*1e3:.0f}ms "
            f"height={self.step_height*1e2:.0f}cm support={lo}-{hi} feet"
        )


# Per-gait parameter sets. The trot entry reproduces the values this module
# shipped with; the other three are reasoned starting points (see each comment),
# not measured optima — retune them against your own runs before trusting them
# for dataset collection.
GO2_GAITS: dict[str, GaitParams] = {
    # Diagonal pairs. Two feet down for 30% of the cycle on either side of each
    # swap, which is what makes it the robust default.
    Go2Gait.TROT: GaitParams(
        name=Go2Gait.TROT,
        phase_offsets=(0.5, 0.0, 0.0, 0.5),
        duty_factor=0.65,
        step_freq=1.35,
        step_height=0.10,
        description="Diagonal pairs (FL+RR / FR+RL). Default; most robust.",
    ),
    # Lateral pairs. Both feet on one side leave the ground together, so the
    # roll axis is the one that loses the robot: step faster, stay down longer,
    # keep the feet low, and weight roll harder than trot does.
    Go2Gait.PACE: GaitParams(
        name=Go2Gait.PACE,
        phase_offsets=(0.5, 0.0, 0.5, 0.0),
        duty_factor=0.70,
        step_freq=1.55,
        step_height=0.07,
        clearance_speed=0.2,
        weights=LocomotionWeights(
            rot=(2200.0, 1000.0, 0.0),   # roll is the failure axis here
            ang_vel=2e2,
        ),
        description="Lateral pairs (FL+RL / FR+RR). Roll-unstable; low, quick steps.",
    ),
    # One leg at a time, evenly spaced a quarter cycle apart. duty >= 0.75 is
    # what keeps three feet down at all times and makes this statically stable;
    # 0.80 leaves margin so an early touchdown never drops support to two.
    # Slow on purpose — this is the gait to use on rough terrain, not for speed.
    Go2Gait.CRAWL: GaitParams(
        name=Go2Gait.CRAWL,
        phase_offsets=(0.25, 0.75, 0.0, 0.5),
        duty_factor=0.80,
        step_freq=1.00,
        step_height=0.09,
        weights=LocomotionWeights(
            rot=(1500.0, 1500.0, 0.0),   # tripod support: hold attitude tightly
            foot=(2e4, 2e4, 1e5),        # one swing leg at a time, so track it well
        ),
        description="One leg at a time; three feet always down. Statically stable, slow.",
    ),
    # Front pair / hind pair. Pitch is the failure axis: in a two-pair gait the
    # support during swing is two feet on a single lateral line, and a
    # stance-force QP cannot generate a pitch moment about that line — the trunk
    # pitches and the base drops. Because both pairs move together, the
    # single-pair support interval *is* the swing time, (1 - duty) / step_freq,
    # so duty and step_freq are the only knobs that shorten it.
    #
    # Measured on flat, 4000 steps, no randomization and no trunk force, against
    # a commanded 0.27 m (min height / 5th pct / peak |pitch|):
    #   duty 0.60 @ 1.80 Hz, 12 cm step  -> 0.134 / 0.154 / 7.8 deg   (falls)
    #   duty 0.75 @ 1.60 Hz,  8 cm step  -> 0.247 / 0.250 / 3.0 deg
    #   duty 0.78 @ 1.55 Hz,  8 cm step  -> 0.261 / 0.263 / 2.6 deg   (chosen)
    #   duty 0.80 @ 1.50 Hz,  7 cm step  -> 0.268 / 0.271 / 2.3 deg   (tracks +5 mm high)
    # The original 0.60 spent 80% of the cycle on one pair, 222 ms at a time, and
    # sagged to 0.134 m — below the 0.135 m crash threshold. 0.78 puts 56% of the
    # cycle on all four feet, cuts the single-pair interval to 142 ms, and holds
    # mean height at 0.269 m against the 0.270 m command. Heavier z and pitch
    # weights do the rest.
    #
    # Honest caveat: at 56% four-support this is a conservative bound, not a
    # ballistic one. A real bound needs a flight phase and angular-momentum
    # control that this stance-force MPC cannot produce. Swing is 142 ms, close
    # to the ~120 ms floor below which foothold tracking degrades, so raising
    # duty further trades pitch stability for missed footholds.
    Go2Gait.BOUND: GaitParams(
        name=Go2Gait.BOUND,
        phase_offsets=(0.5, 0.5, 0.0, 0.0),
        duty_factor=0.78,
        step_freq=1.55,
        step_height=0.08,
        clearance_speed=0.3,
        weights=LocomotionWeights(
            pos=(0.0, 0.0, 2e4),         # hold height harder: this is what sagged
            rot=(1000.0, 4500.0, 0.0),   # pitch is the failure axis here
            ang_vel=3e2,
            lin_vel=4e3,
        ),
        description="Front pair / hind pair. Pitch-unstable; conservative duty.",
    ),
}


# Gait used by ``Go2Locomotion()`` and the module-level ``config`` when no gait
# is passed. Change this one line to switch what the robot walks everywhere,
# the way uncommenting a ``timer_t`` line used to.
DEFAULT_GAIT: str = Go2Gait.TROT


def go2_gait_params(gait: str | GaitParams | None = None) -> GaitParams:
    """Look up a :class:`GaitParams` by tag. ``None`` gives :data:`DEFAULT_GAIT`.

    A :class:`GaitParams` instance passes through, so a caller can hand in a
    one-off ``replace(GO2_GAITS["trot"], step_freq=1.5)`` without registering it.
    """
    if gait is None:
        return GO2_GAITS[DEFAULT_GAIT]
    if isinstance(gait, GaitParams):
        return gait
    key = str(gait).lower().strip().replace("-", "_")
    try:
        return GO2_GAITS[key]
    except KeyError as e:
        known = ", ".join(sorted(GO2_GAITS))
        raise ValueError(f"Unknown Go2 gait {gait!r}; expected one of: {known}") from e
#endregion
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
    mpc_model: str = Go2MpcModel.WHOLE_BODY
    solver_mode = "primal_dual"

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
# region Go2Locomotion
class Go2Locomotion(_Go2Common):
    """
    Terrain estimator on, lateral base position softly unconstrained.

    Gait timing, swing geometry, nominal height and cost weights all come from
    one :class:`GaitParams` entry — see :data:`GO2_GAITS`::

        Go2Locomotion()                  # trot
        Go2Locomotion(Go2Gait.CRAWL)     # crawl
        Go2Locomotion(replace(GO2_GAITS["trot"], step_freq=1.5))   # one-off tweak

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
            ``Go2Gait.TROT`` / ``PACE`` / ``CRAWL`` / ``BOUND``, a ready-made
            :class:`GaitParams`, or ``None`` for :data:`DEFAULT_GAIT`.
        """
        params = go2_gait_params(gait)
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
        return f"Go2Locomotion({self.gait.summary()})"
#endregion
#===========================================================
# region Go2InverseLocomotion
def _go2_inverse_weights(n_joints: int, n_contact: int) -> dict:
    nv = 6 + n_joints
    return {
        "pos": jnp.diag(jnp.array([0.0, 0.0, 1.0e6])),
        "rot": jnp.diag(jnp.array([1.0e3, 1.0e3, 0.0])),
        "q": jnp.diag(jnp.ones(n_joints)) * 1.0e2,
        "vel": jnp.diag(jnp.ones(3)) * 5.0e3,
        "omega": jnp.diag(jnp.ones(3)) * 1.0e2,
        "dq": jnp.diag(jnp.ones(n_joints)) * 1.0e-1,
        "contact": jnp.diag(jnp.tile(jnp.array([1.0e5, 1.0e5, 1.0e5]), n_contact)),
        "acc": jnp.diag(jnp.ones(nv)) * 1.0e-1,
        "tau": jnp.diag(jnp.ones(n_joints)) * 1.0e-1,
        "grf": jnp.diag(jnp.ones(3 * n_contact)) * 1.0e-2,
    }


class Go2InverseLocomotion(Go2Locomotion):
    """Go2 locomotion gait with inverse-dynamics equality MPC."""

    mpc_model: str = Go2MpcModel.INVERSE_DYNAMICS
    solver_mode = "equality"
    equality_num_alpha: int = 10
    regularization = 1e-6
    grf_as_state: bool = False
    N: int = 12
    max_torque: float = 30.0
    min_torque: float = -30.0
    use_terrain_estimation: bool = False

    def __init__(self, gait: str | GaitParams | None = None) -> None:
        super().__init__(gait)
        nq, nv, n, m, equality_dim = inv_dyn_dims(self.n_joints, self.n_contact)
        self.nq = nq
        self.nv = nv
        self.equality_dim = equality_dim
        self._inv_n = n
        self._inv_m = m
        self.u_ref = jnp.zeros(m)
        self._inv_W = _go2_inverse_weights(self.n_joints, self.n_contact)
        dyn_f, eq_f, cost_f = inverse_dynamics_factories(
            self.n_joints, self.n_contact, self.dt, self.N
        )
        self._inv_dynamics = dyn_f
        self._inv_equality = eq_f
        self._inv_cost = cost_f

    @property
    def n(self) -> int:
        return self._inv_n

    @property
    def m(self) -> int:
        return self._inv_m

    @property
    def initial_state(self) -> jnp.ndarray:
        return jnp.concatenate(
            [self.p0, self.quat0, self.q0, jnp.zeros(self.nv)]
        )

    @property
    def W(self):
        return self._inv_W

    @property
    def dynamics(self):
        return self._inv_dynamics

    @property
    def equality(self):
        return self._inv_equality

    @property
    def cost(self):
        return self._inv_cost

    def __repr__(self) -> str:
        return f"Go2InverseLocomotion({self.gait.summary()})"
#endregion
#===========================================================
# region Go2SrbdLocomotion
def _go2_srbd_weights(n_contact: int) -> jnp.ndarray:
    """SRBD cost layout: ``[p(3), rot(3), dp(3), omega(3), grf(3 n_contact)]``."""
    Qp = jnp.diag(jnp.array([0.0, 0.0, 1e4]))
    Qrot = jnp.diag(jnp.array([1e3, 1e3, 0.0]))
    Qdp = jnp.diag(jnp.ones(3)) * 1e3
    Qomega = jnp.diag(jnp.ones(3)) * 1e1
    Qgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-2
    return jax.scipy.linalg.block_diag(Qp, Qrot, Qdp, Qomega, Qgrf)


class Go2SrbdLocomotion(Go2Locomotion):
    """Go2 locomotion gait with centroidal SRBD MPC (GRFs → joint torque)."""

    mpc_model: str = Go2MpcModel.SRBD
    solver_mode = "primal_dual"
    use_terrain_estimation: bool = False
    whole_body_frequency: int = 500

    # Full-robot mass from ``go2_mjx.xml`` (trunk + four legs). Inertia is a
    # standing-pose diagonal estimate, not Aliengo's composite tensor.
    mass: float = 15.206
    inertia = jnp.diag(jnp.array([0.15, 0.35, 0.38]))
    Kp = jnp.diag(jnp.tile(jnp.array([100.0, 100.0, 100.0]), 4))
    Kd = jnp.diag(jnp.tile(jnp.array([5.0, 5.0, 5.0]), 4))

    def __init__(self, gait: str | GaitParams | None = None) -> None:
        super().__init__(gait)
        self.u_ref = jnp.zeros(3 * self.n_contact)
        self._srbd_W = _go2_srbd_weights(self.n_contact)

    @property
    def use_terrain_estimator(self) -> bool:
        return bool(self.use_terrain_estimation)

    @property
    def n(self) -> int:
        return 13

    @property
    def m(self) -> int:
        return 3 * self.n_contact

    @property
    def initial_state(self) -> jnp.ndarray:
        return jnp.concatenate([self.p0, self.quat0, jnp.zeros(6)])

    @property
    def W(self) -> jnp.ndarray:
        return self._srbd_W

    @property
    def cost(self):
        return partial(mpc_objectives.quadruped_srbd_obj, self.n_contact, self.N)

    @property
    def hessian_approx(self):
        return partial(mpc_objectives.quadruped_srbd_hessian_gn, self.n_contact)

    @property
    def dynamics(self):
        mass = float(self.mass)
        inertia = self.inertia
        inertia_inv = jnp.linalg.inv(inertia)
        dt = self.dt

        def _factory(model, mjx_model, contact_id, body_id):
            del model, mjx_model, contact_id, body_id
            return partial(
                mpc_dyn_model.quadruped_srbd_dynamics,
                mass,
                inertia,
                inertia_inv,
                dt,
            )

        return _factory

    def __repr__(self) -> str:
        return f"Go2SrbdLocomotion({self.gait.summary()})"
#endregion
#===========================================================
# region Go2Balance

# Balance is not a gait: the timer is bypassed and support comes from
# ``balance_fixed_contact_mask``. It still needs the same weight layout, so it
# reuses ``LocomotionWeights`` with its own tripod-tuned numbers.
BALANCE_WEIGHTS = LocomotionWeights(
    pos=(9e2, 9e2, 1.2e4),        # hold station in x/y too, unlike locomotion
    rot=(2200.0, 2200.0, 2200.0), # yaw is held as well, there is no yaw command
    joint_pos=1e2,
    lin_vel=8e3,
    ang_vel=3e2,
    joint_vel=1e0,
    torque=1e-1,
    grf=1e-2,
    foot=(7e3, 7e3, 6e4),
)


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

    The ``BALANCE_WEIGHTS`` above are tuned with tripod-style support in mind; they are shared
    across all ``balance_stance`` presets unless you pass ``weights=`` to override them.

    **Why ``n_contact`` stays 4 for every balance preset:** the Go2 whole-body model always
    stacks **four feet** (FL, FR, RL, RR) in the state and in ``_block_W`` — three XYZ blocks per
    foot for foot position cost and three per foot for GRF cost. Tripod / diagonal stances do
    **not** remove legs from the vector; ``balance_fixed_contact_mask`` (and measured contact)
    tells the dynamics and costs which legs are nominal **stance** vs **swing** (mask 0 turns off
    nominal contact/reaction for that foot). A smaller ``n_contact`` would require a different
    model layout (not supported here).
    """

    behaviour: str = "balance"
    swing_tracking: bool = True # this is the flag for the foot tracking cost
    use_balance_fixed_contact: bool = True

    # Tripod: world foot ref from random base-frame sample around ``p_legs0`` (resampled on MPC reset).
    use_tripod_nominal_foot_ref: bool = True
    tripod_foot_ref_sigma = jnp.array([0.03, 0.03, 0.005])

    robot_height: float = 0.27 # this is the nominal height of the robot
    p0 = jnp.array([0, 0, robot_height]) # this is the nominal position of the robot
    initial_height: float = 0.27

    # Gait timer bypassed: every leg sits at phase 0 with duty 1.0, i.e. always stance.
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
        """
        Parameters
        ----------
        balance_stance
            One of ``BalanceStance.FOUR``, ``TRIPOD_SWING_*``, ``THREE_MISSING_*`` (legacy tripod),
            or ``DIAG_FL_RR`` / ``DIAG_FR_RL``. If ``None``, defaults to ``FOUR``.
        weights
            Override for ``BALANCE_WEIGHTS``.
        """
        stance = BalanceStance.FOUR if balance_stance is None else balance_stance
        self.balance_stance = stance
        self.balance_fixed_contact_mask = balance_stance_to_mask(stance)
        self.weights = BALANCE_WEIGHTS if weights is None else weights

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
    gait: str | GaitParams | None = None,
    balance_stance: str | None = None,
    mpc_model: str = Go2MpcModel.WHOLE_BODY,
) -> _Go2Common:
    """
    Parameters
    ----------
    mode
        ``Go2Mode.LOCOMOTION`` / ``Go2Mode.BALANCE`` or the equivalent strings.
    gait
        Locomotion mode only. Gait tag or :class:`GaitParams`.
    balance_stance
        Balance mode only. Nominal MPC contact mask.
    mpc_model
        Locomotion only. ``Go2MpcModel.WHOLE_BODY`` (default),
        ``Go2MpcModel.INVERSE_DYNAMICS``, or ``Go2MpcModel.SRBD``.
    """
    key = mode.lower().strip()
    model_key = str(mpc_model).lower().strip().replace("-", "_")
    if key == Go2Mode.LOCOMOTION:
        if model_key == Go2MpcModel.INVERSE_DYNAMICS:
            return Go2InverseLocomotion(gait)
        if model_key == Go2MpcModel.SRBD:
            return Go2SrbdLocomotion(gait)
        if model_key != Go2MpcModel.WHOLE_BODY:
            raise ValueError(
                f"Unknown Go2 mpc_model {mpc_model!r}; expected "
                f"{Go2MpcModel.WHOLE_BODY!r}, {Go2MpcModel.INVERSE_DYNAMICS!r}, "
                f"or {Go2MpcModel.SRBD!r}."
            )
        return Go2Locomotion(gait)
    if key == Go2Mode.BALANCE:
        if model_key in (Go2MpcModel.INVERSE_DYNAMICS, Go2MpcModel.SRBD):
            raise ValueError(
                f"{mpc_model!r} MPC is only implemented for locomotion, "
                "not balance."
            )
        return Go2Balance(balance_stance=balance_stance)
    raise ValueError(
        f"Unknown Go2 behaviour {mode!r}; expected {Go2Mode.LOCOMOTION!r} or {Go2Mode.BALANCE!r}."
    )


# Backward-compatible default (same content as historical module-level globals)
config = Go2Locomotion()
