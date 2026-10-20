"""
Reset-time domain randomization ranges for quadruped simulations.

Typical use::

    from mpx.config.sim_config.config_reset_randomization import (
        loco_reset_randomization_config,
    )
    from mpx.utils.simulation_utils.reset_randomizer import ResetRandomizer

    randomizer = ResetRandomizer.from_config(loco_reset_randomization_config)
    # In _respawn(), after reset_mpc(...):
    # sample, mpc_data = randomizer.sample_and_apply(targets)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class FloatRangeSpec:
    """Inclusive sampling range for one reset knob."""

    enabled: bool = True
    low: float = 0.0
    high: float = 1.0
    log_uniform: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(
                f"FloatRangeSpec low ({self.low}) must be <= high ({self.high})"
            )
        if self.log_uniform and self.low <= 0.0:
            raise ValueError("log_uniform ranges require low > 0")


# Navigation command limits drawn per episode [m/s, rad/s]. See the
# max_speed / max_yaw_rate fields below for how these were chosen.
NOMINAL_MAX_SPEED = FloatRangeSpec(enabled=True, low=0.25, high=0.50)
NOMINAL_MAX_YAW_RATE = FloatRangeSpec(enabled=True, low=0.15, high=0.55)
# Stairs: on the staired pyramid, turning or side-stepping on the steps ended
# in a roll-over 3x as often as walking straight (8.3 vs 2.85 per minute on
# the steps). Turn slower there, and stride shorter so a step-down is smaller.
STAIRS_MAX_SPEED = FloatRangeSpec(enabled=True, low=0.15, high=0.35)
STAIRS_MAX_YAW_RATE = FloatRangeSpec(enabled=True, low=0.10, high=0.30)

# Velocity command ranges per terrain (--scene). ``for_scene`` swaps them into
# a profile; a scene missing here keeps the profile's own ranges.
SCENE_COMMAND_RANGES: dict[str, dict[str, FloatRangeSpec]] = {
    "flat": {"max_speed": NOMINAL_MAX_SPEED, "max_yaw_rate": NOMINAL_MAX_YAW_RATE},
    "rough": {"max_speed": NOMINAL_MAX_SPEED, "max_yaw_rate": NOMINAL_MAX_YAW_RATE},
    "perlin": {"max_speed": NOMINAL_MAX_SPEED, "max_yaw_rate": NOMINAL_MAX_YAW_RATE},
    "slippery": {"max_speed": NOMINAL_MAX_SPEED, "max_yaw_rate": NOMINAL_MAX_YAW_RATE},
    "ramp": {"max_speed": NOMINAL_MAX_SPEED, "max_yaw_rate": NOMINAL_MAX_YAW_RATE},
    "stairs": {"max_speed": STAIRS_MAX_SPEED, "max_yaw_rate": STAIRS_MAX_YAW_RATE},
    "staired_pyramid": {"max_speed": STAIRS_MAX_SPEED, "max_yaw_rate": STAIRS_MAX_YAW_RATE},
}

# go2_dls, non-slippery floor, tested per gait: (lin max m/s, yaw max rad/s).
# Yaw is always [-yaw max, yaw max]. Linear sign follows the nav mode:
# flat and rough use random_fb, so vx is [-lin max, lin max]; ramp and
# staired_pyramid walk forward only, so vx is [0, lin max].
# Slippery floors are a separate set, not here.
GO2_DLS_NONSLIP_CAPS: dict[str, dict[str, tuple[float, float]]] = {
    "flat": {
        "bound": (0.8, 1.5),
        "pace": (0.8, 1.5),
        "trot": (1.0, 1.5),
        "crawl": (0.8, 1.5),
    },
    "rough": {
        "bound": (0.8, 1.5),
        "pace": (0.7, 1.5),
        "trot": (1.0, 1.5),
        "crawl": (0.8, 1.5),
    },
    "staired_pyramid": {
        "bound": (0.2, 0.6),
        "pace": (0.3, 1.0),
        "trot": (0.5, 1.2),
        "crawl": (0.4, 1.2),
    },
    "ramp": {
        "bound": (0.3, 1.0),
        "pace": (0.4, 1.5),
        "trot": (1.0, 1.5),
        "crawl": (0.5, 1.2),
    },
}


@dataclass
class ResetRandomizationConfig:
    """Master switch plus per-knob ranges for :class:`ResetRandomizer`."""

    enabled: bool = True
    rng_seed: int | None = None

    payload: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.0, high=2.0) # 0.0, 5.0 

    # Realised median base speed in v3 was only 0.26 m/s against a [0.25, 0.70]
    # range. Widened, and the sign is handled by the navigator so reverse and
    # lateral motion are covered too.
    # Measured: at 1.0 m/s this MPC lost the robot in 6 of 10 episodes, and the
    # falls were what pushed joints past their limits. 0.80 keeps the coverage
    # Task 9d asks for while leaving the controller able to track the command.
    max_speed: FloatRangeSpec = NOMINAL_MAX_SPEED # 0.20, 0.80
    # Yaw rate cap handed to the navigator [rad/s]. Was [0.40, 1.50]; 1.50 rad/s
    # is ~86 deg/s and the navigator saturates it the instant a new goal appears
    # (kp_yaw * yaw_error saturates for any error past ~0.5 rad), while the
    # heading gain holds vx near zero until the robot faces the goal. That is a
    # spin-in-place from standstill at the cap, and it was putting the robot on
    # the floor. 0.80 is the navigator's own tuned default and stays inside what
    # this MPC tracks; PointNavigator now also slews toward the cap rather than
    # stepping to it (``yaw_accel_rps2``).
    max_yaw_rate: FloatRangeSpec = NOMINAL_MAX_YAW_RATE # 0.30, 1.50

    # Gait timing is OFF. duty_factor and step_freq are no longer independent
    # knobs: they belong to a gait, and config_go2.GO2_GAITS holds a matched set
    # per gait (trot / pace / crawl / bound). Sampling them independently drew
    # combinations no gait was tuned for — duty 0.80 at 1.00 Hz is a near-static
    # crawl on trot phase offsets, which collapses stride length and reads as the
    # robot struggling to go forward. To vary gait timing, pick a different gait
    # or edit its GaitParams entry; re-enable these only for a deliberate
    # timing-robustness study, and narrow the ranges around the chosen gait.
    step_freq: FloatRangeSpec = FloatRangeSpec(enabled=False, low=1.00, high=1.70)
    duty_factor: FloatRangeSpec = FloatRangeSpec(enabled=False, low=0.55, high=0.80)

    # Contact time constant [s]. Clamped well below the v3 [0.012, 0.035]: the
    # upper end there was 6.6 sim steps at 200 Hz and made the foot bouncy, which
    # is what produced the one-frame contact dropouts. Must stay >= 2 * timestep
    # (0.004 s at the 500 Hz default).
    # Measured on the v4 pilot: of the eight knobs, solref_timeconst separated
    # falls from survivals most strongly (Cohen's d -0.92), with the stiff end
    # losing the robot — a stiffer contact means harder impacts for the MPC to
    # absorb. 0.010-0.014 keeps 5-7 substeps at 500 Hz, stays far tighter than
    # the v3 [0.012, 0.035], and the contact-quality margin (0.75% one-step runs
    # against a 2% limit) covers the small loss of stiffness.
    solref_timeconst: FloatRangeSpec = FloatRangeSpec(
        enabled=False, low=0.010, high=0.014, log_uniform=True
    )

    # Sliding friction μ on foot geoms (geom_friction[:, 0]). Torsional/rolling stay as in XML.
    # MuJoCo contact μ is min(foot, floor); the flat floor is 2.0.
    # Floor raised to the nominal XML value of 1.2 for the standard folders: at
    # μ < 0.3 the fall rate reached 38% of episodes, which wastes wall-clock and
    # skews episode lengths. Genuinely slippery conditions belong in a dedicated
    # `slippery` folder with its own low-friction range, not mixed into nominal
    # locomotion (DATASET_FIX_TASKS_R2, Task R2-8 item 3).
    friction: FloatRangeSpec = FloatRangeSpec(enabled=True, low=1.2, high=2.00)

    # Commanded base height [m]. OFF, and note this knob was never actually in
    # effect: the randomizer writes the sample to ``navigator.robot_height``, but
    # every call site passes ``config.robot_height`` explicitly as the second
    # argument to ``mpc_input``, and the explicit argument wins. So the commanded
    # height has always been the constant 0.27 m from config_go2, and the value
    # recorded in episode metadata was a height the robot was never asked to hold.
    #
    # Leaving it disabled rather than repairing the wiring is deliberate: turning
    # it on would ADD a height step at episode start that is not there today
    # (commanding 0.30 m against a 0.27 m spawn is a 3 cm step the MPC has to
    # absorb in the first tick). Re-enable it only together with fixing the
    # ``mpc_input`` call sites, and ramp the height rather than stepping it.
    #
    # If the robot is dropping onto the ground at spawn, the height variation to
    # look at is the spawner's foot vertical relief, not this: it lifts the base
    # to the lowest collision-free z, which measures +0.010 m on flat but up to
    # the +0.100 m cap on rough. See SpawnConfig.foot_relief_max.
    base_height: FloatRangeSpec = FloatRangeSpec(enabled=False, low=0.21, high=0.30)

    # Per-joint scale on the plant's XML joint damping, armature and
    # frictionloss, drawn independently for each of the 12 joints. 0.5-1.5
    # covers the leg-to-leg spread of the IIT-DLS identified Go2 (knee
    # frictionloss 0.49-1.36 N*m around the 0.91 go2_dls value). Only the plant
    # model is written; the MPC keeps its nominal model. A field that is 0 in
    # the XML stays 0 (go2 has no frictionloss).
    joint_damping_scale: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.5, high=1.5)
    joint_armature_scale: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.5, high=1.5)
    joint_frictionloss_scale: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.5, high=1.5)

    def to_metadata(self) -> dict:
        """Ranges for this profile. A later split can select runs on these."""
        return asdict(self)

    def for_scene(
        self,
        scene: str,
        gait: str | None = None,
        robot: str | None = None,
    ) -> "ResetRandomizationConfig":
        """This profile with the velocity command ranges of ``scene``.

        go2_dls on flat, rough, ramp, or staired_pyramid pins the cap to the
        tested gait ceiling. Other robots, and a gait with no entry, keep the
        scene range.
        """
        ranges = SCENE_COMMAND_RANGES.get(scene, {})
        caps = GO2_DLS_NONSLIP_CAPS.get(scene, {})
        if robot == "go2_dls" and gait in caps:
            lin_max, yaw_max = caps[gait]
            ranges = {
                "max_speed": FloatRangeSpec(enabled=True, low=lin_max, high=lin_max),
                "max_yaw_rate": FloatRangeSpec(enabled=True, low=yaw_max, high=yaw_max),
            }
        return replace(self, **ranges)


# Loco profile follows the dataclass master switch.
# Balance is enabled here; knobs not listed follow the dataclass defaults.
loco_reset_randomization_config = ResetRandomizationConfig()

balance_reset_randomization_config = ResetRandomizationConfig(
    enabled=True,
    payload=FloatRangeSpec(enabled=True, low=0.0, high=4.0),
    solref_timeconst=FloatRangeSpec(
        enabled=False, low=0.010, high=0.014, log_uniform=True
    ),
    friction=FloatRangeSpec(enabled=True, low=1.2, high=2.00),
    base_height=FloatRangeSpec(enabled=False, low=0.21, high=0.30),
)