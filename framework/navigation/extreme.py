"""Near-limit velocity commands that slew instead of stepping.

``--nav extreme`` asks the robot to sit next to its linear and yaw ceilings,
sometimes on one, sometimes on both, and sometimes a full stop that drops
from that ceiling straight to zero and holds. The ceiling is per robot and
per gait. Speeding up is limited to ``accel * dt`` per control tick, because
a step up to those ceilings drops the robot. Braking is immediate. A reversal
passes through zero and then ramps up in the new direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExtremeLimits:
    """Command ceiling and the acceleration allowed while approaching it."""

    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float
    linear_accel_mps2: float
    yaw_accel_rps2: float


# Command ceilings, not brochure top speeds. Columns are
# vx [m/s], vy [m/s], yaw [rad/s], linear accel [m/s²], yaw accel [rad/s²].
# Trot is the robust default. Pace and bound keep yaw light because those gaits
# lose the robot in roll. Crawl is the slow, three-foot gait.
EXTREME_LIMITS: dict[str, dict[str, ExtremeLimits]] = {
    "go2": {
        "trot": ExtremeLimits(1.0, 0.5, 0.8, 0.5, 1.2),
        "pace": ExtremeLimits(0.7, 0.3, 0.4, 0.35, 0.7),
        "crawl": ExtremeLimits(0.45, 0.3, 0.4, 0.3, 0.7),
        "bound": ExtremeLimits(1.15, 0.3, 0.4, 0.4, 0.65),
    },
    "go2_dls": {
        "trot": ExtremeLimits(1.0, 0.5, 0.8, 0.5, 1.2),
        "pace": ExtremeLimits(0.7, 0.3, 0.4, 0.35, 0.7),
        "crawl": ExtremeLimits(0.45, 0.3, 0.4, 0.3, 0.7),
        "bound": ExtremeLimits(1.15, 0.3, 0.4, 0.4, 0.65),
    },
    # No gait switch on this robot; the registered walk is a trot.
    "aliengo": {
        "trot": ExtremeLimits(1.0, 0.4, 0.6, 0.4, 1.0),
    },
    "b2": {
        "trot": ExtremeLimits(1.2, 0.4, 0.5, 0.35, 0.8),
        "pace": ExtremeLimits(0.85, 0.25, 0.3, 0.25, 0.5),
        "crawl": ExtremeLimits(0.55, 0.25, 0.3, 0.2, 0.5),
        "bound": ExtremeLimits(1.4, 0.25, 0.3, 0.3, 0.45),
    },
    "spot": {
        "trot": ExtremeLimits(1.6, 0.5, 0.8, 0.6, 1.2),
        "pace": ExtremeLimits(1.1, 0.3, 0.4, 0.4, 0.7),
        "crawl": ExtremeLimits(0.7, 0.3, 0.4, 0.35, 0.7),
        "bound": ExtremeLimits(1.8, 0.3, 0.4, 0.5, 0.65),
    },
}


def limits_for(robot: str, gait: str = "trot") -> ExtremeLimits:
    try:
        by_gait = EXTREME_LIMITS[robot]
    except KeyError as exc:
        known = ", ".join(sorted(EXTREME_LIMITS))
        raise ValueError(
            f"No extreme velocity limits for robot {robot!r}. Known: {known}."
        ) from exc
    try:
        return by_gait[gait]
    except KeyError as exc:
        known = ", ".join(sorted(by_gait))
        raise ValueError(
            f"No extreme limits for {robot!r} gait {gait!r}. Gaits: {known}."
        ) from exc


class ExtremeNavigator:
    """Slew ``(vx, vy, yaw_rate)`` toward a near-limit target, then hold it.

    Call :meth:`step` once per control tick, with ``dt`` equal to that tick.
    The hold starts only after the command has arrived, so the ramp is not
    eaten by the hold timer. The default hold is 0.5 s.
    """

    def __init__(
        self,
        robot: str,
        gait: str = "trot",
        *,
        dt: float,
        rng: np.random.Generator | None = None,
        hold_s: tuple[float, float] = (0.5, 0.5),
        band: tuple[float, float] = (0.9, 1.0),
    ):
        self.robot = robot
        self.gait = gait
        self.limits = limits_for(robot, gait)
        self.dt = float(dt)
        self.hold_s = (float(hold_s[0]), float(hold_s[1]))
        self.band = (float(band[0]), float(band[1]))
        self._rng = rng if rng is not None else np.random.default_rng()
        self.segment_id = -1
        self._command = np.zeros(3, dtype=np.float64)
        self._target = np.zeros(3, dtype=np.float64)
        self._elapsed = 0.0
        self._hold = 0.0

    def reset(self) -> None:
        """Start from standstill and draw the first target."""
        self.segment_id = -1
        self._command[:] = 0.0
        self._elapsed = 0.0
        self._new_segment()

    def _near_limit(self, limit: float) -> float:
        lo, hi = self.band
        magnitude = float(self._rng.uniform(lo * limit, hi * limit))
        sign = float(self._rng.choice((-1.0, 1.0)))
        return sign * magnitude

    def _new_segment(self) -> None:
        self.segment_id += 1
        # A stop is only drawn once the command is already at a ceiling, so it
        # is a drop from max speed to zero rather than another rest segment.
        at_speed = float(np.max(np.abs(self._command))) > 1e-9
        kinds = ("stop", "linear", "angular", "both") if at_speed else (
            "linear", "angular", "both"
        )
        kind = str(self._rng.choice(kinds))
        target = np.zeros(3, dtype=np.float64)
        limits = self.limits
        if kind in ("linear", "both"):
            target[0] = self._near_limit(limits.vx_mps)
            target[1] = self._near_limit(limits.vy_mps)
        if kind in ("angular", "both"):
            target[2] = self._near_limit(limits.yaw_rate_rps)
        self._target = target
        self._elapsed = 0.0
        self._hold = float(self._rng.uniform(*self.hold_s))

    def _slew(self) -> None:
        """Accelerate within the limit. Brake to the target in one tick."""
        limits = self.limits
        cap = (
            limits.linear_accel_mps2 * self.dt,
            limits.linear_accel_mps2 * self.dt,
            limits.yaw_accel_rps2 * self.dt,
        )
        for i, step in enumerate(cap):
            current = float(self._command[i])
            target = float(self._target[i])
            # A reversal brakes to zero now; the opposite direction ramps after.
            if current * target < 0.0:
                current = 0.0
            if abs(target) <= abs(current):
                self._command[i] = target
                continue
            delta = target - current
            self._command[i] = current + float(np.clip(delta, -step, step))

    def step(self) -> np.ndarray:
        """Advance one control tick and return ``(vx, vy, yaw_rate)``."""
        self._slew()
        if float(np.max(np.abs(self._target - self._command))) <= 1e-9:
            self._elapsed += self.dt
            if self._elapsed >= self._hold:
                self._new_segment()
        return self._command.copy()

    def mpc_input(self, robot_height: float) -> np.ndarray:
        """7D locomotion command: ``[vx, vy, 0, 0, 0, yaw_rate, height]``."""
        vx, vy, wz = self._command
        return np.array(
            [vx, vy, 0.0, 0.0, 0.0, wz, float(robot_height)], dtype=np.float64
        )
