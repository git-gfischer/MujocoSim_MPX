"""
Segmented velocity commands for data collection.

Why this exists
---------------
Goal-following produced a dataset with no turning and no reverse: the audited v4
run had commanded yaw identically zero for all 23,994 rows, `vx` in [0, 0.66],
and a realised median speed of 0.255 m/s — close to marching in place. A
representation cannot be shown to be yaw-invariant on data where the robot never
turns, and a backward trot has a foot-velocity signature that simply was not
present.

What it does
------------
Samples ``(vx, vy, yaw_rate)`` and **holds each command for a random 2-5 s**
before ramping to the next over ~0.3 s. A 60 s episode then contains 12-30
distinct commands including accelerations, decelerations and direction
reversals, rather than one steady state. Roughly 10% of segments are
``zero_command`` (stand in place), which gives the standing folders a
within-distribution counterpart inside locomotion episodes.

The ramp matters: stepping the command discontinuously would make every segment
boundary a transient the controller has to reject, and those transients would
dominate the window statistics the Proprioceptive Image encodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class VelocityCommandConfig:
    """Ranges and timing for :class:`VelocityCommandSampler`."""

    # Forward/backward, lateral, and turning rate. Reverse and turning are the
    # two the audited v4 run had none of.
    vx_mps: Tuple[float, float] = (-1.0, 1.0)
    vy_mps: Tuple[float, float] = (-0.5, 0.5)
    yaw_rate_rps: Tuple[float, float] = (-1.0, 1.0)

    # How long one command is held, before the ramp to the next.
    hold_s: Tuple[float, float] = (2.0, 5.0)

    # Ramp duration between consecutive commands.
    ramp_s: float = 0.3

    # Probability that a segment is "stand still".
    zero_command_prob: float = 0.10

    # Minimum planar speed for a non-zero segment, so the sampler does not spend
    # segments on commands that are indistinguishable from standing.
    min_speed_mps: float = 0.15

    # Budget on the COMBINED command, as a sum of per-axis fractions. Sampling
    # each axis independently lands in the corner of the box — a hard diagonal
    # reverse while turning — which the MPC cannot execute: the pilot that first
    # used independent sampling folded the robot in 14 of 15 episodes. 1.0 allows
    # full command on any one axis, or a proportional split across several.
    max_combined_load: float = 1.0

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "vx_mps": list(self.vx_mps),
            "vy_mps": list(self.vy_mps),
            "yaw_rate_rps": list(self.yaw_rate_rps),
            "hold_s": list(self.hold_s),
            "ramp_s": float(self.ramp_s),
            "zero_command_prob": float(self.zero_command_prob),
            "min_speed_mps": float(self.min_speed_mps),
        }


@dataclass
class VelocityCommandSampler:
    """
    Produces a piecewise-constant, ramped ``(vx, vy, yaw_rate)`` command.

    Call :meth:`step` once per control step. ``segment_id`` increments on each
    new command and is logged as ``cmd_segment_id`` so a sampler can group,
    stratify or exclude transitions.
    """

    config: VelocityCommandConfig = field(default_factory=VelocityCommandConfig)
    dt: float = 0.02
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    segment_id: int = field(default=-1, init=False)
    _command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64), init=False, repr=False
    )
    _from: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64), init=False, repr=False
    )
    _target: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64), init=False, repr=False
    )
    _elapsed: float = field(default=0.0, init=False, repr=False)
    _hold: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        """Start a new episode from standstill and draw the first segment."""
        self.segment_id = -1
        self._command = np.zeros(3, dtype=np.float64)
        self._from = np.zeros(3, dtype=np.float64)
        self._new_segment()

    def _draw(self) -> np.ndarray:
        """One command, either standstill or a sample with a meaningful speed."""
        cfg = self.config
        if self.rng.random() < cfg.zero_command_prob:
            return np.zeros(3, dtype=np.float64)

        # Rejection-sample so a segment is not a near-zero command wearing a
        # non-zero label; give up after a few tries and take what we have.
        for _ in range(8):
            command = np.array(
                [
                    self.rng.uniform(*cfg.vx_mps),
                    self.rng.uniform(*cfg.vy_mps),
                    self.rng.uniform(*cfg.yaw_rate_rps),
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(command[:2]) >= cfg.min_speed_mps:
                return self._limit_combined(command)
        return self._limit_combined(command)

    def _limit_combined(self, command: np.ndarray) -> np.ndarray:
        """
        Scale a command back onto the feasibility budget.

        Scaling rather than rejecting keeps the *direction* the sampler drew, so
        coverage of reverse and turning is preserved; only the magnitude of a
        simultaneous multi-axis demand is reduced.
        """
        cfg = self.config
        spans = np.array(
            [
                max(abs(cfg.vx_mps[0]), abs(cfg.vx_mps[1])),
                max(abs(cfg.vy_mps[0]), abs(cfg.vy_mps[1])),
                max(abs(cfg.yaw_rate_rps[0]), abs(cfg.yaw_rate_rps[1])),
            ],
            dtype=np.float64,
        )
        spans[spans <= 0] = 1.0
        load = float(np.sum(np.abs(command) / spans))
        if load > cfg.max_combined_load:
            command = command * (cfg.max_combined_load / load)
        return command

    def _new_segment(self) -> None:
        self.segment_id += 1
        self._from = self._command.copy()
        self._target = self._draw()
        self._elapsed = 0.0
        self._hold = float(self.rng.uniform(*self.config.hold_s))

    def step(self) -> np.ndarray:
        """Advance one control step and return ``(vx, vy, yaw_rate)``."""
        self._elapsed += self.dt

        ramp = max(self.config.ramp_s, 1e-9)
        if self._elapsed < ramp:
            # Smoothstep rather than linear: zero slope at both ends, so the
            # controller never sees a step change in commanded acceleration.
            u = self._elapsed / ramp
            blend = u * u * (3.0 - 2.0 * u)
            self._command = self._from + blend * (self._target - self._from)
        else:
            self._command = self._target.copy()

        if self._elapsed >= self._hold + ramp:
            self._new_segment()

        return self._command.copy()

    @property
    def command(self) -> np.ndarray:
        """Current command without advancing."""
        return self._command.copy()

    def mpc_input(self, robot_height: float) -> np.ndarray:
        """
        The 7D locomotion command the MPC consumes.

        Layout matches :meth:`PointNavigator.mpc_input`:
        ``[vx, vy, 0, 0, 0, yaw_rate, height]`` — note the yaw rate sits at
        index **5**, not 2. Slicing ``[:3]`` off this vector is what silently
        logged a zero yaw command for the whole of the audited v4 run.
        """
        vx, vy, wz = self._command
        return np.array(
            [vx, vy, 0.0, 0.0, 0.0, wz, float(robot_height)], dtype=np.float64
        )


def command_from_mpc_input(mpc_input: np.ndarray) -> np.ndarray:
    """
    Extract ``(vx, vy, yaw_rate)`` from the 7D MPC command vector.

    One place that knows the layout, so the index-5 trap cannot be re-sprung by
    another caller reaching for ``[:3]``.
    """
    values = np.asarray(mpc_input, dtype=np.float64).reshape(-1)
    if values.size < 6:
        padded = np.zeros(3, dtype=np.float64)
        padded[: min(3, values.size)] = values[:3]
        return padded
    return np.array([values[0], values[1], values[5]], dtype=np.float64)
