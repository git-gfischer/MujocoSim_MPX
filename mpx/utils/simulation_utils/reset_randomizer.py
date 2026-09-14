"""
Sample and apply episode parameters on each MuJoCo respawn.

Sampling is uniform (or log-uniform) over config ranges. Applying writes to
whatever targets are present: payload force, navigator limits, MPC gait timing,
and foot geom ``solref`` time constants and sliding friction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mpx.config.sim_config.config_reset_randomization import (
    FloatRangeSpec,
    ResetRandomizationConfig,
    loco_reset_randomization_config,
)


@dataclass(frozen=True)
class ResetSample:
    """Realized reset knobs. ``None`` means that knob was not sampled."""

    payload_kg: float | None = None
    max_speed: float | None = None
    max_yaw_rate: float | None = None
    step_freq: float | None = None
    duty_factor: float | None = None
    solref_timeconst: float | None = None
    friction: float | None = None

    def to_metadata(self) -> dict[str, float]:
        """JSON-friendly dict of sampled knobs only."""
        return {
            name: float(value)
            for name, value in (
                ("payload_kg", self.payload_kg),
                ("max_speed", self.max_speed),
                ("max_yaw_rate", self.max_yaw_rate),
                ("step_freq", self.step_freq),
                ("duty_factor", self.duty_factor),
                ("solref_timeconst", self.solref_timeconst),
                ("friction", self.friction),
            )
            if value is not None
        }


@dataclass
class ResetTargets:
    """Live objects the randomizer may write. Absent targets are skipped."""

    model: Any = None
    foot_geom_ids: Any = None
    base_weight: Any = None
    navigator: Any = None
    mpc_data: Any = None


class ResetRandomizer:
    """Draw a :class:`ResetSample` and apply it to :class:`ResetTargets`."""

    def __init__(
        self,
        cfg: ResetRandomizationConfig,
        rng: np.random.Generator | None = None,
    ):
        self.cfg = cfg
        self._rng = rng if rng is not None else np.random.default_rng(cfg.rng_seed)

    @classmethod
    def from_config(
        cls,
        cfg: ResetRandomizationConfig = loco_reset_randomization_config,
    ) -> ResetRandomizer:
        return cls(cfg=cfg)

    def sample(self) -> ResetSample:
        """Draw one sample. Master ``enabled=False`` returns an empty sample."""
        if not self.cfg.enabled:
            return ResetSample()
        return ResetSample(
            payload_kg=self._draw(self.cfg.payload),
            max_speed=self._draw(self.cfg.max_speed),
            max_yaw_rate=self._draw(self.cfg.max_yaw_rate),
            step_freq=self._draw(self.cfg.step_freq),
            duty_factor=self._draw(self.cfg.duty_factor),
            solref_timeconst=self._draw(self.cfg.solref_timeconst),
            friction=self._draw(self.cfg.friction),
        )

    def apply(self, sample: ResetSample, targets: ResetTargets) -> Any:
        """Write ``sample`` onto ``targets``. Returns (possibly replaced) ``mpc_data``."""
        mpc_data = targets.mpc_data
        if sample.payload_kg is not None and targets.base_weight is not None:
            targets.base_weight.enabled = True
            targets.base_weight.extra_mass_kg = float(sample.payload_kg)
        if targets.navigator is not None:
            if sample.max_speed is not None:
                targets.navigator.max_speed = float(sample.max_speed)
            if sample.max_yaw_rate is not None:
                targets.navigator.max_yaw_rate = float(sample.max_yaw_rate)
        if mpc_data is not None:
            replace_kwargs = {}
            if sample.step_freq is not None:
                replace_kwargs["step_freq"] = float(sample.step_freq)
            if sample.duty_factor is not None:
                replace_kwargs["duty_factor"] = float(sample.duty_factor)
            if replace_kwargs:
                mpc_data = mpc_data.replace(**replace_kwargs)
        if (
            sample.solref_timeconst is not None
            and targets.model is not None
            and targets.foot_geom_ids is not None
        ):
            timeconst = float(sample.solref_timeconst)
            for geom_id in np.asarray(targets.foot_geom_ids).reshape(-1):
                targets.model.geom_solref[int(geom_id), 0] = timeconst
        if (
            sample.friction is not None
            and targets.model is not None
            and targets.foot_geom_ids is not None
        ):
            mu = float(sample.friction)
            for geom_id in np.asarray(targets.foot_geom_ids).reshape(-1):
                targets.model.geom_friction[int(geom_id), 0] = mu
        return mpc_data

    def sample_and_apply(self, targets: ResetTargets) -> tuple[ResetSample, Any]:
        sample = self.sample()
        mpc_data = self.apply(sample, targets)
        return sample, mpc_data

    def _draw(self, spec: FloatRangeSpec) -> float | None:
        if not spec.enabled:
            return None
        if spec.log_uniform:
            log_low = np.log(spec.low)
            log_high = np.log(spec.high)
            return float(np.exp(self._rng.uniform(log_low, log_high)))
        return float(self._rng.uniform(spec.low, spec.high))
