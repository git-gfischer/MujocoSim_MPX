"""
Sample and apply episode parameters on each MuJoCo respawn.

Sampling is uniform (or log-uniform) over config ranges. Applying writes to
whatever targets are present: payload force, navigator limits, MPC gait timing,
foot geom ``solref`` time constants and sliding friction, and per-joint scales
on the XML joint damping, armature and frictionloss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mpx.config.sim_config.config_reset_randomization import (
    FloatRangeSpec,
    ResetRandomizationConfig,
    loco_reset_randomization_config,
)
from mpx.utils.dataset_collection.dataset_schema import FOOT_ORDER, JOINT_ORDER, N_JOINTS

# Metadata suffix per joint, in the model's actuated-joint order (FL, FR, RL, RR).
_JOINT_NAMES = tuple(f"{foot}_{joint}" for foot in FOOT_ORDER for joint in JOINT_ORDER)
_JOINT_FIELDS = (
    ("joint_damping_scale", "dof_damping"),
    ("joint_armature_scale", "dof_armature"),
    ("joint_frictionloss_scale", "dof_frictionloss"),
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
    base_height: float | None = None
    # One scale per joint, multiplying the XML value.
    joint_damping_scale: tuple[float, ...] | None = None
    joint_armature_scale: tuple[float, ...] | None = None
    joint_frictionloss_scale: tuple[float, ...] | None = None

    def to_metadata(self) -> dict[str, float]:
        """JSON-friendly dict of sampled knobs only."""
        meta = {
            name: float(value)
            for name, value in (
                ("payload_kg", self.payload_kg),
                ("max_speed", self.max_speed),
                ("max_yaw_rate", self.max_yaw_rate),
                ("step_freq", self.step_freq),
                ("duty_factor", self.duty_factor),
                ("solref_timeconst", self.solref_timeconst),
                ("friction", self.friction),
                ("base_height", self.base_height),
            )
            if value is not None
        }
        for name, _ in _JOINT_FIELDS:
            scales = getattr(self, name)
            if scales is not None:
                meta.update(
                    {f"{name}_{joint}": float(s) for joint, s in zip(_JOINT_NAMES, scales)}
                )
        return meta


@dataclass
class ResetTargets:
    """Live objects the randomizer may write. Absent targets are skipped."""

    model: Any = None
    foot_geom_ids: Any = None
    base_weight: Any = None
    navigator: Any = None
    # Segmented velocity command source, used when collecting without a nav
    # mode. It takes the same max_speed / max_yaw_rate knobs as the navigator,
    # so the envelope varies per episode in both modes rather than only one.
    command_sampler: Any = None
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
        # XML joint values per model, captured on first write so scales never compound.
        self._joint_nominal: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

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
            base_height=self._draw(self.cfg.base_height),
            joint_damping_scale=self._draw_joints(self.cfg.joint_damping_scale),
            joint_armature_scale=self._draw_joints(self.cfg.joint_armature_scale),
            joint_frictionloss_scale=self._draw_joints(self.cfg.joint_frictionloss_scale),
        )

    def apply(self, sample: ResetSample, targets: ResetTargets) -> Any:
        """Write ``sample`` onto ``targets``. Returns (possibly replaced) ``mpc_data``."""
        mpc_data = targets.mpc_data
        if sample.payload_kg is not None and targets.base_weight is not None:
            targets.base_weight.enabled = True
            targets.base_weight.extra_mass_kg = float(sample.payload_kg)
        if targets.command_sampler is not None:
            targets.command_sampler.scale_ranges(
                max_speed=sample.max_speed,
                max_yaw_rate=sample.max_yaw_rate,
            )
        if targets.navigator is not None:
            if sample.max_speed is not None:
                targets.navigator.max_speed = float(sample.max_speed)
            if sample.max_yaw_rate is not None:
                targets.navigator.max_yaw_rate = float(sample.max_yaw_rate)
            if sample.base_height is not None and hasattr(targets.navigator, "robot_height"):
                targets.navigator.robot_height = float(sample.base_height)
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
        if targets.model is not None:
            self._apply_joint_scales(sample, targets.model)
        return mpc_data

    def sample_and_apply(self, targets: ResetTargets) -> tuple[ResetSample, Any]:
        sample = self.sample()
        mpc_data = self.apply(sample, targets)
        return sample, mpc_data

    def _apply_joint_scales(self, sample: ResetSample, model: mujoco.MjModel) -> None:
        knobs = [(field, getattr(sample, name)) for name, field in _JOINT_FIELDS]
        if all(scales is None for _, scales in knobs):
            return
        if id(model) not in self._joint_nominal:
            hinge = np.asarray(model.jnt_type) == mujoco.mjtJoint.mjJNT_HINGE
            dofs = np.asarray(model.jnt_dofadr)[hinge]
            if dofs.size != N_JOINTS:
                raise ValueError(f"joint randomization expects {N_JOINTS} hinge joints, got {dofs.size}")
            self._joint_nominal[id(model)] = (
                dofs,
                {field: np.array(getattr(model, field)[dofs]) for field, _ in knobs},
            )
        dofs, nominal = self._joint_nominal[id(model)]
        for field, scales in knobs:
            if scales is not None:
                getattr(model, field)[dofs] = nominal[field] * np.asarray(scales)
        # Armature feeds dof_invweight0 / actuator_acc0, which scale constraint softness.
        mujoco.mj_setConst(model, mujoco.MjData(model))

    def _draw(self, spec: FloatRangeSpec) -> float | None:
        if not spec.enabled:
            return None
        if spec.log_uniform:
            log_low = np.log(spec.low)
            log_high = np.log(spec.high)
            return float(np.exp(self._rng.uniform(log_low, log_high)))
        return float(self._rng.uniform(spec.low, spec.high))

    def _draw_joints(self, spec: FloatRangeSpec) -> tuple[float, ...] | None:
        if not spec.enabled:
            return None
        return tuple(self._draw(spec) for _ in range(N_JOINTS))
