"""Unit tests for reset-time domain randomization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from mpx.config.sim_config.config_reset_randomization import (
    FloatRangeSpec,
    ResetRandomizationConfig,
    balance_reset_randomization_config,
    loco_reset_randomization_config,
)
from mpx.utils.simulation_utils.base_weight import BaseWeightForce
from mpx.utils.simulation_utils.reset_randomizer import (
    ResetRandomizer,
    ResetSample,
    ResetTargets,
)


@dataclass
class FakeMpcData:
    step_freq: float = 1.35
    duty_factor: float = 0.65

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def _all_enabled_config(**kwargs) -> ResetRandomizationConfig:
    defaults = dict(
        enabled=True,
        rng_seed=0,
        payload=FloatRangeSpec(enabled=True, low=1.0, high=2.0),
        max_speed=FloatRangeSpec(enabled=True, low=0.25, high=0.70),
        max_yaw_rate=FloatRangeSpec(enabled=True, low=0.40, high=1.20),
        step_freq=FloatRangeSpec(enabled=True, low=1.00, high=1.70),
        duty_factor=FloatRangeSpec(enabled=True, low=0.55, high=0.80),
        solref_timeconst=FloatRangeSpec(
            enabled=True, low=0.012, high=0.035, log_uniform=True
        ),
        friction=FloatRangeSpec(enabled=True, low=0.40, high=1.50),
    )
    defaults.update(kwargs)
    return ResetRandomizationConfig(**defaults)


def _targets(*, include_nav=True, include_weight=True, include_mpc=True):
    model = SimpleNamespace(
        geom_solref=np.tile(np.array([0.02, 1.0, 0.0, 0.0, 0.0]), (8, 1)).astype(
            np.float64
        ),
        geom_friction=np.tile(np.array([1.2, 0.02, 0.01]), (8, 1)).astype(np.float64),
    )
    nav = SimpleNamespace(max_speed=0.5, max_yaw_rate=0.8) if include_nav else None
    weight = (
        BaseWeightForce(enabled=False, extra_mass_kg=10.0) if include_weight else None
    )
    mpc = FakeMpcData() if include_mpc else None
    return ResetTargets(
        model=model,
        foot_geom_ids=np.array([1, 2, 3, 4]),
        base_weight=weight,
        navigator=nav,
        mpc_data=mpc,
    ), model, weight, nav, mpc


def test_disabled_master_switch_is_noop():
    cfg = _all_enabled_config(enabled=False)
    randomizer = ResetRandomizer.from_config(cfg)
    targets, model, weight, nav, mpc = _targets()

    sample, out_mpc = randomizer.sample_and_apply(targets)

    assert sample == ResetSample()
    assert weight.enabled is False
    assert weight.extra_mass_kg == 10.0
    assert nav.max_speed == 0.5
    assert nav.max_yaw_rate == 0.8
    assert out_mpc.step_freq == 1.35
    assert out_mpc.duty_factor == 0.65
    np.testing.assert_allclose(model.geom_solref[1:5, 0], 0.02)
    np.testing.assert_allclose(model.geom_friction[1:5, 0], 1.2)


def test_sample_respects_enabled_ranges():
    randomizer = ResetRandomizer.from_config(_all_enabled_config())
    for _ in range(40):
        sample = randomizer.sample()
        assert 1.0 <= sample.payload_kg <= 2.0
        assert 0.25 <= sample.max_speed <= 0.70
        assert 0.40 <= sample.max_yaw_rate <= 1.20
        assert 1.00 <= sample.step_freq <= 1.70
        assert 0.55 <= sample.duty_factor <= 0.80
        assert 0.012 <= sample.solref_timeconst <= 0.035
        assert 0.40 <= sample.friction <= 1.50


def test_disabled_knob_is_omitted_from_sample_and_metadata():
    cfg = _all_enabled_config(
        payload=FloatRangeSpec(enabled=False, low=1.0, high=2.0),
        max_speed=FloatRangeSpec(enabled=False, low=0.25, high=0.70),
    )
    sample = ResetRandomizer.from_config(cfg).sample()
    assert sample.payload_kg is None
    assert sample.max_speed is None
    assert sample.step_freq is not None
    meta = sample.to_metadata()
    assert "payload_kg" not in meta
    assert "max_speed" not in meta
    assert "step_freq" in meta


def test_missing_targets_are_skipped():
    randomizer = ResetRandomizer.from_config(_all_enabled_config())
    targets, model, weight, nav, mpc = _targets(
        include_nav=False, include_weight=False, include_mpc=False
    )
    sample, out_mpc = randomizer.sample_and_apply(targets)
    assert sample.payload_kg is not None
    assert sample.max_speed is not None
    assert sample.step_freq is not None
    assert weight is None
    assert nav is None
    assert out_mpc is None
    np.testing.assert_allclose(model.geom_solref[1:5, 0], sample.solref_timeconst)
    np.testing.assert_allclose(model.geom_friction[1:5, 0], sample.friction)


def test_solref_writes_all_feet_equally():
    randomizer = ResetRandomizer.from_config(_all_enabled_config())
    targets, model, *_ = _targets()
    sample, _ = randomizer.sample_and_apply(targets)
    np.testing.assert_allclose(model.geom_solref[1:5, 0], sample.solref_timeconst)
    np.testing.assert_allclose(model.geom_solref[1:5, 1], 1.0)
    np.testing.assert_allclose(model.geom_solref[0, 0], 0.02)
    np.testing.assert_allclose(model.geom_solref[5:, 0], 0.02)


def test_friction_writes_all_feet_sliding_mu_equally():
    randomizer = ResetRandomizer.from_config(_all_enabled_config())
    targets, model, *_ = _targets()
    sample, _ = randomizer.sample_and_apply(targets)
    np.testing.assert_allclose(model.geom_friction[1:5, 0], sample.friction)
    np.testing.assert_allclose(model.geom_friction[1:5, 1], 0.02)
    np.testing.assert_allclose(model.geom_friction[1:5, 2], 0.01)
    np.testing.assert_allclose(model.geom_friction[0, 0], 1.2)
    np.testing.assert_allclose(model.geom_friction[5:, 0], 1.2)


def test_apply_updates_payload_nav_and_mpc():
    randomizer = ResetRandomizer.from_config(_all_enabled_config())
    targets, _, weight, nav, mpc = _targets()
    sample, out_mpc = randomizer.sample_and_apply(targets)
    assert weight.enabled is True
    assert weight.extra_mass_kg == sample.payload_kg
    assert nav.max_speed == sample.max_speed
    assert nav.max_yaw_rate == sample.max_yaw_rate
    assert out_mpc is not mpc
    assert out_mpc.step_freq == sample.step_freq
    assert out_mpc.duty_factor == sample.duty_factor


def test_seeded_sampling_is_reproducible():
    a = ResetRandomizer.from_config(_all_enabled_config(rng_seed=7)).sample()
    b = ResetRandomizer.from_config(_all_enabled_config(rng_seed=7)).sample()
    assert a == b


def test_default_profiles_knob_flags():
    assert loco_reset_randomization_config.enabled is True
    assert balance_reset_randomization_config.enabled is False
    assert balance_reset_randomization_config.max_speed.enabled is False
    assert balance_reset_randomization_config.step_freq.enabled is False
    assert balance_reset_randomization_config.duty_factor.enabled is False
    assert balance_reset_randomization_config.payload.enabled is True
    assert balance_reset_randomization_config.solref_timeconst.enabled is True
    assert balance_reset_randomization_config.friction.enabled is True
    assert loco_reset_randomization_config.payload.enabled is True
    assert loco_reset_randomization_config.max_speed.enabled is True
    assert loco_reset_randomization_config.solref_timeconst.log_uniform is True
    assert loco_reset_randomization_config.friction.enabled is True
    # Nominal folders keep the foot at or above the XML value of 1.2; slippery
    # conditions get a dedicated folder rather than a 38% fall rate mixed into
    # normal locomotion (DATASET_FIX_TASKS_R2 Task R2-8 item 3).
    assert loco_reset_randomization_config.friction.low == 1.20
    assert loco_reset_randomization_config.friction.high == 2.00
    # v4 clamped the contact time constant: the v3 upper end of 0.035 s was
    # 6.6 sim steps at 200 Hz and caused the one-frame contact dropouts.
    assert loco_reset_randomization_config.solref_timeconst.low == 0.010
    assert loco_reset_randomization_config.solref_timeconst.high == 0.014
