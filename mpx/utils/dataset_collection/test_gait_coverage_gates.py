"""Coverage gates that were fit to trot must use the run's gait."""

from __future__ import annotations

import numpy as np
import pytest

import tools.validate_run as validate
from tools.validate_run import (
    ATTITUDE_RMS_MAX_DEG,
    CLEAN_NON_NOMINAL_MAX,
    SPEED_P95_MIN_MPS,
    CHECKS,
)


def _check(name: str):
    return next(fn for n, fn, _severity in CHECKS if n == name)


def test_unknown_gait_keeps_the_trot_limits():
    assert validate._for_gait(SPEED_P95_MIN_MPS, "") == 0.7
    assert validate._for_gait(SPEED_P95_MIN_MPS, "amble") == 0.7
    assert validate._for_gait(CLEAN_NON_NOMINAL_MAX, "") == 0.10
    assert validate._for_gait(ATTITUDE_RMS_MAX_DEG, "") == (3.0, 3.5)


def test_each_gait_has_an_explicit_row():
    for gait in ("trot", "pace", "crawl", "bound"):
        assert gait in SPEED_P95_MIN_MPS
        assert gait in CLEAN_NON_NOMINAL_MAX
        assert gait in ATTITUDE_RMS_MAX_DEG


def test_bound_speed_passes_and_the_same_trace_fails_as_trot(monkeypatch):
    # 0.39 m/s at the 95th percentile: the bound folder, under the trot floor.
    speed = np.zeros((100, 3))
    speed[:, 0] = 0.39
    monkeypatch.setattr(validate, "_cat", lambda run, column: speed)

    def metadata(run):
        return {"gait": run}

    monkeypatch.setattr(validate, "_metadata", metadata)
    _check("speed_range_covered")("bound")
    with pytest.raises(AssertionError, match="for trot"):
        _check("speed_range_covered")("trot")


def test_bound_clean_fraction_passes_and_fails_as_trot(monkeypatch):
    # 11% non-nominal on a clean episode. The crash-tail gate stays at 80%.
    rows = [{"episode_id": "e0", "t": i, "operating_regime": "nominal"} for i in range(89)]
    rows += [{"episode_id": "e0", "t": 89 + i, "operating_regime": "degraded"} for i in range(11)]
    episodes = [{"episode_id": "e0", "terminate_by": "time"}]

    def table(path):
        return episodes if path.endswith("episodes.parquet") else rows

    monkeypatch.setattr(validate, "_table", table)
    monkeypatch.setattr(validate, "_metadata", lambda run: {"gait": run})
    _check("operating_regime_separates_crashes")("bound")
    with pytest.raises(AssertionError, match="for trot"):
        _check("operating_regime_separates_crashes")("trot")


def test_bound_attitude_passes_and_fails_as_trot(monkeypatch):
    error = {
        "roll_rms": 5.07,
        "pitch_rms": 9.36,
        "pitch_median_bias": 0.0,
        "roll_median_bias": -1.01,
    }

    def metadata(run):
        return {"gait": run, "sampler": {"attitude_estimator": {"measured_error_deg": error}}}

    monkeypatch.setattr(validate, "_metadata", metadata)
    _check("attitude_estimator_error_recorded")("bound")
    with pytest.raises(AssertionError, match="for trot"):
        _check("attitude_estimator_error_recorded")("trot")


def test_balance_with_no_reset_knobs_is_one_group(monkeypatch):
    rows = [
        {"randomization_group_id": "same", "reset_randomization": "{}"}
        for _ in range(46)
    ]
    monkeypatch.setattr(validate, "_table", lambda path: rows)
    monkeypatch.setattr(validate, "_metadata", lambda run: {"gait": run})
    _check("randomization_is_per_episode")("balance")
    with pytest.raises(AssertionError, match="want >0.95"):
        _check("randomization_is_per_episode")("trot")


def test_balance_reused_knobs_still_fail(monkeypatch):
    rows = [
        {"randomization_group_id": "same", "reset_randomization": '{"friction": 1.2}'}
        for _ in range(10)
    ]
    monkeypatch.setattr(validate, "_table", lambda path: rows)
    monkeypatch.setattr(validate, "_metadata", lambda run: {"gait": "balance"})
    with pytest.raises(AssertionError, match="want >0.95"):
        _check("randomization_is_per_episode")("balance")


def test_balance_tripod_schedule_is_not_judged_as_a_gait_timer(monkeypatch):
    # Swing foot stays on the ground: mismatch would be ~1, which the
    # locomotion band calls a broken column. The plan is the mask.
    rows = [{"mode": "stand_3leg_FL"}]
    schedule = np.tile(np.array([0, 1, 1, 1], dtype=np.uint8), (20, 1))
    monkeypatch.setattr(validate, "_table", lambda path: rows)
    monkeypatch.setattr(validate, "_cat", lambda run, column: schedule)
    monkeypatch.setattr(validate, "_metadata", lambda run: {"gait": "balance"})
    _check("schedule_mismatch_present")("balance")


def test_empty_grf_window_does_not_crash_the_probe():
    from tools.difficulty_probe import _rmse_1d

    features = np.zeros((0, 18))
    target = np.zeros((0,))
    held_out = np.zeros((0,), dtype=bool)
    assert np.isnan(_rmse_1d(features, target, held_out))


def _as_tripod(monkeypatch, rows):
    monkeypatch.setattr(validate, "_table", lambda path: rows)
    monkeypatch.setattr(validate, "_metadata", lambda run: {"gait": "balance"})


def test_tripod_coverage_gates_do_not_apply_to_trot_or_four_leg(monkeypatch):
    # The measured tripod slice: 2.0 s, 0.63 m/s, roll 7.4°. Trot and
    # stand_4leg must still fail the locomotion floors.
    speed = np.zeros((100, 3))
    speed[:, 0] = 0.63
    monkeypatch.setattr(validate, "_cat", lambda run, column: speed)

    _as_tripod(monkeypatch, [{"mode": "stand_3leg_FL", "duration_s": 2.0}])
    _check("speed_range_covered")("tripod")
    _check("episodes_long_enough")("tripod")
    _check("command_coverage")("tripod")

    monkeypatch.setattr(
        validate, "_table",
        lambda path: [{"mode": "stand_4leg", "duration_s": 2.0}],
    )
    with pytest.raises(AssertionError, match="for balance"):
        _check("speed_range_covered")("four")
    with pytest.raises(AssertionError, match=">=30"):
        _check("episodes_long_enough")("four")

    monkeypatch.setattr(validate, "_metadata", lambda run: {"gait": "trot"})
    monkeypatch.setattr(
        validate, "_table",
        lambda path: [{"mode": "", "duration_s": 2.0}],
    )
    with pytest.raises(AssertionError, match="for trot"):
        _check("speed_range_covered")("trot")
    with pytest.raises(AssertionError, match=">=30"):
        _check("episodes_long_enough")("trot")
    with pytest.raises(AssertionError, match="never turns"):
        _check("command_coverage")("trot")


def test_tripod_attitude_and_height_keep_the_locomotion_floors(monkeypatch):
    error = {
        "roll_rms": 7.36,
        "pitch_rms": 2.79,
        "pitch_median_bias": 0.0,
    }
    labels = np.zeros((10, 4), dtype=np.uint8)
    labels[:, 0] = 1
    height = labels.copy()
    # 88% agreement: flip the first row.
    height[0] = 1 - height[0]

    def cat(run, column):
        return height if column == "contact_from_height" else labels

    monkeypatch.setattr(validate, "_cat", cat)
    monkeypatch.setattr(
        validate, "_metadata",
        lambda run: {
            "gait": "balance",
            "sampler": {"attitude_estimator": {"measured_error_deg": error}},
        },
    )
    monkeypatch.setattr(
        validate, "_table", lambda path: [{"mode": "stand_3leg_RR"}]
    )
    _check("attitude_estimator_error_recorded")("tripod")
    _check("height_label_agrees_with_grf_label")("tripod")

    monkeypatch.setattr(
        validate, "_table", lambda path: [{"mode": "stand_4leg"}]
    )
    with pytest.raises(AssertionError, match="for balance"):
        _check("attitude_estimator_error_recorded")("four")
    with pytest.raises(AssertionError, match="90%"):
        _check("height_label_agrees_with_grf_label")("four")


def test_tripod_clip_gate_does_not_widen_trot_or_four_leg(monkeypatch):
    audit = {
        "definition": "x",
        "scope": "nominal",
        "non_nominal_per_element": {},
        "per_element": {"grf_base": 0.17, "foot_pos_base": 0.036},
        "per_row_any": {"grf_base": 0.97, "foot_pos_base": 0.30},
    }

    def metadata(run):
        gait = "trot" if run == "trot" else "balance"
        return {"gait": gait, "clipping_audit": audit}

    monkeypatch.setattr(validate, "_metadata", metadata)
    monkeypatch.setattr(
        validate, "_table", lambda path: [{"mode": "stand_3leg_FL"}]
    )
    _check("no_channel_clipping")("tripod")

    monkeypatch.setattr(
        validate, "_table", lambda path: [{"mode": "stand_4leg"}]
    )
    with pytest.raises(AssertionError, match="grf_base"):
        _check("no_channel_clipping")("four")
    with pytest.raises(AssertionError, match="grf_base"):
        _check("no_channel_clipping")("trot")


def test_tripod_skips_the_locomotion_difficulty_probe(monkeypatch):
    import tools.difficulty_probe as probe

    def boom(run):
        raise AssertionError("locomotion probe must not run on 3-leg balance")

    monkeypatch.setattr(probe, "probe_run", boom)
    monkeypatch.setattr(validate, "_metadata", lambda run: {"gait": "balance"})
    monkeypatch.setattr(validate, "_table", lambda path: [{"mode": "stand_3leg_FL"}])
    _check("task_not_saturated")("tripod")

    monkeypatch.setattr(validate, "_table", lambda path: [{"mode": "stand_4leg"}])
    with pytest.raises(AssertionError, match="must not run"):
        _check("task_not_saturated")("four")


def test_empty_classifier_test_slice_does_not_crash():
    from tools.difficulty_probe import _accuracy_1d

    features = np.zeros((4, 18))
    labels = np.array([0, 1, 0, 1])
    held_out = np.zeros(4, dtype=bool)
    assert np.isnan(_accuracy_1d(features, labels, held_out))


def test_balance_all_stance_schedule_fails(monkeypatch):
    rows = [{"mode": "stand_3leg_FR"}]
    schedule = np.ones((20, 4), dtype=np.uint8)
    monkeypatch.setattr(validate, "_table", lambda path: rows)
    monkeypatch.setattr(validate, "_cat", lambda run, column: schedule)
    monkeypatch.setattr(validate, "_metadata", lambda run: {"gait": "balance"})
    with pytest.raises(AssertionError, match="tripod"):
        _check("schedule_mismatch_present")("balance")
