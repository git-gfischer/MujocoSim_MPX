"""
Tests for round 5 (DATASET_FIX_TASKS_R5).

Each test names the defect it locks down, so a future regression reports what
broke rather than only which assert failed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mpx.config.sim_config.config_dataset_bucket import (
    DatasetBucketConfig,
    dataset_collection_config,
)
from mpx.utils.dataset_collection.dataset_bucket_system import (
    DatasetBucketSystem,
    perturbation_boundaries,
    perturbation_level,
)
from mpx.utils.dataset_collection.dataset_schema import COLUMNS_BY_NAME
from mpx.utils.dataset_collection.operating_regime import (
    REGIME_ORDER,
    OperatingRegimeConfig,
    apply_dwell,
    base_tilt_deg,
    body_contact_level,
    classify,
    posture_level,
    regime_counts,
)
from mpx.utils.dataset_collection.signal_bounds import (
    NOMINAL_SCOPE,
    clipping_audit,
    load_signal_bounds,
    signal_bounds_version,
)

from mpx.utils.dataset_collection.test_round4_fixes import make_record

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── R5-2: operating regime ───────────────────────────────────────────────────

def test_a_collapsed_posture_is_not_nominal():
    """
    Episode 00023 of the audited run terminated ``goal_reached`` with 76.2% of
    frames below 0.20 m — a full minute of walking in a collapsed posture,
    indistinguishable from a healthy run by anything the dataset recorded.
    """
    height = np.full(200, 0.190)
    regime = classify(height, np.zeros(200), np.zeros(200))
    assert (regime == "degraded").all()


def test_the_crash_predicate_alone_would_have_missed_it():
    """
    ``crash_height_m = 0.15`` / ``crash_tilt_deg = 60`` are TERMINATE conditions.
    The degradation profile shows the robot below 0.195 m and past 12 deg of tilt
    for more than a second before either trips.
    """
    height = np.concatenate([np.full(60, 0.25), np.linspace(0.24, 0.14, 40)])
    tilt = np.concatenate([np.zeros(60), np.linspace(2.0, 50.0, 40)])
    regime = classify(height, tilt, np.zeros(100))

    # The old predicate fires only in the last handful of frames.
    old_predicate = (height < 0.15) | (tilt > 60.0)
    assert old_predicate.sum() < 5
    # The graded one sees the approach.
    assert (regime != "nominal").sum() > 25
    assert regime[-1] == "failed"


def test_regime_is_the_worst_of_the_signals():
    height = np.full(50, 0.25)
    tilt = np.zeros(50)
    tilt[10:40] = 30.0                       # severe on tilt alone
    regime = classify(height, tilt, np.zeros(50))
    assert set(regime[15:35]) == {"severe"}


def test_body_contact_forces_severe_and_survives_the_dwell_filter():
    """If the body is touching the ground, the posture thresholds are moot."""
    body = np.zeros(40)
    body[20] = 40.0                          # a single control frame
    regime = classify(np.full(40, 0.25), np.zeros(40), body)
    assert regime[20] == "severe", (
        "a dwell filter must never label a frame nominal while the robot's "
        "body is carrying load"
    )
    assert regime[19] == "nominal" and regime[21] == "nominal"


def test_the_dwell_filter_rejects_a_single_noisy_frame():
    level = np.zeros(20, dtype=np.int8)
    level[10] = 1                            # one frame of posture noise
    assert (apply_dwell(level, dwell_steps=3) == 0).all()


def test_failed_is_absorbing():
    """The robot does not get back up; this is the v4 post_failure latch."""
    height = np.full(30, 0.25)
    height[10:14] = 0.10
    regime = classify(height, np.zeros(30), np.zeros(30))
    assert regime[-1] == "failed"
    assert (regime[np.flatnonzero(regime == "failed")[0]:] == "failed").all()


def test_tilt_is_the_total_angle_not_roll_or_pitch_alone():
    level = np.array([[1.0, 0.0, 0.0, 0.0]])            # upright
    assert base_tilt_deg(level)[0] == pytest.approx(0.0, abs=1e-6)
    rolled = np.array([[np.cos(0.25), np.sin(0.25), 0.0, 0.0]])
    assert base_tilt_deg(rolled)[0] == pytest.approx(np.degrees(0.5), abs=0.1)


def test_regime_columns_exist_with_the_right_roles():
    assert COLUMNS_BY_NAME["operating_regime"].role == "context"
    assert COLUMNS_BY_NAME["cmd_tracking_error"].role == "context"
    # R5-8: readable straight off the episode file, no join required.
    for name in ("post_failure", "valid"):
        assert COLUMNS_BY_NAME[name].role == "context"
    for name in ("base_tilt_deg", "non_foot_contact_n", "base_height_terrain"):
        assert COLUMNS_BY_NAME[name].role == "privileged"


def test_valid_means_recognisable_locomotion_not_merely_upright():
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    record = make_record("ep", n_steps=120)
    record.arrays["base_height_terrain"][40:80] = 0.16     # severe
    bucket.add_episode(record)

    rows = {r["t"]: r for r in bucket.index_rows(shuffle=False)}
    assert rows[10]["operating_regime"] == "nominal" and rows[10]["valid"]
    severe = [r for r in rows.values() if r["operating_regime"] == "severe"]
    assert severe, "the collapsed stretch was not annotated"
    assert not any(r["valid"] for r in severe), (
        "severe frames are kept and indexed, but they are not training data"
    )


def test_the_regime_is_tracked_per_bucket_not_in_the_key():
    """Four levels in the key would multiply it and leave most cells empty."""
    from mpx.utils.dataset_collection.dataset_bucket_system import BucketKey

    assert "operating_regime" not in BucketKey.__dataclass_fields__
    bucket = DatasetBucketSystem(dataset_summary_path=None)
    record = make_record("ep", n_steps=120)
    record.arrays["base_height_terrain"][60:] = 0.16
    bucket.add_episode(record)
    shares = bucket.nominal_fraction_by_bucket()
    assert shares and all(0.0 <= v <= 1.0 for v in shares.values())
    assert sum(bucket.regime_counts().values()) == bucket.total_samples_stored


# ── R5-3: the clipping audit measures the data you train on ──────────────────

def test_clipping_audit_is_scoped_and_reports_the_scope():
    bounds = load_signal_bounds()
    stance = bounds["stance_posture_rad"]
    row = np.tile([stance["HAA"], stance["HFE"], stance["KFE"]], 4)
    arrays = {"joint_pos": np.tile(row, (100, 1)).astype(np.float32)}
    # Push the last 20 rows out of range — as if they were crash frames.
    arrays["joint_pos"][80:, 0] = 99.0
    mask = np.ones(100, dtype=bool)
    mask[80:] = False

    everything = clipping_audit(arrays, bounds)
    nominal = clipping_audit(arrays, bounds, mask=mask, scope=NOMINAL_SCOPE)

    assert everything["per_row_any"]["joint_pos"] == pytest.approx(0.20)
    assert nominal["per_row_any"]["joint_pos"] == 0.0
    assert "nominal" in nominal["scope"], (
        "the scope has to travel with the number: 83.1% of the clipping rows on "
        "the audited run were crash frames"
    )


# ── R5-4: every declared perturbation level must be reachable ────────────────

def test_perturbation_boundaries_come_from_the_sampler_range():
    """
    A fixed 0.35 x body weight put `large` at 61.6 N against a sampler maximum
    of 50.0 N, so the level was structurally empty and BucketKey carried a value
    that could never occur.
    """
    config = DatasetBucketConfig(perturbation_force_range_n=(10.0, 50.0))
    floor, mid = perturbation_boundaries(176.0, config)
    assert floor == pytest.approx(17.6)
    assert mid == pytest.approx(33.8)
    # Both bins sit inside the sampler's reachable range.
    assert 10.0 <= mid <= 50.0


def test_every_perturbation_level_is_reachable_by_the_sampler():
    config = dataset_collection_config.bucket
    low, high = config.perturbation_force_range_n
    floor, mid = perturbation_boundaries(176.0, config)
    reachable = {
        perturbation_level(f, 176.0, config)
        for f in np.linspace(0.0, high, 200)
    }
    assert reachable == set(("none", "small", "large")), (
        f"sampler range ({low}, {high}) cannot produce every declared level; "
        f"boundaries are {floor:.1f} / {mid:.1f} N"
    )


def test_perturbation_floor_still_scales_with_payload():
    light, heavy = 176.0, 176.0 + 50.0 * 9.81
    assert perturbation_level(25.0, light) == "small"
    assert perturbation_level(25.0, heavy) == "none"


# ── R5-1: the bounds file is a rendering of the YAMLs ────────────────────────

def test_signal_bounds_are_rendered_from_the_yaml():
    bounds = load_signal_bounds()
    assert signal_bounds_version(bounds) >= 5
    assert bounds["source"] == "go2_constrains.yaml + go2_stance.yaml"
    for key in ("constraints_sha256", "stance_sha256"):
        assert bounds.get(key), f"{key} missing — the provenance is the point"


def test_bounds_are_per_leg():
    """``foot_pos.FL.x`` and ``foot_pos.RL.x`` are genuinely different."""
    bounds = load_signal_bounds()["signals"]["foot_pos_base"]["bounds"]
    assert set(bounds) == {"FL", "FR", "RL", "RR"}
    assert bounds["FL"]["x"] != bounds["RL"]["x"], (
        "front feet reach forward and hind feet reach back; a union of the two "
        "wastes most of the range"
    )


def test_left_and_right_mirror_exactly_in_y():
    """The sagittal-mirror augmentation MI-HGNN and ECNN use depends on it."""
    signals = load_signal_bounds()["signals"]
    for name in ("foot_pos_base", "foot_vel_base", "grf_base"):
        bounds = signals[name]["bounds"]
        for left, right in (("FL", "FR"), ("RL", "RR")):
            assert bounds[left]["y"] == [-b for b in reversed(bounds[right]["y"])]
            assert bounds[left]["x"] == bounds[right]["x"]


def test_hfe_follows_the_model_not_the_original_yaml():
    """
    The original YAML and the MJCF were mirror images on HFE, and the YAML sign
    clipped 11.4% of the most heavily loaded joint in the robot.
    """
    signals = load_signal_bounds()["signals"]
    hfe = signals["joint_pos"]["bounds"]["FL"]["HFE"]
    stance = signals["joint_pos"]["stance_value"]["FL"]["HFE"]
    assert hfe[0] < stance < hfe[1]
    assert stance > 0, "stance HFE is positive in the model's convention"


def test_imu_bounds_carry_gravity():
    """YAML ``lin_acc`` is kinematic; the stored ``imu_acc_body`` includes g."""
    bounds = load_signal_bounds()
    acc = bounds["signals"]["imu_acc_body"]
    assert acc["bounds"]["z"][0] == pytest.approx(-21.0 + 9.81)
    assert acc["stance_value"]["z"] == pytest.approx(9.81)
    assert "9.81" in bounds["imu_acc_gravity_handling"]


def test_sensor_full_scale_is_not_a_normalisation_range():
    """It must never reach ``ImageEncoder(constraints=...)``."""
    bounds = load_signal_bounds()
    assert "sensor_full_scale" in bounds
    assert "sensor_full_scale" not in bounds["signals"]
    for signal in bounds["signals"].values():
        assert "sensor_full_scale" not in signal["bounds"]


def test_stance_values_are_measured_not_zero():
    """
    ``ImageEncoder`` centres its trajectory canvas on ``stance_value``; 0.0 for
    ``foot_pos_base.z`` against a measured -0.23 m wasted half the canvas.
    """
    stance = load_signal_bounds()["signals"]["foot_pos_base"]["stance_value"]
    assert stance["FL"]["z"] == pytest.approx(-0.230)
