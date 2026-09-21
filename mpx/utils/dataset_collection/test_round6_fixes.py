"""
Tests for round 6 (DATASET_FIX_TASKS_R6).

Each test names the defect it locks down, so a future regression reports what
broke rather than only which assert failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from mpx.utils.dataset_collection.operating_regime import (
    OperatingRegimeConfig,
    apply_dwell,
    classify,
)
from mpx.utils.dataset_collection.signal_bounds import (
    load_signal_bounds,
    signal_bounds_version,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── R6-1: the narrowed rails, and the recorded exceptions ────────────────────

def test_bounds_are_version_six():
    assert signal_bounds_version(load_signal_bounds()) >= 6


def test_tangential_grf_rails_were_narrowed():
    """
    The v5 rails came from the measured MAXIMUM, but the tangential distribution
    is heavy-tailed by ~3x (p99.9 ~ +-60 N, max ~ +-178 N). Covering the impact
    transients cost half the contrast for the stance bulk, which is what the GAF,
    spike and slope encoders actually read.
    """
    grf = load_signal_bounds()["signals"]["grf_base"]["bounds"]
    for leg in ("FL", "FR"):
        assert grf[leg]["x"] == [-90.0, 90.0]
        assert grf[leg]["y"] == [-90.0, 90.0]
    for leg in ("RL", "RR"):
        assert grf[leg]["x"] == [-70.0, 70.0]
        assert grf[leg]["y"] == [-70.0, 70.0]


def test_the_grf_load_bearing_axis_is_left_generous():
    """
    z is the load-bearing axis AND the regression target, so clipping its peaks
    corrupts the label rather than merely the image. Tighten only after the
    stairs pilot confirms vertical loads do not rise.
    """
    grf = load_signal_bounds()["signals"]["grf_base"]["bounds"]
    assert grf["FL"]["z"] == [-30.0, 340.0]
    assert grf["RL"]["z"] == [-30.0, 280.0]


def test_narrowing_preserved_left_right_mirror_symmetry():
    from tools.build_signal_bounds import assert_mirror_symmetry

    assert_mirror_symmetry(load_signal_bounds())     # raises if broken


def test_low_contrast_exceptions_are_recorded_not_left_open():
    """
    A channel below the span gate on purpose has to be a decision, or it gets
    re-reported as a finding on every run until someone silently narrows it.
    """
    from tools.build_signal_bounds import _is_accepted

    payload = load_signal_bounds()
    accepted = payload["low_contrast_accepted"]
    assert accepted, "low_contrast_accepted is empty"
    for pattern, reason in accepted.items():
        assert len(reason) > 40, f"{pattern} has no stated reason"

    assert _is_accepted("joint_pos.FL.HFE", payload)
    assert _is_accepted("joint_pos.RL.HAA", payload)
    assert _is_accepted("base_lin_vel.x", payload)
    # The ones v5.1 actually fixed must NOT be waved through.
    assert not _is_accepted("grf_base.RR.x", payload)
    assert not _is_accepted("base_lin_vel.z", payload)


def test_base_lin_vel_x_stays_wide_for_reverse():
    """Reverse is absent from the reconciliation run and will fill the negative side."""
    bounds = load_signal_bounds()["signals"]["base_lin_vel"]["bounds"]
    assert bounds["x"] == [-1.2, 1.2]
    assert bounds["y"] == [-0.7, 0.7]
    assert bounds["z"] == [-0.35, 0.35]


# ── R6-2a: regime chatter is about run length, not transition count ──────────

def test_the_dwell_filter_cannot_emit_a_short_interior_run():
    """
    This is the property the chatter gate rests on. If it ever stops holding,
    the gate's attribution ("a short run must come from the body-contact floor")
    is wrong and the gate is meaningless.
    """
    rng = np.random.default_rng(0)
    dwell = 3
    for _ in range(2000):
        raw = rng.integers(0, 4, size=80).astype(np.int8)
        out = apply_dwell(raw, dwell)
        starts = [0] + [i for i in range(1, len(out)) if out[i] != out[i - 1]]
        segments = list(zip(starts, starts[1:] + [len(out)]))
        for index, (start, end) in enumerate(segments[1:-1], start=1):
            assert end - start >= dwell, (
                f"interior run of {end - start} < dwell {dwell}: {out.tolist()}"
            )


def test_only_the_body_contact_floor_makes_a_short_run():
    """
    The floor bypasses the dwell filter on purpose, so it is the one thing that
    can truncate a run below the dwell — which is exactly why a short run with no
    body contact nearby is a bug rather than a property of the robot.
    """
    config = OperatingRegimeConfig()
    height = np.full(40, 0.25)
    tilt = np.zeros(40)
    body = np.zeros(40)
    height[20:] = 0.18                       # nominal -> degraded, dwell-filtered
    body[22] = 200.0                         # truncates the degraded run
    regime = classify(height, tilt, body, config=config)

    starts = [0] + [i for i in range(1, 40) if regime[i] != regime[i - 1]]
    segments = list(zip(starts, starts[1:] + [40]))
    short = [
        (s, e) for i, (s, e) in enumerate(segments)
        if e - s < config.dwell_steps and 0 < i < len(segments) - 1
    ]
    assert short, "the fixture did not produce a short run"
    for start, end in short:
        window = body[max(start - config.dwell_steps, 0) : end + config.dwell_steps]
        assert (window > config.non_foot_contact_force_n).any(), (
            "a short interior run with no body contact means the dwell filter "
            "is not being applied"
        )


def test_sustained_oscillation_is_recorded_not_suppressed():
    """
    A robot genuinely oscillating around the 0.195 m rail produces sustained
    blocks, and the annotation SHOULD record them. Episode 00013 of the audited
    run flips 79 times in 1,277 steps and every flip is a real block — the old
    ``n/20`` transition gate called that chatter.
    """
    height = np.concatenate(
        [np.full(10, 0.25), np.full(10, 0.18)] * 8
    )
    regime = classify(height, np.zeros(len(height)), np.zeros(len(height)))
    changes = sum(1 for a, b in zip(regime, regime[1:]) if a != b)
    assert changes >= 10, "the oscillation was suppressed"

    starts = [0] + [i for i in range(1, len(regime)) if regime[i] != regime[i - 1]]
    segments = list(zip(starts, starts[1:] + [len(regime)]))
    interior = [e - s for s, e in segments[1:-1]]
    assert all(length >= 3 for length in interior), (
        "the blocks are sustained, so none of them is chatter"
    )


# ── R6-2b: yaw invariance is a reduction, not an absolute ────────────────────

def test_the_body_frame_removes_heading_and_the_world_frame_does_not():
    """
    ``grf_base`` is yaw-invariant by construction, so the world-frame column is
    the control. An absolute threshold on the base correlation just measures how
    many episodes were collected.
    """
    rng = np.random.default_rng(0)
    n = 4000
    yaw = rng.uniform(-np.pi, np.pi, n)
    # A constant force in the BODY frame, pushed out to world through the yaw.
    body = np.tile([12.0, -3.0, 80.0], (n, 1)) + rng.normal(0, 0.5, (n, 3))
    world = np.stack(
        [
            np.cos(yaw) * body[:, 0] - np.sin(yaw) * body[:, 1],
            np.sin(yaw) * body[:, 0] + np.cos(yaw) * body[:, 1],
            body[:, 2],
        ],
        axis=1,
    )
    heading = np.cos(yaw)

    def max_abs_corr(values: np.ndarray) -> float:
        return max(
            abs(float(np.corrcoef(heading, values[:, i])[0, 1]))
            for i in range(values.shape[1])
        )

    base_corr = max_abs_corr(body)
    world_corr = max_abs_corr(world)
    assert base_corr < 0.5 * world_corr, (base_corr, world_corr)


# ── R6-2c: the fold machinery ────────────────────────────────────────────────

def test_every_episode_is_held_out_exactly_once():
    from tools.difficulty_probe import _episode_fold_masks

    episode_ids = np.asarray(
        [f"ep_{i:02d}" for i in range(7) for _ in range(30)], dtype=object
    )
    masks = _episode_fold_masks(episode_ids, n_folds=5)
    assert len(masks) == 5
    held = np.zeros(len(episode_ids), dtype=int)
    for mask in masks:
        held += mask.astype(int)
    assert (held == 1).all(), "a row is held out in zero or several folds"


# ── Appendix: the split must not concentrate the crashes ─────────────────────

def _episode_row(episode_id: str, steps: int, crash: bool, frac_nominal: float):
    return {
        "episode_id": episode_id,
        "randomization_group_id": episode_id,
        "n_steps": steps,
        "terrain": "flat",
        "gait": "trot",
        "terminate_reason": "crash" if crash else "goal_reached",
        "frac_nominal": frac_nominal,
        "run_dir": "run",
    }


def test_crash_episodes_do_not_all_land_in_val():
    """
    The audited run put all four val episodes in the crash stratum, leaving val
    46% nominal against 94% for train and test — so early stopping and model
    selection would have been tuned on the fallen-robot regime. Crash episodes
    are short, val has the smallest quota so it fills last and with the smallest
    groups, and "smallest" and "crash" are the same set.
    """
    from tools.make_manifest import TARGET, _group_episodes, assign_groups

    rows = [
        _episode_row(f"ok_{i:02d}", 1800, False, 0.97) for i in range(15)
    ] + [
        _episode_row(f"crash_{i:02d}", 300, True, 0.45) for i in range(10)
    ]
    groups = _group_episodes(rows)
    assignment = assign_groups(
        groups, policy="stratified_by_group", test_terrain=None,
        test_gait=None, target=TARGET,
    )
    per_split: dict = {}
    for gid, split in assignment.items():
        per_split.setdefault(split, []).append(gid)

    assert set(per_split) == {"train", "val", "test"}
    for split, members in per_split.items():
        crashes = sum(1 for gid in members if gid.startswith("crash"))
        assert crashes < len(members), (
            f"split '{split}' is entirely crash episodes: {members}"
        )


def test_the_manifest_flags_a_regime_imbalanced_split(tmp_path, monkeypatch):
    """Assert the OUTCOME: it catches any stratifier bug, whatever the mechanism."""
    import tools.make_manifest as manifest

    rows = [
        _episode_row(f"ok_{i:02d}", 1800, False, 0.97) for i in range(15)
    ] + [
        _episode_row(f"crash_{i:02d}", 300, True, 0.10) for i in range(10)
    ]
    monkeypatch.setattr(manifest, "_read_episode_tables", lambda root: rows)
    built = manifest.build_manifest(tmp_path)

    overall = built["frac_nominal"]
    for split, entry in built["splits"].items():
        drift = abs(entry["frac_nominal"] - overall)
        flagged = any(
            f"split '{split}'" in w and "nominal" in w for w in built["warnings"]
        )
        assert (drift <= manifest.MAX_FRAC_NOMINAL_DRIFT) or flagged, (
            f"split '{split}' drifts {drift:.2f} and nothing warned"
        )
