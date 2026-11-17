"""Extreme navigation slews to a per-robot ceiling instead of stepping to it."""

import numpy as np
import pytest

from framework.navigation.extreme import EXTREME_LIMITS, ExtremeNavigator, limits_for


def test_each_robot_and_gait_has_its_own_ceiling():
    assert limits_for("go2", "trot") is EXTREME_LIMITS["go2"]["trot"]
    assert limits_for("go2").vx_mps != limits_for("b2").vx_mps
    assert limits_for("go2", "trot").vx_mps != limits_for("go2", "crawl").vx_mps
    assert limits_for("go2", "bound").vx_mps != limits_for("b2", "bound").vx_mps
    assert limits_for("spot").linear_accel_mps2 != limits_for("aliengo").linear_accel_mps2
    with pytest.raises(ValueError, match="mini_cheetah"):
        limits_for("mini_cheetah")
    with pytest.raises(ValueError, match="pace"):
        limits_for("aliengo", "pace")


def _speed_increase(previous: float, command: float) -> float:
    """How far this step moved away from zero. A reversal only counts past zero."""
    if previous * command < 0.0:
        return abs(command)
    if abs(command) > abs(previous):
        return abs(command) - abs(previous)
    return 0.0


def test_acceleration_is_limited_and_braking_is_instant():
    dt = 0.02
    nav = ExtremeNavigator("go2", dt=dt, rng=np.random.default_rng(0), hold_s=(0.2, 0.4))
    nav.reset()
    limits = nav.limits
    cap = np.array(
        [
            limits.linear_accel_mps2 * dt,
            limits.linear_accel_mps2 * dt,
            limits.yaw_accel_rps2 * dt,
        ]
    )
    previous = np.zeros(3)
    reached = np.zeros(3, dtype=bool)
    braked = False
    for _ in range(4000):
        command = nav.step()
        increase = np.array([_speed_increase(previous[i], command[i]) for i in range(3)])
        assert np.all(increase <= cap + 1e-9)
        braked = braked or bool(np.any(np.abs(command - previous) > cap + 1e-9))
        previous = command
        reached |= np.abs(command) >= np.array(
            [0.9 * limits.vx_mps, 0.9 * limits.vy_mps, 0.9 * limits.yaw_rate_rps]
        )
    assert reached.all()
    assert braked
    assert previous[0] <= limits.vx_mps + 1e-9
    assert np.abs(previous[2]) <= limits.yaw_rate_rps + 1e-9


def test_full_stop_drops_from_the_ceiling_to_zero():
    nav = ExtremeNavigator(
        "go2", dt=0.02, rng=np.random.default_rng(2), hold_s=(0.05, 0.05)
    )
    nav.reset()
    limits = nav.limits
    previous = np.zeros(3)
    dropped = False
    for _ in range(8000):
        command = nav.step()
        was_at_ceiling = (
            abs(previous[0]) >= 0.9 * limits.vx_mps
            or abs(previous[2]) >= 0.9 * limits.yaw_rate_rps
        )
        if was_at_ceiling and np.max(np.abs(command)) < 1e-9:
            dropped = True
            break
        previous = command
    assert dropped


def test_segments_include_stop_linear_yaw_and_both_signs():
    nav = ExtremeNavigator(
        "go2", dt=0.02, rng=np.random.default_rng(1), hold_s=(0.05, 0.05)
    )
    nav.reset()
    commands = np.stack([nav.step() for _ in range(8000)])
    assert np.any(np.max(np.abs(commands), axis=1) < 1e-9)
    assert commands[:, 0].min() < 0.0 and commands[:, 0].max() > 0.0
    assert commands[:, 2].min() < 0.0 and commands[:, 2].max() > 0.0
    # A yaw-only stretch leaves vx and vy at rest while yaw is near its ceiling.
    yaw_only = (np.abs(commands[:, 0]) < 1e-9) & (np.abs(commands[:, 1]) < 1e-9)
    assert np.any(yaw_only & (np.abs(commands[:, 2]) >= 0.7))
    # A linear-only stretch leaves yaw at rest while vx is near its ceiling.
    linear_only = np.abs(commands[:, 2]) < 1e-9
    assert np.any(linear_only & (np.abs(commands[:, 0]) >= 0.9))
