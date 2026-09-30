"""Respawn picks one swing foot and keeps its target in that foot's quadrant."""

import numpy as np

from mpx.simulators.quadruped.quad_3balance_multiEnv import (
    _STUCK_GRACE_S,
    _STUCK_STILL_S,
    _mirror_swing_offset,
    _swing_contact_mask,
    _update_stuck,
)


def test_contact_mask_clears_only_the_swing_foot():
    assert _swing_contact_mask(4, 0).tolist() == [0, 1, 1, 1]
    assert _swing_contact_mask(4, 1).tolist() == [1, 0, 1, 1]
    assert _swing_contact_mask(4, 3).tolist() == [1, 1, 1, 0]


def test_mirror_puts_the_sample_in_the_nominal_quadrant():
    # FL +x+y, FR +x-y, RL -x+y, RR -x-y
    legs = np.array([
        0.2, 0.1, 0.0,
        0.2, -0.1, 0.0,
        -0.2, 0.1, 0.0,
        -0.2, -0.1, 0.0,
    ])
    sample = np.array([-0.3, 0.25, -0.22])
    fr = _mirror_swing_offset(legs, 1, sample)
    rr = _mirror_swing_offset(legs, 3, sample)
    assert fr[0] > 0 and fr[1] < 0
    assert rr[0] < 0 and rr[1] < 0
    assert fr[2] == sample[2]


def test_settled_miss_resets_only_after_the_grace_period():
    still = np.zeros(18)
    moving = np.zeros(18)
    moving[0] = 0.4
    since, outside, stuck = _update_stuck(
        arrived=False, qvel=still, dt=_STUCK_GRACE_S - 0.1,
        since_reset_s=0.0, still_outside_s=0.0,
    )
    assert not stuck
    since, outside, stuck = _update_stuck(
        arrived=False, qvel=still, dt=_STUCK_STILL_S,
        since_reset_s=since, still_outside_s=outside,
    )
    assert stuck
    _, _, stuck = _update_stuck(
        arrived=False, qvel=moving, dt=5.0,
        since_reset_s=10.0, still_outside_s=10.0,
    )
    assert not stuck
    _, _, stuck = _update_stuck(
        arrived=True, qvel=still, dt=5.0,
        since_reset_s=10.0, still_outside_s=10.0,
    )
    assert not stuck
