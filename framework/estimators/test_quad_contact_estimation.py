"""Contact-frame axes are stored as rows. The world force must use the transpose."""

import numpy as np

from framework.estimators.quad_contact_estimation import _contact_force_world


def test_a_tangential_force_follows_the_stored_axis_not_its_transpose():
    # Yawed 30 deg. At 90 deg the second row equals the second column, so
    # both multiplies agree and the test would not catch the transpose.
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    frame = np.array([0.0, 0.0, 1.0, c, s, 0.0, -s, c, 0.0])
    force_local = np.array([0.0, 5.0, 0.0])
    world = _contact_force_world(frame, force_local)
    assert np.allclose(world, [5.0 * c, 5.0 * s, 0.0])
    reflected = frame.reshape(3, 3) @ force_local
    assert not np.allclose(reflected, world)
