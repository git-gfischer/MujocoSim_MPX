"""Eq. (7): P = 0.5 at the configured force, and the 0.5 gate."""

import math

import numpy as np

from framework.config.sim_config.config_camurri_contact import CamurriContactConfig
from framework.estimators.camurri_contact import (
    camurri_sphere_rgba,
    estimate_camurri_contacts,
    stance_probability,
    upward_grf,
)


def test_probability_is_half_at_the_configured_force_and_gates_there():
    cfg = CamurriContactConfig()
    f_half = -cfg.beta0 / cfg.beta
    assert math.isclose(f_half, 15.0, rel_tol=1e-6)

    p = stance_probability([0.0, f_half, 200.0], cfg.beta, cfg.beta0)
    assert math.isclose(float(p[0]), 0.05, rel_tol=1e-4)
    assert math.isclose(float(p[1]), 0.5, rel_tol=1e-4)
    assert float(p[2]) > 0.99

    grf = np.zeros((4, 3))
    grf[0, 2] = 0.0
    grf[1, 2] = f_half
    grf[2, 2] = f_half + 5.0
    grf[3, 2] = 80.0
    prob, stance = estimate_camurri_contacts(grf, cfg)
    assert np.allclose(prob[:2], [0.05, 0.5], rtol=1e-4)
    # The paper puts a foot in C only when P exceeds 0.5.
    assert list(stance) == [0.0, 0.0, 1.0, 1.0]


def test_sphere_is_green_on_a_match_and_red_on_a_miss():
    match = camurri_sphere_rgba(1, 1)
    miss = camurri_sphere_rgba(1, 0)
    assert match[1] > miss[1]
    assert miss[0] > match[0]
    assert np.allclose(camurri_sphere_rgba(0, 0), match)


def test_go2_force_into_the_ground_is_stance():
    # Rough Go2 logs: stance world Z is negative, swing is about 0.
    grf = np.zeros((2, 3))
    grf[0, 2] = -80.0
    grf[1, 2] = -2.0
    _, stance = estimate_camurri_contacts(upward_grf(grf))
    assert list(stance) == [1.0, 0.0]


def test_contact_normal_uses_the_terrain_axis_not_world_z():
    cfg = CamurriContactConfig(force_axis="contact_normal")
    # 40 N straight up a 45-degree slope: world Z is only 28 N, along-normal is 40.
    grf = np.array([[0.0, 28.284, 28.284]])
    normal = np.array([[0.0, 0.70710678, 0.70710678]])
    along, _ = estimate_camurri_contacts(grf, cfg, normal)
    vertical, _ = estimate_camurri_contacts(grf, CamurriContactConfig())
    assert float(along[0]) > float(vertical[0])
