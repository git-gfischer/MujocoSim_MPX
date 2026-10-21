import numpy as np

from mpx.estimators.foot_force_jacobian import format_jacobian_grf, jacobian_foot_force


def test_identity_jacobian_matches_solver_sign():
    force = jacobian_foot_force(
        tau=np.array([0.0, 0.0, -50.0]),
        J=np.eye(3),
        damping=0.0,
    )
    assert force.shape == (3,)
    assert np.allclose(force, [0.0, 0.0, -50.0])


def test_rank_deficient_jacobian_stays_finite():
    J = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    force = jacobian_foot_force(tau=np.array([1.0, 2.0, 3.0]), J=J)
    assert force.shape == (3,)
    assert np.isfinite(force).all()


def test_bias_changes_the_force():
    tau = np.array([0.0, 0.0, -50.0])
    J = np.eye(3)
    plain = jacobian_foot_force(tau, J, damping=0.0)
    biased = jacobian_foot_force(tau, J, tau_bias=np.array([0.0, 0.0, 10.0]), damping=0.0)
    assert not np.allclose(plain, biased)
    assert np.allclose(biased, [0.0, 0.0, -60.0])


def test_format_prints_solver_then_estimate_xyz():
    solver = np.zeros((4, 3))
    estimated = np.zeros((4, 3))
    solver[0] = (1.0, 2.0, -3.0)
    estimated[0] = (4.0, 5.0, -6.0)
    line = format_jacobian_grf(solver, estimated)
    assert line.startswith("Jacobian GRF FL=(1.0 2.0 -3.0)/(4.0 5.0 -6.0)")
    assert "FR=" in line and "RL=" in line and "RR=" in line


from mpx.config.sim_config.config_foot_force_jacobian import FootForceJacobianConfig
from mpx.estimators.foot_force_jacobian import estimate_foot_force_jacobian


def test_disabled_returns_zeros_without_a_model():
    cfg = FootForceJacobianConfig(enabled=False)
    force = estimate_foot_force_jacobian(None, None, [0, 1, 2, 3], cfg)
    assert force.shape == (4, 3)
    assert force.dtype == np.float64
    assert np.all(force == 0.0)
