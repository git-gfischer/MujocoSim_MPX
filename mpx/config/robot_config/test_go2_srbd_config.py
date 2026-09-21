"""Go2 SRBD config layout (no JAX compile of the QP)."""

import pytest

from mpx.config.robot_config.config_go2 import (
    Go2MpcModel,
    Go2Mode,
    go2_config,
)


def test_go2_srbd_state_and_cost_layout():
    cfg = go2_config(Go2Mode.LOCOMOTION, mpc_model=Go2MpcModel.SRBD)
    assert cfg.mpc_model == "srbd"
    assert cfg.n == 13
    assert cfg.m == 12
    assert cfg.initial_state.shape == (13,)
    assert cfg.u_ref.shape == (12,)
    assert cfg.W.shape == (24, 24)
    assert cfg.Kp.shape == (12, 12)
    assert cfg.mass > 10.0


def test_go2_srbd_rejected_for_balance():
    with pytest.raises(ValueError, match="not balance"):
        go2_config(Go2Mode.BALANCE, mpc_model=Go2MpcModel.SRBD)
