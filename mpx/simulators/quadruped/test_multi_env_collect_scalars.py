"""Collect reads gait scalars off batched MPCData.

mjx PyTreeNode only treats jax.Array-typed fields as pytree data, so
``duty_factor: float`` is metadata: vmap does not give it a leading env axis.
Indexing it with ``[i]`` is the collect crash.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mpx.simulators.quadruped.quad_locomotion_multiEnv import (
    _collect_command_source,
    _env_scalar,
    _resolve_headless_steps,
    _tick_navigators,
)
from mpx.navigation.pointNav import PointNavigator
from mpx.utils.mpc_wrapper import MPCData


def _dummy_mpc_data() -> MPCData:
    return MPCData(
        dt=0.02,
        duty_factor=0.65,
        step_freq=1.35,
        step_height=0.08,
        contact_time=jnp.array([0.5, 0.0, 0.0, 0.5]),
        liftoff=jnp.zeros(12),
        X0=jnp.zeros((2, 3)),
        U0=jnp.zeros((1, 3)),
        V0=jnp.zeros((1, 3)),
        W=jnp.eye(3),
    )


def test_vmapped_duty_factor_has_no_env_axis():
    batched = jax.vmap(lambda _: _dummy_mpc_data())(jnp.arange(4))
    assert np.asarray(batched.duty_factor).ndim == 0
    assert batched.contact_time.shape[0] == 4


def test_env_scalar_reads_shared_duty_factor_and_batched_arrays():
    batched = jax.vmap(lambda _: _dummy_mpc_data())(jnp.arange(4))
    for i in range(4):
        assert _env_scalar(batched.duty_factor, i) == 0.65
    assert _env_scalar(np.array([0.1, 0.2, 0.3, 0.4]), 2) == 0.3


def test_nav_random_drives_to_goals_during_collect():
    use_nav, use_sampler = _collect_command_source(
        nav="random", segmented_commands=True
    )
    assert use_nav is True
    assert use_sampler is False


def test_nav_vel_collect_uses_velocity_segments():
    use_nav, use_sampler = _collect_command_source(
        nav="vel", segmented_commands=True
    )
    assert use_nav is False
    assert use_sampler is True


def test_collect_headless_steps_default_covers_one_episode():
    assert _resolve_headless_steps(
        collect=True, steps=None, episode_duration_s=60.0, sim_hz=500.0
    ) == 30_000
    assert _resolve_headless_steps(
        collect=False, steps=None, episode_duration_s=60.0, sim_hz=500.0
    ) == 2000
    assert _resolve_headless_steps(
        collect=True, steps=50_000, episode_duration_s=60.0, sim_hz=500.0
    ) == 50_000


def test_tick_navigators_resamples_when_the_goal_is_reached():
    nav = PointNavigator(
        robot_height=0.3,
        auto_resample=True,
        goal_distance=(2.0, 2.0),
        seed=0,
    )
    qpos = np.zeros(7)
    qpos[3] = 1.0
    nav.reset(qpos)
    first = np.array(nav.goal_xy, copy=True)
    qpos[:2] = nav.goal_xy
    _tick_navigators([nav], qpos[None])
    assert not np.allclose(nav.goal_xy, first)
