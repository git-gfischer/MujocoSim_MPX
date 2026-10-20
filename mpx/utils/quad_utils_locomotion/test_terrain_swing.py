"""Swing targets from a sampled terrain height grid."""

from __future__ import annotations

from functools import partial

import mujoco
import numpy as np
import pytest
import jax.numpy as jnp

from mpx.utils.quad_utils_locomotion.reference_generator_locomotion import (
    base_height_reference,
    call_ref,
    dynamic_terrain_kwargs,
    reference_generator_locomotion,
    sample_terrain_height,
    swing_landing_and_apex,
    terrain_reference_kwargs,
)
from mpx.utils.simulation_utils.measured_kinematics import (
    TerrainProbe,
    apply_rough_terrain_mpc,
    build_terrain_height_grid,
    terrain_overlay_grid,
)


_BOX = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.05"/>
    <geom name="step" type="box" pos="1 0 0.1" size="0.2 0.2 0.1"/>
  </worldbody>
</mujoco>
"""


def test_height_grid_matches_ray_on_a_box():
    model = mujoco.MjModel.from_xml_string(_BOX)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grid, origin, resolution = build_terrain_height_grid(model, resolution=0.05)
    probe = TerrainProbe(model)

    def at(x, y):
        ix = int(np.rint((x - origin[0]) / resolution))
        iy = int(np.rint((y - origin[1]) / resolution))
        return float(grid[ix, iy])

    assert at(1.0, 0.0) == pytest.approx(probe.height_at(data, (1.0, 0.0)))
    assert at(1.0, 0.0) == pytest.approx(0.2)
    assert at(0.75, 0.0) == pytest.approx(0.0)


def test_rough_flag_turns_the_height_grid_on_and_off():
    model = mujoco.MjModel.from_xml_string(_BOX)

    class Config:
        pass

    blind = Config()
    apply_rough_terrain_mpc(blind, model, scene="rough", terrain_aware=False)
    assert not hasattr(blind, "terrain_height_grid")

    other = Config()
    apply_rough_terrain_mpc(other, model, scene="flat", terrain_aware=True)
    assert not hasattr(other, "terrain_height_grid")

    aware = Config()
    apply_rough_terrain_mpc(aware, model, scene="rough", terrain_aware=True)
    assert aware.terrain_height_grid.shape[0] > 0


def test_ramp_and_staired_pyramid_use_the_same_height_grid():
    model = mujoco.MjModel.from_xml_string(_BOX)
    for scene in ("ramp", "staired_pyramid"):
        aware = type("Config", (), {})()
        apply_rough_terrain_mpc(aware, model, scene=scene, terrain_aware=True)
        assert terrain_reference_kwargs(aware)["use_terrain_height"] is True
        overlay = terrain_overlay_grid(aware)
        assert overlay is not None
        assert overlay[0].shape == aware.terrain_height_grid.shape

        blind = type("Config", (), {})()
        apply_rough_terrain_mpc(blind, model, scene=scene, terrain_aware=False)
        assert terrain_reference_kwargs(blind) == {}
        assert terrain_overlay_grid(blind) is None


def test_ramp_and_applied_staired_pyramid_are_not_flat_slabs():
    from pathlib import Path

    from mpx.utils.simulation_utils.pyramid_terrain import PyramidTerrain

    from mpx.utils.simulation_utils.ramp_pyramid import RampPyramid

    root = Path(__file__).resolve().parents[2] / "data" / "go2"
    ramp = mujoco.MjModel.from_xml_path(str(root / "scene_ramp.xml"))
    ramp_sample = RampPyramid.from_scene(
        "ramp", ramp, rng=np.random.default_rng(1)
    ).apply(ramp)
    ramp_grid, _, _ = build_terrain_height_grid(ramp, resolution=0.2)
    assert float(ramp_grid.max() - ramp_grid.min()) > 0.2
    assert float(ramp_grid.max()) == pytest.approx(ramp_sample.height, abs=0.08)

    stair = mujoco.MjModel.from_xml_path(str(root / "scene_staired_pyramid.xml"))
    sample = PyramidTerrain.from_scene(
        "staired_pyramid", stair, rng=np.random.default_rng(0)
    ).apply(stair)
    grid, origin, resolution = build_terrain_height_grid(stair, resolution=0.1)

    def at(x, y):
        ix = int(np.rint((x - origin[0]) / resolution))
        iy = int(np.rint((y - origin[1]) / resolution))
        return float(grid[ix, iy])

    assert at(0.0, 0.0) == pytest.approx(sample.height, abs=0.03)
    assert at(sample.base_half - 0.05, 0.0) == pytest.approx(sample.rise, abs=0.03)
    assert at(sample.base_half + 1.0, 0.0) == pytest.approx(0.0, abs=0.02)


def test_a_jitted_solve_reads_the_map_it_is_passed():
    # quad_locomotion jits the whole MPC step. A map stored on the wrapper is
    # frozen into the first trace, so a respawned pyramid kept the old steps.
    import jax

    class Wrapper:
        _ref_gen = staticmethod(
            lambda **kw: kw["terrain_grid"].sum() + kw["terrain_origin"].sum()
        )

    solve = jax.jit(lambda terrain: call_ref(Wrapper(), terrain=terrain))
    origin = jnp.zeros(2, dtype=jnp.float32)
    assert float(solve((jnp.zeros((4, 4), jnp.float32), origin))) == 0.0
    assert float(solve((jnp.ones((4, 4), jnp.float32), origin))) == 16.0


def test_a_fixed_extent_keeps_one_grid_shape_across_pyramids():
    # A new shape retraces and recompiles the whole MPC on every respawn.
    from pathlib import Path

    from mpx.utils.simulation_utils.pyramid_terrain import PyramidTerrain

    root = Path(__file__).resolve().parents[2] / "data" / "go2_dls"
    model = mujoco.MjModel.from_xml_path(str(root / "scene_staired_pyramid.xml"))
    terrain = PyramidTerrain.from_scene(
        "staired_pyramid", model, rng=np.random.default_rng(3)
    )
    extent = terrain.cfg.base_half.high + 0.5
    shapes = set()
    for _ in range(3):
        sample = terrain.apply(model)
        grid, origin, res = build_terrain_height_grid(model, 0.05, extent=extent)
        shapes.add(grid.shape)
        top = sample_terrain_height(0.0, 0.0, grid, origin, res)
        edge = sample_terrain_height(sample.base_half - 0.05, 0.0, grid, origin, res)
        assert float(top) == pytest.approx(sample.height, abs=0.01)
        assert float(edge) == pytest.approx(sample.rise, abs=0.01)
    assert len(shapes) == 1


def test_stairs_height_grid_lands_the_swing():
    model = mujoco.MjModel.from_xml_string(_BOX)
    aware = type("Config", (), {})()
    apply_rough_terrain_mpc(aware, model, scene="stairs", terrain_aware=True)
    assert terrain_reference_kwargs(aware)["use_terrain_height"] is True
    overlay = terrain_overlay_grid(aware)
    assert overlay is not None
    assert overlay[0].shape == aware.terrain_height_grid.shape

    blind = type("Config", (), {})()
    apply_rough_terrain_mpc(blind, model, scene="stairs", terrain_aware=False)
    assert terrain_reference_kwargs(blind) == {}


def test_landing_height_follows_the_grid_and_flat_ground_does_not_change():
    resolution = 0.1
    grid = jnp.zeros((10, 4), dtype=jnp.float32)
    grid = grid.at[5, 0].set(0.12)
    origin = jnp.array([0.0, 0.0], dtype=jnp.float32)
    liftoff_z = jnp.array([0.02, 0.02, 0.02, 0.02])
    liftoff_x = jnp.zeros(4)
    liftoff_y = jnp.zeros(4)
    foothold_x = jnp.array([0.5, 0.0, 0.0, 0.0])
    foothold_y = jnp.zeros(4)

    landing, apex = swing_landing_and_apex(
        liftoff_x, liftoff_y, liftoff_z,
        foothold_x, foothold_y,
        0.08,
        grid, origin, resolution,
    )
    assert float(landing[0]) == pytest.approx(0.14)
    assert float(apex[0]) == pytest.approx(0.22)
    assert float(landing[1]) == pytest.approx(0.02)
    assert float(apex[1]) == pytest.approx(0.10)


def test_a_foothold_on_the_lip_aims_at_the_tread():
    grid = jnp.zeros((10, 4), dtype=jnp.float32)
    grid = grid.at[5, 0].set(0.05)
    origin = jnp.zeros(2, dtype=jnp.float32)
    # x=0.45 rounds to the floor cell beside the 5 cm tread.
    landing, apex = swing_landing_and_apex(
        jnp.array([0.2]), jnp.zeros(1), jnp.array([0.02]),
        jnp.array([0.45]), jnp.zeros(1),
        0.065,
        grid, origin, 0.1,
    )
    assert float(landing[0]) == pytest.approx(0.07)
    assert float(apex[0]) >= 0.05 + 0.065


def test_approaching_a_tall_cell_does_not_swing_over_it():
    grid = jnp.zeros((20, 4), dtype=jnp.float32)
    grid = grid.at[10, 0].set(0.40)
    origin = jnp.zeros(2, dtype=jnp.float32)
    landing, apex = swing_landing_and_apex(
        jnp.array([0.5]), jnp.zeros(1), jnp.array([0.02]),
        jnp.array([0.9]), jnp.zeros(1),
        0.065,
        grid, origin, 0.1,
    )
    assert float(landing[0]) == pytest.approx(0.07)
    assert float(apex[0]) < 0.20


def test_a_rebuilt_height_map_is_read_from_its_own_origin():
    class Config:
        terrain_height_grid = jnp.zeros((4, 4), dtype=jnp.float32)
        terrain_height_origin = jnp.array([-5.0, -5.0], dtype=jnp.float32)
        terrain_height_resolution = 0.5
        terrain_grid_dynamic = True

    kw, _, origin = dynamic_terrain_kwargs(Config())
    assert "terrain_grid" not in kw
    assert "terrain_origin" not in kw
    new = jnp.full((3, 3), 0.6, dtype=jnp.float32)
    new_origin = jnp.array([-0.5, -0.5], dtype=jnp.float32)
    missed = sample_terrain_height(0.0, 0.0, new, origin, 0.5)
    hit = sample_terrain_height(0.0, 0.0, new, new_origin, 0.5)
    assert float(missed) == 0.0
    assert float(hit) == pytest.approx(0.6)


def test_a_foot_above_the_plateau_does_not_lift_the_base():
    grid = jnp.full((8, 8), 0.55, dtype=jnp.float32)
    origin = jnp.zeros(2, dtype=jnp.float32)
    liftoff_z = jnp.full(4, 0.63)
    z = base_height_reference(
        True, 0.0, 0.27,
        jnp.zeros(4), jnp.zeros(4), liftoff_z,
        jnp.zeros(5), jnp.zeros(5),
        grid, origin, 0.05,
    )
    assert float(z[0]) == pytest.approx(0.55 + 0.27 + 0.02)
    landing, _ = swing_landing_and_apex(
        jnp.zeros(1), jnp.zeros(1), jnp.array([0.63]),
        jnp.zeros(1), jnp.zeros(1),
        0.065, grid, origin, 0.05,
    )
    assert float(landing[0]) == pytest.approx(0.55 + 0.02)


def test_a_foot_on_the_floor_lands_on_a_stair_several_steps_up():
    # Logged climbing the staired pyramid: a rear foot on the floor with its
    # foothold three stairs up. Capped at one step height, it aimed into the
    # riser, never got onto the stairs, and the robot tipped back.
    # Floor up to x = 0.5 m, then 5 cm rises on 10 cm treads.
    x = jnp.arange(40) * 0.05
    steps = jnp.where(x >= 0.5, 0.05 * (jnp.floor((x - 0.5) / 0.1 + 1e-6) + 1), 0.0)
    grid = jnp.tile(steps[:, None], (1, 4)).astype(jnp.float32)
    origin = jnp.zeros(2, dtype=jnp.float32)
    landing, apex = swing_landing_and_apex(
        jnp.array([0.45]), jnp.zeros(1), jnp.array([0.018]),
        jnp.array([0.75]), jnp.zeros(1),
        0.065,
        grid, origin, 0.05,
    )
    tread = float(sample_terrain_height(0.75, 0.0, grid, origin, 0.05))
    assert tread == pytest.approx(0.15)
    assert float(landing[0]) >= tread
    assert float(apex[0]) >= tread + 0.065


def test_walking_down_the_base_follows_the_ground_under_it_not_the_rear_feet():
    # Feet logged on the staired pyramid, base over the 0.55 stair. A clamp to
    # the feet put the base target at 0.96 instead of 0.84.
    grid = jnp.full((8, 8), 0.55, dtype=jnp.float32)
    origin = jnp.zeros(2, dtype=jnp.float32)
    liftoff_z = jnp.array([0.69, 0.47, 0.72, 0.71])
    z = base_height_reference(
        True, 0.0, 0.27,
        jnp.zeros(4), jnp.zeros(4), liftoff_z,
        jnp.zeros(5), jnp.zeros(5),
        grid, origin, 0.05,
    )
    assert float(z[0]) == pytest.approx(0.55 + 0.27 + 0.015)


def test_lip_raises_the_apex_above_a_mid_swing_step():
    grid = jnp.zeros((10, 4), dtype=jnp.float32)
    grid = grid.at[2, 0].set(0.05)
    origin = jnp.array([0.0, 0.0], dtype=jnp.float32)
    landing, apex = swing_landing_and_apex(
        jnp.zeros(1), jnp.zeros(1), jnp.array([0.02]),
        jnp.array([0.6]), jnp.zeros(1),
        0.08,
        grid, origin, 0.1,
    )
    assert float(landing[0]) == pytest.approx(0.02)
    assert float(apex[0]) == pytest.approx(0.15)


def test_reference_swing_ends_on_the_grid_height():
    n_joints = 12
    n_contact = 4
    resolution = 0.1
    grid = jnp.zeros((10, 4), dtype=jnp.float32).at[5, 0].set(0.05)
    origin = jnp.array([0.0, 0.0], dtype=jnp.float32)
    foot0 = jnp.zeros(3 * n_contact).at[0].set(0.5)
    liftoff = jnp.zeros(3 * n_contact).at[2].set(0.02)
    x = jnp.zeros(7 + 2 * n_joints + 6).at[3].set(1.0).at[2].set(0.3)
    command = jnp.zeros(7).at[6].set(0.3)
    common = dict(
        foot0=foot0,
        q0=jnp.zeros(n_joints),
        t_timer=jnp.array([0.5, 0.0, 0.0, 0.0]),
        x=x,
        foot=liftoff,
        input=command,
        duty_factor=0.5,
        step_freq=1.0,
        step_height=0.08,
        liftoff=liftoff,
        contact=jnp.zeros(n_contact),
        clearence_speed=0.2,
    )
    flat_ref = partial(
        reference_generator_locomotion, False, 30, 0.02, n_joints, n_contact, 12.0,
    )
    flat, _, _ = flat_ref(**common)
    raised, raised_contact, _ = flat_ref(
        **common,
        use_terrain_height=True,
        terrain_grid=grid,
        terrain_origin=origin,
        terrain_resolution=resolution,
    )
    foot_z = 3 + 4 + n_joints + 3 + 3 + 2
    swing = np.asarray(raised_contact[:, 0]) < 0.5
    assert swing.any()
    idx = int(np.flatnonzero(swing)[-1])
    assert float(flat[idx, foot_z]) < 0.06
    assert float(raised[idx, foot_z]) > 0.06
    # Standing on the flat cell: base height stays clearance plus foot height.
    assert float(raised[0, 2]) == pytest.approx(float(flat[0, 2]))


def test_base_height_reference_follows_the_grid_ahead():
    n_joints = 12
    n_contact = 4
    grid = jnp.zeros((20, 4), dtype=jnp.float32).at[5, 0].set(0.20)
    origin = jnp.array([0.0, 0.0], dtype=jnp.float32)
    liftoff = jnp.zeros(3 * n_contact).at[2].set(0.02)
    x = jnp.zeros(7 + 2 * n_joints + 6).at[3].set(1.0).at[2].set(0.3)
    command = jnp.zeros(7).at[0].set(1.0).at[6].set(0.30)
    ref, _, _ = reference_generator_locomotion(
        False, 30, 0.02, n_joints, n_contact, 12.0,
        foot0=jnp.zeros(3 * n_contact),
        q0=jnp.zeros(n_joints),
        t_timer=jnp.zeros(n_contact),
        x=x,
        foot=liftoff,
        input=command,
        duty_factor=0.6,
        step_freq=1.0,
        step_height=0.08,
        liftoff=liftoff,
        contact=jnp.ones(n_contact),
        clearence_speed=0.2,
        use_terrain_height=True,
        terrain_grid=grid,
        terrain_origin=origin,
        terrain_resolution=0.1,
    )
    # Horizon walks +x at 1 m/s and crosses the 0.20 m cell at x = 0.5 m.
    z = np.asarray(ref[:, 2])
    assert z[0] == pytest.approx(0.305)
    assert z.max() == pytest.approx(0.505, abs=0.02)
