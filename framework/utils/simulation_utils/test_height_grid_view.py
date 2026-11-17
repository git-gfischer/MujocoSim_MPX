"""Viewer tiles for the rough-terrain height grid."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np

from framework.utils.simulation_utils.sim_utils import (
    _HEIGHT_GRID_LIFT,
    _height_grid_stride,
    render_height_grid,
)


def test_stride_shrinks_until_the_window_fits():
    assert _height_grid_stride(41, 10_000) == 1
    assert _height_grid_stride(41, 100) ** 2 >= 1
    side = (41 + _height_grid_stride(41, 100) - 1) // _height_grid_stride(41, 100)
    assert side * side <= 100
    assert _height_grid_stride(5, 0) == 0


def _viewer(maxgeom: int = 2000):
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='plane' size='1 1 0.1'/></worldbody></mujoco>"
    )
    return SimpleNamespace(user_scn=mujoco.MjvScene(model, maxgeom=maxgeom))


def test_tiles_sit_on_the_sampled_cell_and_reuse_geoms():
    grid = np.array([[0.0, 0.1], [0.2, 0.4]], dtype=np.float32)
    viewer = _viewer()
    ids = render_height_grid(
        viewer,
        grid,
        np.array([0.0, 0.0]),
        1.0,
        np.array([0.0, 0.0]),
        half_extent=1.0,
    )
    ngeom = viewer.user_scn.ngeom
    # 3x3 window around the origin cell; index 4 is grid[0, 0].
    center = viewer.user_scn.geoms[ids[4]]
    np.testing.assert_allclose(center.pos, [0.0, 0.0, _HEIGHT_GRID_LIFT], atol=1e-5)
    high = viewer.user_scn.geoms[ids[8]]
    np.testing.assert_allclose(high.pos, [1.0, 1.0, 0.4 + _HEIGHT_GRID_LIFT], atol=1e-5)
    assert viewer.user_scn.geoms[ids[0]].rgba[3] == 0.0
    assert center.rgba[2] > high.rgba[2]

    ids_again = render_height_grid(
        viewer,
        grid,
        np.array([0.0, 0.0]),
        1.0,
        np.array([1.0, 1.0]),
        ids,
        half_extent=1.0,
    )
    assert ids_again is ids
    assert viewer.user_scn.ngeom == ngeom


def test_missing_grid_draws_nothing():
    viewer = _viewer()
    assert render_height_grid(viewer, None, None, 0.05, np.zeros(2)) == []
    assert viewer.user_scn.ngeom == 0
