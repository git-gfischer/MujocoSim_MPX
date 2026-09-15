"""Unit tests for randomized truncated-pyramid terrains."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mpx.config.sim_config.config_reset_randomization import FloatRangeSpec
from mpx.config.sim_config.config_pyramid_terrain import pyramid_terrain_config
from mpx.utils.simulation_utils.pyramid_terrain import (
    PyramidSample,
    PyramidTerrain,
    apply_pyramid_boxes,
    sample_pyramid,
    sample_staired_pyramid,
    spawn_p0_and_xy_yaw,
    _layer_geom_name,
)

_GO2 = Path(__file__).resolve().parents[2] / "data" / "go2"


def test_smooth_sample_has_valid_top():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = sample_pyramid(pyramid_terrain_config, rng)
        assert s.kind == "pyramid"
        assert s.top_half >= 0.4
        assert s.top_half < s.base_half
        assert s.slope_deg is not None
        assert 8.0 <= s.slope_deg <= 15.0
        assert s.n_steps >= 2
        assert s.rise == pytest.approx(pyramid_terrain_config.smooth_rise)


def test_staired_sample_has_valid_top_and_integer_steps():
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = sample_staired_pyramid(pyramid_terrain_config, rng)
        assert s.kind == "staired_pyramid"
        assert s.n_steps >= 1
        assert s.n_steps <= pyramid_terrain_config.max_layers
        assert abs(s.height - s.n_steps * s.rise) < 1e-9
        assert s.top_half >= 0.4
        assert s.top_half < s.base_half
        grade_deg = float(np.rad2deg(np.arctan(s.rise / s.tread)))
        assert grade_deg <= 15.0


def test_smooth_clamp_when_ranges_impossible():
    cfg = replace(
        pyramid_terrain_config,
        sample_retries=1,
        base_half=FloatRangeSpec(True, 0.5, 0.5),
        height=FloatRangeSpec(True, 0.8, 0.8),
        slope_deg=FloatRangeSpec(True, 80.0, 80.0),
    )
    s = sample_pyramid(cfg, np.random.default_rng(0))
    assert s.top_half >= 0.4
    assert s.top_half < s.base_half


def test_smooth_clamp_preserves_requested_slope():
    cfg = replace(
        pyramid_terrain_config,
        sample_retries=1,
        base_half=FloatRangeSpec(True, 1.5, 1.5),
        height=FloatRangeSpec(True, 0.8, 0.8),
        slope_deg=FloatRangeSpec(True, 20.0, 20.0),
    )
    s = sample_pyramid(cfg, np.random.default_rng(0))
    assert s.slope_deg == pytest.approx(20.0, abs=0.6)
    assert s.top_half >= 0.4
    assert s.top_half < s.base_half


def _assert_layer_xml(path: Path):
    model = mujoco.MjModel.from_xml_path(str(path))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(24):
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, _layer_geom_name(i)
        )
        assert geom_id >= 0
        assert int(model.geom_bodyid[geom_id]) > 0
        assert int(model.geom_matid[geom_id]) != int(model.geom_matid[floor_id])
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "pyramid_hfield") < 0


def test_pyramid_xml_has_layer_boxes():
    _assert_layer_xml(_GO2 / "scene_pyramid.xml")


def test_staired_xml_has_layer_boxes():
    _assert_layer_xml(_GO2 / "scene_staired_pyramid.xml")


def _load_ids(model):
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _layer_geom_name(i))
        for i in range(24)
    ]


def test_pyramid_boxes_are_solid_plates():
    model = mujoco.MjModel.from_xml_path(str(_GO2 / "scene_pyramid.xml"))
    layer_ids = _load_ids(model)
    sample = PyramidSample(
        kind="pyramid",
        base_half=2.0,
        top_half=0.74,
        height=0.4,
        rise=0.05,
        tread=0.18,
        n_steps=8,
    )
    apply_pyramid_boxes(
        model,
        sample,
        layer_ids,
        model.geom_contype.copy(),
        model.geom_conaffinity.copy(),
    )
    gid0 = layer_ids[0]
    body0 = int(model.geom_bodyid[gid0])
    assert model.body_pos[body0, 0] == pytest.approx(0.0, abs=1e-6)
    assert model.body_pos[body0, 2] == pytest.approx(sample.rise / 2.0, abs=1e-6)
    assert model.geom_size[gid0, 0] == pytest.approx(sample.base_half, abs=1e-6)
    assert model.geom_size[gid0, 1] == pytest.approx(sample.base_half, abs=1e-6)
    unused = layer_ids[8]
    assert int(model.geom_contype[unused]) == 0
    top = layer_ids[7]
    top_body = int(model.geom_bodyid[top])
    top_height = sample.n_steps * sample.rise
    assert model.body_pos[top_body, 2] == pytest.approx(top_height / 2.0, abs=1e-6)
    assert model.geom_size[top, 2] == pytest.approx(top_height / 2.0, abs=1e-6)
    assert model.geom_size[top, 0] == pytest.approx(sample.top_half, abs=1e-6)


def test_resized_pyramid_produces_physics_contacts():
    """Contacts must follow the visible plates, including after a resize."""
    model = mujoco.MjModel.from_xml_path(str(_GO2 / "scene_pyramid.xml"))
    data = mujoco.MjData(model)
    terrain = PyramidTerrain.from_scene(
        "pyramid", model, rng=np.random.default_rng(1)
    )
    sample = terrain.apply(model)
    data.qpos[:] = model.qpos0
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = float(sample.height) + float(model.qpos0[2]) - 0.03
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    pyr_hits = 0
    for i in range(int(data.ncon)):
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2) or ""
        if n1.startswith("pyr_") or n2.startswith("pyr_"):
            pyr_hits += 1
    assert pyr_hits > 0, (
        f"robot fell through the visual pyramid (z={data.qpos[2]:.3f}, ncon={data.ncon})"
    )
    for _ in range(200):
        mujoco.mj_step(model, data)
    min_stand_z = float(sample.height) + 0.25
    assert float(data.qpos[2]) > min_stand_z, (
        f"robot dropped through the pyramid (z={data.qpos[2]:.3f}, height={sample.height:.3f})"
    )
    data.qpos[:2] = 10.0
    mujoco.mj_forward(model, data)
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(
        model,
        data,
        np.array([0.0, 0.0, 2.0], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
        None,
        1,
        -1,
        geomid,
    )
    hit = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])) or ""
    assert hit.startswith("pyr_"), f"ray missed pyramid geoms (hit={hit!r})"
    assert 2.0 - dist == pytest.approx(sample.height, abs=0.03)


def test_floor_spawn_is_outside_base():
    sample = PyramidSample(
        kind="pyramid",
        base_half=2.0,
        top_half=0.6,
        height=0.5,
        slope_deg=20.0,
        spawn_on_top=False,
    )
    rng = np.random.default_rng(0)
    for _ in range(40):
        p0, (x, y, yaw) = spawn_p0_and_xy_yaw(sample, robot_height=0.27, rng=rng)
        assert max(abs(x), abs(y)) > sample.base_half + 0.6
        assert p0[2] == pytest.approx(0.27)


def test_top_spawn_is_on_platform():
    sample = PyramidSample(
        kind="pyramid",
        base_half=2.0,
        top_half=0.6,
        height=0.5,
        slope_deg=20.0,
        spawn_on_top=True,
    )
    rng = np.random.default_rng(1)
    for _ in range(40):
        p0, (x, y, yaw) = spawn_p0_and_xy_yaw(sample, robot_height=0.27, rng=rng)
        assert abs(x) <= sample.top_half
        assert abs(y) <= sample.top_half
        assert p0[2] == pytest.approx(sample.height + 0.27)


def test_from_scene_flat_is_noop():
    model = mujoco.MjModel.from_xml_path(str(_GO2 / "scene_pyramid.xml"))
    terrain = PyramidTerrain.from_scene("flat", model)
    assert terrain.apply(model) is None


def test_from_scene_pyramid_applies_boxes():
    model = mujoco.MjModel.from_xml_path(str(_GO2 / "scene_pyramid.xml"))
    terrain = PyramidTerrain.from_scene(
        "pyramid", model, rng=np.random.default_rng(2)
    )
    sample = terrain.apply(model)
    assert sample is not None
    assert sample.kind == "pyramid"
    layer0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pyr_00")
    body0 = int(model.geom_bodyid[layer0])
    assert model.body_pos[body0, 0] == pytest.approx(0.0, abs=1e-6)
    assert model.geom_size[layer0, 0] == pytest.approx(sample.base_half, abs=1e-6)


def test_from_scene_staired_applies_boxes():
    model = mujoco.MjModel.from_xml_path(str(_GO2 / "scene_staired_pyramid.xml"))
    terrain = PyramidTerrain.from_scene(
        "staired_pyramid", model, rng=np.random.default_rng(3)
    )
    sample = terrain.apply(model)
    assert sample is not None
    assert sample.kind == "staired_pyramid"
    layer0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pyr_00")
    body0 = int(model.geom_bodyid[layer0])
    assert model.body_pos[body0, 0] == pytest.approx(0.0, abs=1e-6)
    assert model.geom_size[layer0, 0] == pytest.approx(sample.base_half, abs=1e-6)


def test_scene_to_terrain_pyramid_labels():
    from mpx.utils.dataset_collection.dataset_bucket_system import TerrainType
    from mpx.utils.dataset_collection.episode_recorder import scene_to_terrain

    assert scene_to_terrain("pyramid") is TerrainType.ROUGH
    assert scene_to_terrain("staired_pyramid") is TerrainType.STAIRS
