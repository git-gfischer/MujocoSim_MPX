"""Unit tests for the still4 pilars scene."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from framework.config.sim_config.config_pilar_terrain import pilar_terrain_config
from framework.config.sim_config.config_reset_randomization import FloatRangeSpec
from framework.utils.dataset_collection.dataset_bucket_system import TerrainType
from framework.utils.dataset_collection.episode_recorder import scene_to_terrain
from framework.utils.simulation_utils.pilar_terrain import (
    FOOT_XY,
    PilarTerrain,
    sample_pilars,
    spawn_p0_and_xy_yaw,
)

_GO2_DLS = Path(__file__).resolve().parents[2] / "data" / "go2_dls"


def _load(scene: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_GO2_DLS / f"scene_{scene}.xml"))


def test_sample_heights_stay_inside_config_range():
    rng = np.random.default_rng(0)
    low, high = pilar_terrain_config.height.low, pilar_terrain_config.height.high
    for _ in range(40):
        sample = sample_pilars(pilar_terrain_config, rng)
        assert len(sample.heights) == 4
        assert all(low <= h <= high for h in sample.heights)


def test_positive_height_places_box_on_nominal_foot():
    cfg = replace(
        pilar_terrain_config,
        height=FloatRangeSpec(True, 0.06, 0.06),
    )
    model = _load("pilars")
    sample = PilarTerrain.from_scene("pilars", model, cfg=cfg, rng=np.random.default_rng(1)).apply(model)
    assert sample is not None
    assert sample.mean_height == pytest.approx(0.06)
    for (x, y), name, height in zip(FOOT_XY, ("pilar_fl", "pilar_fr", "pilar_rl", "pilar_rr"), sample.heights):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        body_id = int(model.geom_bodyid[geom_id])
        assert model.body_pos[body_id, 0] == pytest.approx(x)
        assert model.body_pos[body_id, 1] == pytest.approx(y)
        assert model.body_pos[body_id, 2] == pytest.approx(height / 2.0)
        assert model.geom_size[geom_id, 2] == pytest.approx(height / 2.0)
        assert int(model.geom_contype[geom_id]) != 0


def test_zero_height_disables_contact():
    cfg = replace(
        pilar_terrain_config,
        height=FloatRangeSpec(True, 0.0, 0.0),
    )
    model = _load("pilars")
    PilarTerrain.from_scene("pilars", model, cfg=cfg, rng=np.random.default_rng(2)).apply(model)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pilar_fl")
    assert int(model.geom_contype[geom_id]) == 0


def test_spawn_is_origin_raised_by_mean_height():
    sample = sample_pilars(
        replace(pilar_terrain_config, height=FloatRangeSpec(True, 0.02, 0.08)),
        np.random.default_rng(3),
    )
    p0, xy_yaw = spawn_p0_and_xy_yaw(sample, robot_height=0.27)
    assert xy_yaw == (0.0, 0.0, 0.0)
    assert p0[0] == pytest.approx(0.0)
    assert p0[1] == pytest.approx(0.0)
    assert p0[2] == pytest.approx(0.27 + sample.mean_height)


def test_flat_scene_is_noop():
    model = _load("flat")
    assert PilarTerrain.from_scene("flat", model).apply(model) is None


def test_scene_to_terrain_pilars_is_rough():
    assert scene_to_terrain("pilars") is TerrainType.ROUGH
