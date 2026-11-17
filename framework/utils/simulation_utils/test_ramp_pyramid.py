"""Ramp pyramid: slope, height, facing spawn, and the far-side goal."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from framework.utils.simulation_utils.measured_kinematics import (
    TerrainProbe,
    build_terrain_height_grid,
)
from framework.utils.simulation_utils.ramp_pyramid import SLOPE_DEG, RampPyramid
from framework.utils.spawner.facing_center import opposite_side_xy, sample_facing_center
from framework.utils.spawner.spawner import RobotMapSpawner


def _ramp_model() -> mujoco.MjModel:
    root = Path(__file__).resolve().parents[2] / "data" / "go2" / "scene_ramp.xml"
    return mujoco.MjModel.from_xml_path(str(root))


def test_slope_draw_sets_height_and_changes_across_resets():
    model = _ramp_model()
    terrain = RampPyramid.from_scene("ramp", model, rng=np.random.default_rng(0))
    samples = [terrain.apply(model) for _ in range(12)]
    slopes = [s.slope_deg for s in samples]
    assert min(slopes) >= SLOPE_DEG[0]
    assert max(slopes) <= SLOPE_DEG[1]
    assert max(slopes) - min(slopes) > 1.0
    run = samples[0].base_half - samples[0].top_half
    for sample in samples:
        expect = run * np.tan(np.deg2rad(sample.slope_deg))
        assert sample.height == pytest.approx(expect)


def test_smooth_face_matches_the_slope_and_stops_the_robot():
    model = _ramp_model()
    data = mujoco.MjData(model)
    sample = RampPyramid.from_scene(
        "ramp", model, rng=np.random.default_rng(2)
    ).apply(model)
    mujoco.mj_forward(model, data)
    probe = TerrainProbe(model)
    assert probe.height_at(data, (0.0, 0.0)) == pytest.approx(sample.height, abs=0.03)
    assert probe.height_at(data, (0.0, sample.top_half * 0.5)) == pytest.approx(
        sample.height, abs=0.03
    )
    outside = sample.base_half + 0.4
    assert probe.height_at(data, (outside, 0.0)) == pytest.approx(0.0, abs=0.02)
    run = sample.base_half - sample.top_half
    xs = np.linspace(sample.top_half, sample.base_half, 9)
    zs = [probe.height_at(data, (float(x), 0.0)) for x in xs]
    for x, z in zip(xs, zs):
        expect = sample.height * (sample.base_half - float(x)) / run
        assert z == pytest.approx(expect, abs=0.04)
    assert all(zs[i] > zs[i + 1] + 1e-3 for i in range(len(zs) - 1))

    mid = 0.5 * (sample.base_half + sample.top_half)
    local = probe.height_at(data, (mid, 0.0))
    grid, origin, resolution = build_terrain_height_grid(model, resolution=0.05)

    def at(x, y):
        ix = int(np.rint((x - origin[0]) / resolution))
        iy = int(np.rint((y - origin[1]) / resolution))
        return float(grid[ix, iy])

    assert at(0.0, 0.0) == pytest.approx(sample.height, abs=0.04)
    assert at(mid, 0.0) == pytest.approx(local, abs=0.05)

    q0 = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
    data.qpos[:] = 0
    data.qpos[0] = mid
    data.qpos[2] = local + 0.6
    data.qpos[3] = 1.0
    data.qpos[7:19] = q0
    hit_ramp = False
    for _ in range(800):
        mujoco.mj_step(model, data)
        for k in range(data.ncon):
            for g in (int(data.contact[k].geom1), int(data.contact[k].geom2)):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                hit_ramp = hit_ramp or name.startswith("ramp_")
    # Collapsed onto the pyramid, not through it onto the floor.
    assert float(data.qpos[2]) > local
    assert hit_ramp


def test_steep_face_holds_a_planted_robot():
    model = _ramp_model()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ramp_00")
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    assert int(model.geom_priority[gid]) > int(model.geom_priority[foot])
    assert float(model.geom_friction[gid][0]) >= 1.5

    terrain = RampPyramid.from_scene("ramp", model, rng=np.random.default_rng(5))
    sample = terrain.apply(model)
    steep = SLOPE_DEG[1] - 0.25 * (SLOPE_DEG[1] - SLOPE_DEG[0])
    for _ in range(30):
        if sample.slope_deg >= steep:
            break
        sample = terrain.apply(model)
    assert sample.slope_deg >= steep
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mid = 0.5 * (sample.base_half + sample.top_half)
    local = TerrainProbe(model).height_at(data, (mid, 0.0))
    q0 = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
    data.qpos[:] = 0
    data.qpos[0] = mid
    data.qpos[2] = local + 0.3
    data.qpos[3] = 1.0
    data.qpos[7:19] = q0
    for _ in range(400):
        q = np.asarray(data.qpos[7:19])
        dq = np.asarray(data.qvel[6:18])
        data.ctrl = np.clip(80.0 * (q0 - q) - 2.0 * dq, -23.0, 23.0)
        mujoco.mj_step(model, data)
    assert abs(float(data.qpos[0]) - mid) < 0.2
    assert float(data.qpos[2]) > local


def test_spawn_faces_the_center_and_the_goal_is_opposite():
    model = _ramp_model()
    data = mujoco.MjData(model)
    terrain = RampPyramid.from_scene("ramp", model, rng=np.random.default_rng(3))
    p0, xy_yaw = sample_facing_center(
        np.random.default_rng(4), terrain.spawn_reach, 0.3
    )
    x, y, yaw = xy_yaw
    quat = RobotMapSpawner.orient_with_yaw(np.array([1.0, 0.0, 0.0, 0.0]), yaw)
    data.qpos[:3] = (x, y, 0.3)
    data.qpos[3:7] = quat
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    forward = np.array(data.xmat[bid]).reshape(3, 3)[:, 0]
    toward = -np.array([x, y, 0.0])
    toward /= np.linalg.norm(toward)
    assert np.dot(forward, toward) > 0.99

    goal = terrain.opposite_goal(np.array([x, y, 0.3]))
    assert np.dot(goal, np.array([x, y])) < 0.0
    # Spawn and goal sit on a square band just past the ramp's base edge, so
    # the out-of-bounds region can stay close to the ramp.
    assert float(np.max(np.abs(goal))) == pytest.approx(terrain.spawn_reach)
    assert np.allclose(goal, opposite_side_xy((x, y), terrain.spawn_reach))
    assert max(abs(x), abs(y)) == pytest.approx(terrain.spawn_reach)
    assert terrain.base_half < terrain.spawn_reach <= terrain.base_half + 1.0
    assert p0.shape == (3,)
