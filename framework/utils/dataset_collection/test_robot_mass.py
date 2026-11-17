"""body_weight_n is the robot, not terrain welded to the world."""

from pathlib import Path

import mujoco

from framework.utils.dataset_collection.episode_recorder import read_episode_conditions
from framework.utils.simulation_utils.pyramid_terrain import stamp_scene

_SCENE = Path(__file__).resolve().parents[3] / "framework" / "data" / "go2_dls"


def test_ramp_weight_matches_the_flat_robot():
    flat = read_episode_conditions(
        mujoco.MjModel.from_xml_path(str(_SCENE / "scene_flat.xml")), []
    )
    ramp = read_episode_conditions(
        mujoco.MjModel.from_xml_path(str(_SCENE / "scene_ramp.xml")), []
    )
    assert abs(flat["body_weight_n"] - ramp["body_weight_n"]) < 1e-6
    assert 140.0 < ramp["body_weight_n"] < 160.0


def test_staired_pyramid_weight_matches_the_flat_robot():
    flat = read_episode_conditions(
        mujoco.MjModel.from_xml_path(str(_SCENE / "scene_flat.xml")), []
    )
    stairs = mujoco.MjModel.from_xml_path(str(_SCENE / "scene_staired_pyramid.xml"))
    stamp_scene("staired_pyramid", stairs)
    got = read_episode_conditions(stairs, [])
    assert abs(flat["body_weight_n"] - got["body_weight_n"]) < 1e-6
    assert 140.0 < got["body_weight_n"] < 160.0


def test_aliengo_trunk_is_the_robot():
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[3] / "framework" / "data" / "aliengo" / "scene_flat.xml")
    )
    weight = read_episode_conditions(model, [])["body_weight_n"]
    assert weight > 50.0


def test_spot_root_name_excludes_a_welded_mass():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="body">
              <inertial pos="0 0 0" mass="10" diaginertia="0.1 0.1 0.1"/>
            </body>
            <body name="ramp">
              <inertial pos="0 0 0" mass="1000" diaginertia="1 1 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    weight = read_episode_conditions(model, [])["body_weight_n"]
    assert abs(weight - 10.0 * 9.81) < 1e-6
