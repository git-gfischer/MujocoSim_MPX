"""SlipInjector writes pair friction onto the foot and the terrain geom."""
import mujoco
import numpy as np

from framework.utils.simulation_utils.slip_injection import (
    SlipInjectionConfig,
    SlipInjector,
    draw_mu,
    parse_mu_bands,
)

_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="2 0.005 0.0001"/>
    <body name="base">
      <geom name="FL" type="sphere" size="0.02" priority="1" friction="1.2 0.02 0.01"/>
      <geom name="FR" type="sphere" size="0.02" priority="1" friction="1.2 0.02 0.01"/>
    </body>
    <body name="ramp">
      <geom name="ramp_00" type="box" size="0.5 0.5 0.1" friction="1.8 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model():
    model = mujoco.MjModel.from_xml_string(_XML)
    foot = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("FL", "FR")
    ]
    return model, foot


def test_an_event_sets_pair_friction_and_slip_event():
    model, foot = _model()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ramp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ramp_00")
    cfg = SlipInjectionConfig(mu_bands=((0.2, 0.2),), on_s=(10.0, 10.0), off_s=(10.0, 10.0))
    injector = SlipInjector(model, foot, sim_hz=500.0, rng=np.random.default_rng(0), cfg=cfg)
    # left starts at 0, so the first tick opens an event on every foot.
    event = injector.tick(model)
    assert event.tolist() == [1, 1]
    np.testing.assert_allclose(model.geom_friction[foot, 0], 0.2)
    assert model.geom_friction[floor, 0] == 0.2
    assert model.geom_friction[ramp, 0] == 0.2
    assert injector.terrain_ids.tolist() == [floor, ramp]


def test_capture_nominal_restores_the_episode_foot_and_the_xml_terrain():
    model, foot = _model()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ramp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ramp_00")
    cfg = SlipInjectionConfig(mu_bands=((0.2, 0.2),), on_s=(10.0, 10.0), off_s=(10.0, 10.0))
    injector = SlipInjector(model, foot, sim_hz=500.0, rng=np.random.default_rng(0), cfg=cfg)
    injector.tick(model)
    model.geom_friction[foot, 0] = 1.5
    injector.capture_nominal(model)
    np.testing.assert_allclose(model.geom_friction[foot, 0], 1.5)
    assert model.geom_friction[floor, 0] == 2.0
    assert model.geom_friction[ramp, 0] == 1.8
    assert injector.tick(model).tolist() == [0, 0]
    np.testing.assert_allclose(model.geom_friction[foot, 0], 1.5)


def test_mu_bands_hold_out_the_gap_and_cover_both_sides():
    bands = parse_mu_bands("0.15-0.2,0.3-0.55")
    assert bands == ((0.15, 0.2), (0.3, 0.55))
    mu = np.array([draw_mu(np.random.default_rng(i), bands) for i in range(4000)])
    assert not ((mu >= 0.2) & (mu < 0.3)).any()
    assert ((mu >= 0.15) & (mu <= 0.55)).all()
    # Uniform over the union: the 0.05-wide band gets 0.05 / 0.30 of the draws.
    assert abs((mu < 0.2).mean() - 0.05 / 0.30) < 0.03
    # The applied μ is min(foot, terrain), still inside the union.
    model, foot = _model()
    cfg = SlipInjectionConfig(mu_bands=bands, on_s=(10.0, 10.0), off_s=(10.0, 10.0))
    for seed in range(200):
        injector = SlipInjector(model, foot, sim_hz=500.0, rng=np.random.default_rng(seed), cfg=cfg)
        injector.tick(model)
        applied = model.geom_friction[foot, 0]
        assert not ((applied >= 0.2) & (applied < 0.3)).any()


def test_bad_bands_raise():
    for bad in ((), ((0.3, 0.2),), ((0.0, 0.2),)):
        try:
            SlipInjectionConfig(mu_bands=bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


if __name__ == "__main__":
    test_an_event_sets_pair_friction_and_slip_event()
    test_capture_nominal_restores_the_episode_foot_and_the_xml_terrain()
    test_mu_bands_hold_out_the_gap_and_cover_both_sides()
    test_bad_bands_raise()
    print("ok")
