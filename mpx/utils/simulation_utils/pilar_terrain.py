"""Four foot pillars for the still4 ``pilars`` scene.

Each reset draws a height per foot from :data:`pilar_terrain_config` and
rewrites the welded box for that foot. A height of 0 hides the box so the
foot stands on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mpx.config.sim_config.config_pilar_terrain import (
    PilarTerrainConfig,
    pilar_terrain_config,
)

# Nominal Go2 stance feet, FL FR RL RR, matching ``p_legs0`` xy.
FOOT_XY: tuple[tuple[float, float], ...] = (
    (0.192, 0.142),
    (0.192, -0.142),
    (-0.195, 0.142),
    (-0.195, -0.142),
)
GEOM_NAMES: tuple[str, ...] = ("pilar_fl", "pilar_fr", "pilar_rl", "pilar_rr")
_META_KEYS: tuple[str, ...] = ("pilar_h_fl", "pilar_h_fr", "pilar_h_rl", "pilar_h_rr")


@dataclass(frozen=True)
class PilarSample:
    """Four pillar heights in metres, FL FR RL RR."""

    heights: tuple[float, float, float, float]

    @property
    def mean_height(self) -> float:
        return float(sum(self.heights) / 4.0)

    def to_metadata(self) -> dict[str, float]:
        return {key: float(height) for key, height in zip(_META_KEYS, self.heights)}


def sample_pilars(cfg: PilarTerrainConfig, rng: np.random.Generator) -> PilarSample:
    low = float(cfg.height.low)
    high = float(cfg.height.high)
    drawn = [float(rng.uniform(low, high)) for _ in range(4)]
    return PilarSample(heights=(drawn[0], drawn[1], drawn[2], drawn[3]))


def _set_box(
    model: Any,
    geom_id: int,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    contype: int,
    conaffinity: int,
) -> None:
    body_id = int(model.geom_bodyid[geom_id])
    if body_id > 0:
        model.body_pos[body_id] = pos
        model.geom_pos[geom_id] = (0.0, 0.0, 0.0)
    else:
        model.geom_pos[geom_id] = pos
    model.geom_size[geom_id] = size
    model.geom_rbound[geom_id] = float(np.linalg.norm(size))
    model.geom_contype[geom_id] = contype
    model.geom_conaffinity[geom_id] = conaffinity
    if hasattr(model, "geom_aabb"):
        model.geom_aabb[geom_id, :3] = 0.0
        model.geom_aabb[geom_id, 3:] = size
    if hasattr(model, "geom_rgba"):
        model.geom_rgba[geom_id, 3] = 0.0 if contype == 0 else 1.0


def _hide_box(model: Any, geom_id: int, xy: tuple[float, float]) -> None:
    _set_box(
        model,
        geom_id,
        (xy[0], xy[1], -2.0),
        (1e-4, 1e-4, 1e-4),
        contype=0,
        conaffinity=0,
    )


def apply_pilar_boxes(
    model: Any,
    sample: PilarSample,
    geom_ids: list[int],
    compiled_contype: Any,
    compiled_conaffinity: Any,
    *,
    half_xy: float,
) -> None:
    for geom_id, xy, height in zip(geom_ids, FOOT_XY, sample.heights):
        gid = int(geom_id)
        if height <= 0.0:
            _hide_box(model, gid, xy)
            continue
        half_z = float(height) / 2.0
        on_type = int(compiled_contype[gid]) or 1
        on_aff = int(compiled_conaffinity[gid]) or 1
        _set_box(
            model,
            gid,
            (xy[0], xy[1], half_z),
            (float(half_xy), float(half_xy), half_z),
            contype=on_type,
            conaffinity=on_aff,
        )


def spawn_p0_and_xy_yaw(
    sample: PilarSample,
    robot_height: float,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    p0 = np.array([0.0, 0.0, float(robot_height) + sample.mean_height], dtype=np.float64)
    return p0, (0.0, 0.0, 0.0)


def foot_anchor(sample: PilarSample) -> np.ndarray:
    """World foot targets: nominal xy, z = that pillar's height."""
    out = np.zeros(12, dtype=np.float64)
    for i, ((x, y), height) in enumerate(zip(FOOT_XY, sample.heights)):
        out[3 * i : 3 * i + 3] = (x, y, height)
    return out


class PilarTerrain:
    """No-op unless ``scene`` is ``pilars``."""

    def __init__(
        self,
        active: bool,
        cfg: PilarTerrainConfig,
        rng: np.random.Generator,
        geom_ids: list[int] | None = None,
        compiled_contype: Any = None,
        compiled_conaffinity: Any = None,
    ):
        self.active = active
        self.cfg = cfg
        self._rng = rng
        self.geom_ids = list(geom_ids or [])
        self.compiled_contype = compiled_contype
        self.compiled_conaffinity = compiled_conaffinity

    @classmethod
    def from_scene(
        cls,
        scene: str,
        model: Any,
        cfg: PilarTerrainConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> PilarTerrain:
        cfg = cfg or pilar_terrain_config
        rng = rng if rng is not None else np.random.default_rng()
        if scene != "pilars":
            return cls(active=False, cfg=cfg, rng=rng)
        import mujoco

        geom_ids = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in GEOM_NAMES
        ]
        if any(i < 0 for i in geom_ids):
            raise ValueError("pilars XML is missing pilar_fl/fr/rl/rr geoms")
        return cls(
            active=True,
            cfg=cfg,
            rng=rng,
            geom_ids=geom_ids,
            compiled_contype=np.array(model.geom_contype, copy=True),
            compiled_conaffinity=np.array(model.geom_conaffinity, copy=True),
        )

    def apply(self, model: Any) -> PilarSample | None:
        if not self.active:
            return None
        import mujoco

        sample = sample_pilars(self.cfg, self._rng)
        apply_pilar_boxes(
            model,
            sample,
            self.geom_ids,
            self.compiled_contype,
            self.compiled_conaffinity,
            half_xy=self.cfg.half_xy,
        )
        mujoco.mj_forward(model, mujoco.MjData(model))
        return sample

    def spawn_p0_and_xy_yaw(
        self, sample: PilarSample, robot_height: float
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        return spawn_p0_and_xy_yaw(sample, robot_height)

    def foot_anchor(self, sample: PilarSample) -> np.ndarray:
        return foot_anchor(sample)
