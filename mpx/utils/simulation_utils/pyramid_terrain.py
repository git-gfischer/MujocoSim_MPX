"""Sample and apply truncated-pyramid terrains on MuJoCo respawn."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mpx.config.sim_config.config_pyramid_terrain import (
    PyramidTerrainConfig,
    pyramid_terrain_config,
)
from mpx.config.sim_config.config_reset_randomization import FloatRangeSpec


@dataclass(frozen=True)
class PyramidSample:
    """Realized pyramid dimensions for one episode."""

    kind: str
    base_half: float
    top_half: float
    height: float
    slope_deg: float | None = None
    rise: float | None = None
    tread: float | None = None
    n_steps: int | None = None
    spawn_on_top: bool = False

    def to_metadata(self) -> dict[str, float | int | str | bool]:
        meta: dict[str, float | int | str | bool] = {
            "pyramid_kind": self.kind,
            "base_half": float(self.base_half),
            "top_half": float(self.top_half),
            "height": float(self.height),
            "spawn_on_top": bool(self.spawn_on_top),
        }
        if self.slope_deg is not None:
            meta["slope_deg"] = float(self.slope_deg)
        if self.rise is not None:
            meta["rise"] = float(self.rise)
        if self.tread is not None:
            meta["tread"] = float(self.tread)
        if self.n_steps is not None:
            meta["n_steps"] = int(self.n_steps)
        return meta


def _draw(spec: FloatRangeSpec, rng: np.random.Generator) -> float:
    return float(rng.uniform(spec.low, spec.high))


def _smooth_ok(base_half: float, top_half: float, min_top_half: float) -> bool:
    return top_half >= min_top_half and top_half < base_half


def _layer_geom_name(index: int) -> str:
    return f"pyr_{index:02d}"


def sample_pyramid(
    cfg: PyramidTerrainConfig, rng: np.random.Generator
) -> PyramidSample:
    """Smooth frustum approximated by thin box steps (same construction as stairs)."""
    rise = float(cfg.smooth_rise)
    last: tuple[float, float, float] | None = None
    for _ in range(max(1, int(cfg.sample_retries))):
        base_half = _draw(cfg.base_half, rng)
        height = _draw(cfg.height, rng)
        slope_deg = _draw(cfg.slope_deg, rng)
        tan_s = float(np.tan(np.deg2rad(slope_deg)))
        if tan_s <= 1e-9:
            continue
        tread = rise / tan_s
        n_steps = int(np.clip(round(height / rise), 2, cfg.max_layers))
        top_half = base_half - (n_steps - 1) * tread
        last = (base_half, height, slope_deg)
        if _smooth_ok(base_half, top_half, cfg.min_top_half) and tread > 1e-4:
            return PyramidSample(
                kind="pyramid",
                base_half=base_half,
                top_half=top_half,
                height=n_steps * rise,
                slope_deg=slope_deg,
                rise=rise,
                tread=tread,
                n_steps=n_steps,
            )
    slope_deg = last[2] if last is not None else cfg.slope_deg.low
    tan_s = max(float(np.tan(np.deg2rad(slope_deg))), 1e-6)
    tread = rise / tan_s
    n_steps = int(np.clip(round((last[1] if last else cfg.height.low) / rise), 2, cfg.max_layers))
    top_half = cfg.min_top_half
    base_half = top_half + (n_steps - 1) * tread
    if base_half > cfg.max_base_half:
        n_steps = max(2, 1 + int((cfg.max_base_half - top_half) / max(tread, 1e-4)))
        n_steps = min(n_steps, cfg.max_layers)
        base_half = top_half + (n_steps - 1) * tread
    return PyramidSample(
        kind="pyramid",
        base_half=base_half,
        top_half=top_half,
        height=n_steps * rise,
        slope_deg=slope_deg,
        rise=rise,
        tread=tread,
        n_steps=n_steps,
    )


def sample_staired_pyramid(
    cfg: PyramidTerrainConfig, rng: np.random.Generator
) -> PyramidSample:
    """Stair-sized steps. Only the base (XY) and the step count (Z) change."""
    rise = float(cfg.stair_rise)
    tread = float(cfg.stair_tread)
    base_half = _draw(cfg.base_half, rng)
    max_n = 1 + int((base_half - cfg.min_top_half) / max(tread, 1e-4))
    max_n = int(np.clip(max_n, 2, cfg.max_layers))
    n_lo = int(np.clip(round(cfg.staired_height.low / rise), 2, max_n))
    n_hi = int(np.clip(round(cfg.staired_height.high / rise), n_lo, max_n))
    n_steps = int(rng.integers(n_lo, n_hi + 1))
    top_half = base_half - (n_steps - 1) * tread
    return PyramidSample(
        kind="staired_pyramid",
        base_half=base_half,
        top_half=top_half,
        height=n_steps * rise,
        rise=rise,
        tread=tread,
        n_steps=n_steps,
    )


def _set_box(
    model: Any,
    geom_id: int,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    contype: int,
    conaffinity: int,
) -> None:
    """Move a welded body (collision + visuals) and resize its box geom.

    Worldbody geoms keep a compile-time BVH, so the viewer can follow
    ``geom_pos``/``geom_size`` while contacts still use the XML AABB. Each
    layer sits on its own body; ``body_pos`` updates both.
    """
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


# Park unused layers underground so they cannot steal rays or contacts.
PLACEHOLDER_POS = (0.0, 0.0, -2.0)


def _hide_box(model: Any, geom_id: int) -> None:
    _set_box(
        model,
        geom_id,
        PLACEHOLDER_POS,
        (1e-4, 1e-4, 1e-4),
        contype=0,
        conaffinity=0,
    )


def apply_pyramid_boxes(
    model: Any,
    sample: PyramidSample,
    layer_ids: list[int],
    compiled_contype: Any,
    compiled_conaffinity: Any,
) -> None:
    """Same plates as ``scene_stairs.xml``: each tread is ``rise`` thick.

    Stairs sit the box center at ``(i + 0.5) * rise`` with half-height
    ``rise / 2``. A column from the floor makes the height map treat the
    whole stack as one tall solid.
    """
    n_steps = int(sample.n_steps or 0)
    rise = float(sample.rise or 0.05)
    tread = float(sample.tread or 0.10)
    base_half = float(sample.base_half)
    first = int(layer_ids[0])
    on_type = int(compiled_contype[first]) or 1
    on_aff = int(compiled_conaffinity[first]) or 1
    half_z = rise / 2.0
    for i, geom_id in enumerate(layer_ids):
        if i < n_steps:
            half_xy = base_half - i * tread
            _set_box(
                model,
                int(geom_id),
                (0.0, 0.0, (i + 0.5) * rise),
                (half_xy, half_xy, half_z),
                contype=on_type,
                conaffinity=on_aff,
            )
        else:
            _hide_box(model, int(geom_id))


def floor_ring_outer(
    sample: PyramidSample, cfg: PyramidTerrainConfig = pyramid_terrain_config
) -> float:
    """Farthest ``max(|x|, |y|)`` a floor spawn can land at."""
    return float(sample.base_half) + cfg.floor_margin + cfg.floor_ring


def opposite_goal(
    qpos, sample: PyramidSample, cfg: PyramidTerrainConfig = pyramid_terrain_config
) -> np.ndarray:
    """Goal straight across the pyramid, mid-way through the far floor ring.

    Off the base, inside every floor spawn and so inside the reset region.
    """
    v = -np.asarray(qpos, dtype=np.float64)[:2]
    if float(np.max(np.abs(v))) < 1e-6:
        v = np.array([1.0, 0.0])
    reach = float(sample.base_half) + cfg.floor_margin + 0.5 * cfg.floor_ring
    return v * reach / float(np.max(np.abs(v)))


def spawn_p0_and_xy_yaw(
    sample: PyramidSample,
    robot_height: float,
    rng: np.random.Generator,
    cfg: PyramidTerrainConfig = pyramid_terrain_config,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return base ``p0`` (z only) and ``(x, y, yaw)`` for ``apply_to_data``."""
    if sample.spawn_on_top:
        half = max(0.05, float(sample.top_half) - cfg.top_margin)
        x = float(rng.uniform(-half, half))
        y = float(rng.uniform(-half, half))
        p0 = np.array([0.0, 0.0, float(sample.height) + float(robot_height)], dtype=np.float64)
    else:
        inner = float(sample.base_half) + cfg.floor_margin
        outer = floor_ring_outer(sample, cfg)
        x = y = 0.0
        for _ in range(64):
            x = float(rng.uniform(-outer, outer))
            y = float(rng.uniform(-outer, outer))
            if max(abs(x), abs(y)) > inner:
                break
        else:
            x = inner + 0.1
            y = 0.0
        p0 = np.array([0.0, 0.0, float(robot_height)], dtype=np.float64)
    yaw = float(rng.uniform(-np.pi, np.pi))
    return p0, (x, y, yaw)


class PyramidTerrain:
    """No-op unless ``--scene`` is ``pyramid`` or ``staired_pyramid``."""

    def __init__(
        self,
        kind: str | None,
        cfg: PyramidTerrainConfig,
        rng: np.random.Generator,
        layer_ids: list[int] | None = None,
        compiled_contype: Any = None,
        compiled_conaffinity: Any = None,
    ):
        self.kind = kind
        self.cfg = cfg
        self._rng = rng
        self.layer_ids = list(layer_ids or [])
        self.compiled_contype = compiled_contype
        self.compiled_conaffinity = compiled_conaffinity

    @classmethod
    def from_scene(
        cls,
        scene: str,
        model: Any,
        cfg: PyramidTerrainConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> PyramidTerrain:
        cfg = cfg or pyramid_terrain_config
        rng = rng if rng is not None else np.random.default_rng()
        if scene not in ("pyramid", "staired_pyramid"):
            return cls(kind=None, cfg=cfg, rng=rng)
        import mujoco

        layer_ids = [
            int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, _layer_geom_name(i)
                )
            )
            for i in range(cfg.max_layers)
        ]
        if any(i < 0 for i in layer_ids):
            raise ValueError("pyramid XML is missing pyr_00.. geoms")
        return cls(
            kind=scene,
            cfg=cfg,
            rng=rng,
            layer_ids=layer_ids,
            compiled_contype=np.array(model.geom_contype, copy=True),
            compiled_conaffinity=np.array(model.geom_conaffinity, copy=True),
        )

    def apply(self, model: Any) -> PyramidSample | None:
        if self.kind is None:
            return None
        spawn_on_top = bool(self._rng.random() < 0.5)
        if self.kind == "pyramid":
            sample = sample_pyramid(self.cfg, self._rng)
        else:
            sample = sample_staired_pyramid(self.cfg, self._rng)
        sample = replace(sample, spawn_on_top=spawn_on_top)
        apply_pyramid_boxes(
            model,
            sample,
            self.layer_ids,
            self.compiled_contype,
            self.compiled_conaffinity,
        )
        return sample

    def spawn_p0_and_xy_yaw(
        self, sample: PyramidSample, robot_height: float
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        return spawn_p0_and_xy_yaw(sample, robot_height, self._rng, self.cfg)

    def floor_ring_outer(self, sample: PyramidSample) -> float:
        return floor_ring_outer(sample, self.cfg)

    def try_spawn(
        self,
        spawner: Any,
        model: Any,
        data: Any,
        quat0: Any,
        q0: Any,
        sample: PyramidSample,
        robot_height: float,
    ) -> None:
        from mpx.utils.spawner.spawner import SpawnCollisionError

        attempts = max(1, int(getattr(spawner, "max_spawn_attempts", 512)))
        last_exc: Exception | None = None
        for _ in range(attempts):
            p0, xy_yaw = self.spawn_p0_and_xy_yaw(sample, robot_height)
            try:
                spawner.apply_to_data(model, data, p0, quat0, q0, xy_yaw=xy_yaw)
                return
            except SpawnCollisionError as exc:
                last_exc = exc
        p0 = np.array([0.0, 0.0, float(robot_height)], dtype=np.float64)
        xy_yaw = (float(sample.base_half) + self.cfg.safe_spawn_offset, 0.0, 0.0)
        try:
            spawner.apply_to_data(model, data, p0, quat0, q0, xy_yaw=xy_yaw)
        except SpawnCollisionError:
            if last_exc is not None:
                raise last_exc
            raise


def stamp_scene(scene: str, model: Any, rng: np.random.Generator | None = None) -> None:
    """Place ramp or staired-pyramid geoms. Other scenes keep the XML."""
    from mpx.utils.simulation_utils.ramp_pyramid import RampPyramid

    ramp = RampPyramid.from_scene(scene, model, rng=rng)
    if ramp.active:
        ramp.apply(model)
    PyramidTerrain.from_scene(scene, model, rng=rng).apply(model)


def copy_scene_stamp(src: Any, dst: Any) -> None:
    """Copy a stamped terrain onto another model loaded from the same XML."""
    dst.body_pos[:] = src.body_pos
    dst.geom_pos[:] = src.geom_pos
    dst.geom_size[:] = src.geom_size
    dst.geom_rbound[:] = src.geom_rbound
    dst.geom_contype[:] = src.geom_contype
    dst.geom_conaffinity[:] = src.geom_conaffinity
    dst.geom_rgba[:] = src.geom_rgba
    dst.geom_aabb[:] = src.geom_aabb
