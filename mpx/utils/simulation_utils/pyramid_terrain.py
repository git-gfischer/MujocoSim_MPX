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


def _staired_from_draws(
    base_half: float,
    height: float,
    rise: float,
    tread: float,
    *,
    min_top_half: float,
    max_layers: int,
) -> tuple[float, float, float, float, float, int]:
    rise = max(float(rise), 1e-4)
    tread = max(float(tread), 1e-4)
    n_steps = int(np.clip(round(height / rise), 2, max_layers))

    def top_for(n: int, base: float) -> float:
        return base - (n - 1) * tread

    while n_steps >= 1:
        top_half = top_for(n_steps, base_half)
        if top_half >= min_top_half and top_half < base_half:
            return base_half, top_half, n_steps * rise, rise, tread, n_steps
        n_steps -= 1
        if n_steps < 2:
            break
    n_steps = 2
    top_half = top_for(n_steps, base_half)
    if top_half < min_top_half or top_half >= base_half:
        base_half = min_top_half + n_steps * tread + 0.05
        top_half = top_for(n_steps, base_half)
    return base_half, top_half, n_steps * rise, rise, tread, n_steps


def sample_staired_pyramid(
    cfg: PyramidTerrainConfig, rng: np.random.Generator
) -> PyramidSample:
    last: tuple[float, float, float, float] | None = None
    for _ in range(max(1, int(cfg.sample_retries))):
        base_half = _draw(cfg.base_half, rng)
        height = _draw(cfg.height, rng)
        rise = _draw(cfg.rise, rng)
        tread = _draw(cfg.tread, rng)
        last = (base_half, height, rise, tread)
        n_steps = int(np.clip(round(height / max(rise, 1e-4)), 2, cfg.max_layers))
        top_half = base_half - (n_steps - 1) * tread
        if top_half >= cfg.min_top_half and top_half < base_half:
            return PyramidSample(
                kind="staired_pyramid",
                base_half=base_half,
                top_half=top_half,
                height=n_steps * rise,
                rise=rise,
                tread=tread,
                n_steps=n_steps,
            )
    base_half, top_half, height, rise, tread, n_steps = _staired_from_draws(
        last[0], last[1], last[2], last[3],
        min_top_half=cfg.min_top_half,
        max_layers=cfg.max_layers,
    )
    return PyramidSample(
        kind="staired_pyramid",
        base_half=base_half,
        top_half=top_half,
        height=height,
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
    """Solid ziggurat: each layer is a square column from the floor up.

    Thin stacked treads can look like a pyramid in the viewer while physics
    tunnels through them. Building each layer from z=0 makes the visible
    volume the same as the collision volume.
    """
    n_steps = int(sample.n_steps or 0)
    rise = float(sample.rise or 0.05)
    tread = float(sample.tread or 0.10)
    base_half = float(sample.base_half)
    first = int(layer_ids[0])
    on_type = int(compiled_contype[first]) or 1
    on_aff = int(compiled_conaffinity[first]) or 1
    for i, geom_id in enumerate(layer_ids):
        if i < n_steps:
            half_xy = base_half - i * tread
            height_i = (i + 1) * rise
            half_z = height_i / 2.0
            _set_box(
                model,
                int(geom_id),
                (0.0, 0.0, half_z),
                (half_xy, half_xy, half_z),
                contype=on_type,
                conaffinity=on_aff,
            )
        else:
            _hide_box(model, int(geom_id))


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
        outer = max(inner + cfg.floor_ring, 4.0)
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
