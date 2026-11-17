"""Time-varying slip: randomize the sliding μ between each foot and the terrain.

The contact μ MuJoCo uses is the foot geom's, because the foot priority is
above the floor. The value written there is min(foot draw, terrain draw),
which is the pair friction. The terrain draw is also written onto the floor,
ramp, and stair geoms. ``slip_event`` is 1 on a foot while its event is on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlipInjectionConfig:
    # Below and around the MPC cone (0.5–0.7). Held for the whole event.
    # μ is uniform over the union of the bands, so a gap such as
    # ((0.15, 0.2), (0.3, 0.55)) holds out [0.2, 0.3) for testing.
    mu_bands: tuple[tuple[float, float], ...] = ((0.15, 0.55),)
    on_s: tuple[float, float] = (0.4, 1.5)
    off_s: tuple[float, float] = (1.0, 3.0)

    def __post_init__(self) -> None:
        if not self.mu_bands or any(not 0.0 < lo <= hi for lo, hi in self.mu_bands):
            raise ValueError(f"mu_bands must be non-empty (low, high) pairs with 0 < low <= high: {self.mu_bands}")


def parse_mu_bands(text: str) -> tuple[tuple[float, float], ...]:
    """``"0.15-0.2,0.3-0.55"`` -> ``((0.15, 0.2), (0.3, 0.55))``."""
    bands = []
    for part in text.split(","):
        lo, hi = part.split("-")
        bands.append((float(lo), float(hi)))
    return tuple(bands)


def draw_mu(rng, bands) -> float:
    """Uniform over the union of ``bands``: pick a band by width, then uniform inside it."""
    widths = np.array([hi - lo for lo, hi in bands], dtype=np.float64)
    p = widths / widths.sum() if widths.sum() > 0 else None
    lo, hi = bands[int(rng.choice(len(bands), p=p))]
    return float(rng.uniform(lo, hi))


def terrain_geom_ids(model, foot_geom_ids) -> np.ndarray:
    """Collision geoms that are not part of the robot (floor, ramp, stairs, rubble)."""
    foot_bodies = {int(model.geom_bodyid[int(g)]) for g in np.asarray(foot_geom_ids).reshape(-1)}
    roots: set[int] = set()
    for body in foot_bodies:
        while int(model.body_parentid[body]) != 0:
            body = int(model.body_parentid[body])
        roots.add(int(body))

    def _robot(body: int) -> bool:
        while body != 0:
            if body in roots:
                return True
            body = int(model.body_parentid[body])
        return False

    ids = [
        gid
        for gid in range(int(model.ngeom))
        if not _robot(int(model.geom_bodyid[gid]))
        and (
            int(model.geom_contype[gid]) != 0
            or int(model.geom_conaffinity[gid]) != 0
        )
    ]
    if not ids:
        raise ValueError("slip injection found no terrain collision geoms")
    return np.asarray(ids, dtype=int)


class SlipInjector:
    def __init__(self, model, foot_geom_ids, sim_hz: float, rng, cfg: SlipInjectionConfig | None = None):
        self.cfg = cfg or SlipInjectionConfig()
        self._rng = rng
        self.dt = 1.0 / float(sim_hz)
        self.foot_ids = np.asarray(foot_geom_ids, dtype=int).reshape(-1)
        self.terrain_ids = terrain_geom_ids(model, self.foot_ids)
        self.n = int(self.foot_ids.size)
        self.nominal_foot = np.array(model.geom_friction[self.foot_ids, 0], dtype=np.float64)
        self.nominal_terrain = np.array(model.geom_friction[self.terrain_ids, 0], dtype=np.float64)
        self.active = np.zeros(self.n, dtype=bool)
        self.left = np.zeros(self.n, dtype=np.float64)
        self.foot_draw = self.nominal_foot.copy()
        self._terrain_draw: float | None = None

    def capture_nominal(self, model) -> None:
        """Episode foot μ, after the reset randomizer. Terrain goes back to the XML value."""
        self.nominal_foot = np.array(model.geom_friction[self.foot_ids, 0], dtype=np.float64)
        self.active[:] = False
        self._terrain_draw = None
        self.left[:] = self._rng.uniform(self.cfg.off_s[0], self.cfg.off_s[1], size=self.n)
        self._write(model, self.nominal_foot, self.nominal_terrain)

    def tick(self, model) -> np.ndarray:
        """One sim step. Returns ``slip_event`` ``(n_feet,)`` uint8."""
        self.left -= self.dt
        for i in np.flatnonzero(self.left <= 0.0):
            if self.active[i]:
                self.active[i] = False
                self.left[i] = float(self._rng.uniform(*self.cfg.off_s))
            else:
                self.active[i] = True
                self.left[i] = float(self._rng.uniform(*self.cfg.on_s))
                self.foot_draw[i] = draw_mu(self._rng, self.cfg.mu_bands)
        if self.active.any():
            if self._terrain_draw is None:
                self._terrain_draw = draw_mu(self._rng, self.cfg.mu_bands)
            applied = np.where(
                self.active,
                np.minimum(self.foot_draw, self._terrain_draw),
                self.nominal_foot,
            )
            self._write(model, applied, self._terrain_draw)
        else:
            self._terrain_draw = None
            self._write(model, self.nominal_foot, self.nominal_terrain)
        return self.active.astype(np.uint8)

    def _write(self, model, foot_mu, terrain_mu) -> None:
        model.geom_friction[self.foot_ids, 0] = np.asarray(foot_mu, dtype=np.float64)
        model.geom_friction[self.terrain_ids, 0] = (
            float(terrain_mu) if np.ndim(terrain_mu) == 0 else terrain_mu
        )
