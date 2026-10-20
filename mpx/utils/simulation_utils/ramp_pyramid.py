"""Ramp scene: a smooth truncated pyramid, redrawn every reset.

Each reset shows one convex frustum. A heightfield only catches the feet in
glances, so they roll down the face no matter how high its friction is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpx.utils.spawner.facing_center import opposite_side_xy, sample_facing_center

SLOPE_DEG = (2.5, 22.5)
SLOPE_STEPS = (2.5, 7.5, 12.5, 17.5, 22.5)
BASE_HALF = 2.0
TOP_HALF = 0.4
# Spawns and goals sit this far past the base edge, as max(|x|, |y|).
SPAWN_MARGIN = 0.75
_HIDDEN = (0.0, 0.0, -8.0)


@dataclass(frozen=True)
class RampSample:
    slope_deg: float
    height: float
    base_half: float
    top_half: float


def _geom_ids(model) -> list[int]:
    import mujoco

    ids: list[int] = []
    i = 0
    while True:
        gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"ramp_{i:02d}"))
        if gid < 0:
            break
        ids.append(gid)
        i += 1
    return ids


class RampPyramid:
    """No-op unless ``--scene ramp`` has the ``ramp_00`` .. frustum meshes."""

    def __init__(
        self,
        active: bool,
        rng: np.random.Generator,
        geom_ids: list[int] | None = None,
        contype=None,
        conaffinity=None,
        base_half: float = BASE_HALF,
        top_half: float = TOP_HALF,
    ):
        self.active = active
        self._rng = rng
        self._geom_ids = list(geom_ids or [])
        self._contype = contype
        self._conaffinity = conaffinity
        self.base_half = float(base_half)
        self.top_half = float(top_half)
        self.spawn_reach = self.base_half + SPAWN_MARGIN

    @classmethod
    def from_scene(
        cls,
        scene: str,
        model,
        rng: np.random.Generator | None = None,
    ) -> RampPyramid:
        rng = rng if rng is not None else np.random.default_rng()
        if scene != "ramp":
            return cls(active=False, rng=rng)
        ids = _geom_ids(model)
        if len(ids) != len(SLOPE_STEPS):
            return cls(active=False, rng=rng)
        return cls(
            active=True,
            rng=rng,
            geom_ids=ids,
            contype=np.array(model.geom_contype, copy=True),
            conaffinity=np.array(model.geom_conaffinity, copy=True),
        )

    def apply(self, model) -> RampSample | None:
        if not self.active:
            return None
        draw = float(self._rng.uniform(*SLOPE_DEG))
        index = int(np.argmin(np.abs(np.asarray(SLOPE_STEPS) - draw)))
        slope = float(SLOPE_STEPS[index])
        run = self.base_half - self.top_half
        sample = RampSample(
            slope_deg=slope,
            height=run * float(np.tan(np.deg2rad(slope))),
            base_half=self.base_half,
            top_half=self.top_half,
        )
        for i, gid in enumerate(self._geom_ids):
            body = int(model.geom_bodyid[gid])
            if i == index:
                model.body_pos[body] = (0.0, 0.0, 0.0)
                model.geom_contype[gid] = int(self._contype[gid])
                model.geom_conaffinity[gid] = int(self._conaffinity[gid])
                model.geom_rgba[gid, 3] = 1.0
            else:
                model.body_pos[body] = _HIDDEN
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
                model.geom_rgba[gid, 3] = 0.0
        return sample

    def opposite_goal(self, qpos) -> np.ndarray:
        """Navigation goal on the far side of the pyramid from the robot."""
        return opposite_side_xy(np.asarray(qpos)[:2], self.spawn_reach)

    def try_spawn(self, spawner, model, data, quat0, q0, robot_height: float) -> None:
        from mpx.utils.spawner.spawner import SpawnCollisionError

        attempts = min(32, max(1, int(getattr(spawner, "max_spawn_attempts", 32))))
        last_exc: Exception | None = None
        for _ in range(attempts):
            p0, xy_yaw = sample_facing_center(
                self._rng, self.spawn_reach, robot_height
            )
            try:
                spawner.apply_to_data(model, data, p0, quat0, q0, xy_yaw=xy_yaw)
                return
            except SpawnCollisionError as exc:
                last_exc = exc
        p0 = np.array([0.0, 0.0, float(robot_height)], dtype=np.float64)
        xy_yaw = (self.spawn_reach, 0.0, float(np.pi))
        try:
            spawner.apply_to_data(model, data, p0, quat0, q0, xy_yaw=xy_yaw)
        except SpawnCollisionError:
            if last_exc is not None:
                raise last_exc
            raise
