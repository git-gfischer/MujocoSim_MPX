"""
Random map spawn settings for quadruped examples (``mjx_quad.py``, etc.).

Sample base X/Y/yaw inside a :class:`~mpx.utils.spawner.SpawnRegion`; Z comes from
the robot ``config.p0`` plus optional foot vertical relief in the spawner.

Typical use::

    from mpx.config.config_spawn import spawn_config, SpawnConfig

    # Defaults (module-level instance):
    region = spawn_config.spawn_region()

    # Or override fields / construct a custom profile:
    spawn_config = SpawnConfig(verbose=True, rng_seed=0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from mpx.utils.spawner.spawner import SpawnRegion
import glfw


# @dataclass(frozen=True)
# class BasePoseRandomizationConfig:
#     """Sampling ranges for base pose perturbations around a nominal pose."""

#     enabled: bool = True
#     z_offset_range: tuple[float, float] = (-0.2, 0.2)
#     roll_range_deg: tuple[float, float] = (-40.5, 40.5)
#     pitch_range_deg: tuple[float, float] = (-40.5, 40.5)
#     yaw_range_deg: tuple[float, float] = (-180.0, 180.0)
#     min_base_height: float = 0.05

@dataclass
class SpawnConfig:
    """Map spawn sampling, collision checks, viewer overlay, and keyboard respawn."""

    # XY/yaw sampling bounds (metres, radians). Base Z is not sampled here.
    manual_region_x: tuple[float, float] = (0.5, 6.5) # meters
    manual_region_y: tuple[float, float] = (-3.5, 3.5) # meters
    region_yaw: tuple[float, float] = (-np.pi, np.pi)

    use_random_map_spawn: bool = True
    show_spawn_region: bool = False  # red transparent ring in passive viewer
    region_z: float = 0.08  # ring height above floor [m]

    # Collision validation (see ``RobotMapSpawner``).
    check_collisions: bool = True
    check_self: bool = True
    check_env: bool = True
    max_attempts: int = 512
    on_collision_exhausted: Literal["raise", "origin"] = "origin"
    robot_root_body_name: str = "base"

    # Raise base z in steps after each (x, y, yaw) sample so feet clear the floor.
    try_foot_vertical_relief: bool = True
    foot_relief_step: float = 0.005  # [m]
    foot_relief_max: float = 0.10  # [m]

    # Base-pose randomization applied before spawn/reset in balance mode.
    # base_pose: BasePoseRandomizationConfig = field(
    # default_factory=BasePoseRandomizationConfig)

    rng_seed: int | None = None  # e.g. 0 for repeatable random poses
    verbose: bool = False

    # GLFW keycodes: B, Backspace — new random map spawn (not MuJoCo fixed snapshot reset).
    respawn_keycodes: tuple[int, ...] = (66, 259)

    crash_height_fraction: float = 0.5   # fraction of robot_height
    crash_tilt_deg: float = 60.0

    @property
    def check_self_collision(self) -> bool:
        return self.check_collisions and self.check_self

    @property
    def check_env_collision(self) -> bool:
        return self.check_collisions and self.check_env

    @property
    def check_any_collision(self) -> bool:
        return self.check_self_collision or self.check_env_collision

    def spawn_region(self) -> SpawnRegion:
        """Build a :class:`SpawnRegion` from ``manual_region_*`` and ``region_yaw``."""
        return SpawnRegion(
            x=self.manual_region_x,
            y=self.manual_region_y,
            yaw=self.region_yaw,
        )


# Default profile for examples; reassign or construct ``SpawnConfig(...)`` to tune.
spawn_config = SpawnConfig()
