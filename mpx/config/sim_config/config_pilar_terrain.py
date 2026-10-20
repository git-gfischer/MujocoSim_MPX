"""Sampling ranges for the still4 pilars scene.

Change ``height.low`` and ``height.high`` (metres) to change how tall the
pillars are. ``half_xy`` is the horizontal half-size. Both the simulator
and the tests read this object.
"""

from __future__ import annotations

from dataclasses import dataclass

from mpx.config.sim_config.config_reset_randomization import FloatRangeSpec


@dataclass
class PilarTerrainConfig:
    """Constants for :class:`mpx.utils.simulation_utils.pilar_terrain.PilarTerrain`."""

    half_xy: float = 0.20
    height: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.0, high=0.15)


pilar_terrain_config = PilarTerrainConfig()
