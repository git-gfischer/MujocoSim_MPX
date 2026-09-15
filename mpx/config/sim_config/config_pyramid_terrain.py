"""Sampling ranges for randomized truncated-pyramid terrains."""

from __future__ import annotations

from dataclasses import dataclass

from mpx.config.sim_config.config_reset_randomization import FloatRangeSpec


@dataclass
class PyramidTerrainConfig:
    """Uniform ranges and constants for :class:`PyramidTerrain`."""

    sample_retries: int = 32
    max_layers: int = 24
    min_top_half: float = 0.4
    max_base_half: float = 3.8
    smooth_rise: float = 0.025

    floor_margin: float = 0.6
    top_margin: float = 0.15
    floor_ring: float = 1.0
    safe_spawn_offset: float = 0.8

    base_half: FloatRangeSpec = FloatRangeSpec(enabled=True, low=2.0, high=3.5)
    height: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.15, high=0.40)
    slope_deg: FloatRangeSpec = FloatRangeSpec(enabled=True, low=8.0, high=15.0)
    rise: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.025, high=0.045)
    tread: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.20, high=0.32)


pyramid_terrain_config = PyramidTerrainConfig()
