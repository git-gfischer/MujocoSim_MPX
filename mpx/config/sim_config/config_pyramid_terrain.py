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
    # Floor spawns land in (base + floor_margin, base + floor_margin + floor_ring],
    # as max(|x|, |y|). Kept narrow: the out-of-bounds reset is built on it.
    floor_ring: float = 0.3
    safe_spawn_offset: float = 0.8

    base_half: FloatRangeSpec = FloatRangeSpec(enabled=True, low=2.0, high=3.5)
    height: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.15, high=0.40)
    slope_deg: FloatRangeSpec = FloatRangeSpec(enabled=True, low=8.0, high=15.0)
    # Same step as scene_stairs.xml: box size is the half-extent, so the
    # tread is 0.10 m and the rise is 0.05 m. The staired pyramid keeps
    # these fixed and only redraws the footprint and the step count.
    stair_rise: float = 0.05
    stair_tread: float = 0.10
    staired_height: FloatRangeSpec = FloatRangeSpec(enabled=True, low=0.15, high=1.20)


pyramid_terrain_config = PyramidTerrainConfig()
