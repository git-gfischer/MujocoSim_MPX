from dataclasses import dataclass

@dataclass(frozen=True)
class BasePoseRandomizationConfig:
    """Sampling ranges for base pose perturbations around a nominal pose."""

    enabled: bool = True
    z_offset_range: tuple[float, float] = (-0.2, 0.2)
    roll_range_deg: tuple[float, float] = (-40.5, 40.5)
    pitch_range_deg: tuple[float, float] = (-40.5, 40.5)
    yaw_range_deg: tuple[float, float] = (-40.0, 40.0)
    min_base_height: float = 0.05