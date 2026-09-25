"""Sim-time proprioceptive-image inference switch.

Typical use::

    from mpx.config.sim_config.config_pi_inference import pi_inference_config
    from pi_sim_inference import SimPIInference

    infer = SimPIInference(pi_inference_config)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CFG = _REPO / "mpx/addons/ProprioceptiveImage/PI_NN/config/NN_config.yaml"


@dataclass
class PIInferenceSimConfig:
    """Report-only inference during the addon locomotion sim."""

    enabled: bool = False
    cfg_path: str = str(_DEFAULT_CFG)
    weights_path: str = ""
    print_every: int = 10

    def __post_init__(self) -> None:
        if self.print_every < 1:
            raise ValueError(f"print_every must be >= 1, got {self.print_every}")


pi_inference_config = PIInferenceSimConfig()
