"""
Live proprioception plotter settings for quadruped simulations.

Which signal windows open is fixed here. Set ``enabled=True`` to spawn them
(no ``--plot`` CLI flag). Each plot window still has a buffer-length slider
so the visible history can change live.

Typical use::

    from mpx.config.sim_config.config_live_plotter import live_plotter_config
    from mpx.utils.simulation_utils.live_plotter import ProprioceptivePlotter

    plotter = ProprioceptivePlotter.from_config(cfg=live_plotter_config)
"""
from __future__ import annotations

from dataclasses import dataclass


# Names accepted by ``ProprioceptivePlotter.SIGNALS``.
LIVE_PLOTTER_SIGNALS = (
    "Torque",
    "JointPos",
    "JointVel",
    "FootContacts",
    "GRF",
    "FootVel",
    "AngVel",
    "LinAcc",
)


@dataclass
class LivePlotterConfig:
    """Configuration for ``ProprioceptivePlotter``."""

    # If False, simulators skip spawning the plotter (replaces the old ``--plot`` flag).
    enabled: bool = False

    # Signal windows to spawn. Order is ignored; unknown names are skipped.
    signals: tuple[str, ...] = (  
                                  #"JointPos",
                                  #"JointVel",
                                  "Torque",
                                  #"GRF",
                                  #"FootVel",
                                  #"FootContacts", 
                                  #"AngVel",
                                 # "LinAcc"
                                )

    # Initial sliding-window length [samples]. Editable live via the buffer slider.
    window_size: int = 200

    # Inclusive range of the in-figure buffer slider [samples].
    buffer_range: tuple[int, int] = (50, 1000)


# Default profile used by examples; reassign or construct ``LivePlotterConfig(...)`` to tune.
live_plotter_config = LivePlotterConfig()
