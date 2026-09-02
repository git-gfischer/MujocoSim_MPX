"""
Random external base-force pulse settings for quadruped simulations.

Typical use::

    from mpx.config.sim_config.config_ext_base_forces import ext_base_force_config
    from mpx.utils.simulation_utils.base_force_perturbation import RandomBaseForcePerturbation

    pert = RandomBaseForcePerturbation(
        rng=np.random.default_rng(ext_base_force_config.rng_seed),
        sim_dt=1.0 / sim_frequency,
        force_magnitude_range=ext_base_force_config.force_magnitude_range,
        duration_range_s=ext_base_force_config.duration_range_s,
        cooldown_range_s=ext_base_force_config.cooldown_range_s,
        enabled=ext_base_force_config.enabled,
    )
    pert.reset(start_after_cooldown=ext_base_force_config.start_after_cooldown)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtBaseForceConfig:
    """Configuration for ``RandomBaseForcePerturbation``."""

    enabled: bool = True

    # Random constant force pulse magnitude [N].
    force_magnitude_range: tuple[float, float] = (10.0, 50.0)

    # Pulse duration [s].
    duration_range_s: tuple[float, float] = (0.05, 0.30)

    # Idle wait between pulses [s].
    cooldown_range_s: tuple[float, float] = (0.3, 1.0)

    # Reproducibility for pulse timing/magnitude/orientation sampling.
    rng_seed: int | None = 42

    # If True, first pulse starts after a sampled cooldown on reset.
    start_after_cooldown: bool = True


# Default profile used by examples; reassign or construct ``ExtBaseForceConfig(...)`` to tune.
ext_base_force_config = ExtBaseForceConfig()
