from jax import numpy as jnp
from functools import partial

from mpx.utils.quad_utils_balance.reference_generator_balance import reference_generator_balance

from mpx.utils.quad_utils_locomotion.reference_generator_locomotion import (
    reference_generator_locomotion,
    reference_generator_srbd,
    reference_barell_roll,
)

__all__ = [
    "whole_body_reference_partial",
    "reference_generator_balance",
    "reference_generator_locomotion",
    "reference_generator_srbd",
    "reference_barell_roll",
]


# region choose the reference generator automatically based on the controller config
def _is_balance_controller(config) -> bool:
    """
    Infer whether the active controller should use balance references.

    Priority:
    1) explicit behaviour tag (``balance`` / ``locomotion``)
    2) legacy ``use_balance_fixed_contact`` flag.
    """
    behaviour = getattr(config, "behaviour", None)
    if behaviour is not None:
        behaviour_key = str(behaviour).strip().lower()
        if behaviour_key == "balance":
            return True
        if behaviour_key == "locomotion":
            return False
    return bool(getattr(config, "use_balance_fixed_contact", False))


def whole_body_reference_partial(config, robot_mass, *, clearence_speed_preset: float | None):
    """
    Build a partially-bound whole-body reference generator.

    Locomotion/balance is selected automatically from the controller config.
    """
    bind_kw: dict = {
        "use_terrain_estimator": config.use_terrain_estimation,
        "N": config.N,
        "dt": config.dt,
        "n_joints": config.n_joints,
        "n_contact": config.n_contact,
        "mass": robot_mass,
        "foot0": config.p_legs0,
        "q0": config.q0,
    }
    if clearence_speed_preset is not None:
        bind_kw["clearence_speed"] = clearence_speed_preset

    if _is_balance_controller(config):
        bind_kw["fixed_contact_mask"] = getattr(
            config,
            "balance_fixed_contact_mask",
            jnp.ones(config.n_contact, dtype=jnp.float32),
        )
        return partial(reference_generator_balance, **bind_kw)
    return partial(reference_generator_locomotion, **bind_kw)
# endregion
