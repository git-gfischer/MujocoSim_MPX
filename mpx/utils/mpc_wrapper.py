"""MPC wrapper compatibility shim.

The single-environment wrapper was split by behaviour:

- :mod:`mpx.utils.mpc_wrapper_locomotion` — shared online MPC core
  (``mpx_data``, ``BatchedMPCControllerWrapper``, ``LocomotionMPCControllerWrapper``).
- :mod:`mpx.utils.mpc_wrapper_balance` — reduced-support balance behaviour
  (``BalanceMPCControllerWrapper``), a subclass of the locomotion wrapper.

``MPCControllerWrapper`` stays available here (aliased to the balance wrapper, which
inherits the full locomotion path) so existing call sites keep working for both
locomotion and balance configurations — the reference generator is selected
automatically from ``config``.
"""

# from mpx.utils.quad_utils_locomotion.mpc_wrapper_locomotion import (
#     mpx_data,
#     BatchedMPCControllerWrapper,
#     LocomotionMPCControllerWrapper,
# )
# from mpx.utils.quad_utils_balance.mpc_wrapper_balance import BalanceMPCControllerWrapper
# from mujoco.mjx._src.dataclasses import PyTreeNode
# import jax.numpy as jnp

# # Backwards-compatible name used across the simulators. The balance wrapper is a
# # superset of the locomotion wrapper, so it handles both behaviours.
# MPCControllerWrapper = BalanceMPCControllerWrapper

# __all__ = [
#     "mpx_data",
#     "BatchedMPCControllerWrapper",
#     "LocomotionMPCControllerWrapper",
#     "BalanceMPCControllerWrapper",
#     "MPCControllerWrapper",
# ]
import jax.numpy as jnp
from mujoco.mjx._src.dataclasses import PyTreeNode
class MPCData(PyTreeNode):
    """Carry state for the pure functional MPC API."""

    dt: float
    duty_factor: float
    step_freq: float
    step_height: float
    contact_time: jnp.ndarray
    liftoff: jnp.ndarray
    X0: jnp.ndarray
    U0: jnp.ndarray
    V0: jnp.ndarray
    W: jnp.ndarray


mpx_data = MPCData