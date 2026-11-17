"""Which controller the simulators run. The default is the MPX MPC."""

DEFAULT_CONTROLLER = "mpc"


def make_controller(config, task, *, name=DEFAULT_CONTROLLER, feet=None, limited_memory=True):
    """Build the MuJoCo connector for ``task``.

    ``task`` is ``"locomotion"`` or ``"balance"``. Balance also needs ``feet``
    of 3 or 4. ``name`` selects the control law. Only ``"mpc"`` is registered.
    """
    if name != DEFAULT_CONTROLLER:
        raise ValueError(
            f"unknown controller {name!r}. The default is {DEFAULT_CONTROLLER!r}. "
            "See framework/controllers/README.md."
        )
    from framework.controllers.mpx.connectors.mujoco import MujocoMpxConnector

    if task == "locomotion":
        return MujocoMpxConnector.for_locomotion(config, limited_memory=limited_memory)
    if task == "balance":
        return MujocoMpxConnector.for_balance(config, feet, limited_memory=limited_memory)
    raise ValueError(f"mpc has no task {task!r}; expected locomotion or balance")
