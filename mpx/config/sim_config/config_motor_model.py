"""
Actuator model between the MPC torque command and the torque MuJoCo applies.

Off by default. Off means ``MotorModel`` returns the command unchanged, so the
simulator keeps its ideal motors: applied torque = commanded torque, clipped
only by the XML ``forcerange``.

On, every physics step, in this order:

1. first-order lag of the applied torque toward the command (current-loop
   bandwidth), time constant ``torque_time_constant_s``;
2. joint torque-speed envelope: full ``peak_torque_nm`` up to
   ``corner_speed_rad_s``, then linear down to 0 at ``no_load_speed_rad_s``.
   The envelope limits only torque that pushes the joint in its direction of
   motion (motoring). Braking torque keeps the full peak.

``joint_torque_measured`` reads ``data.actuator_force``, so with the model on
it is this lagged, saturated torque plus ``SensorNoise``. ``joint_torque_cmd``
stays the MPC command.

Torques and speeds are joint side (after the gearbox). Per-joint tuples are
(HAA hip, HFE thigh, KFE calf) and repeat for the four legs.

The parameter values live in ``mpx/data/go2/go2_motor_params.yaml``; only
``enabled`` is set here.

Typical use::

    from mpx.utils.simulation_utils.motor_model import MotorModel

    motor = MotorModel.from_config(dt=1.0 / sim_hz, n_joints=12)
    data.ctrl = motor(tau_cmd, data.qvel[6:18])
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GO2_MOTOR_PARAMS = Path(__file__).resolve().parents[2] / "data" / "go2" / "go2_motor_params.yaml"


@dataclass
class MotorModelConfig:
    """Configuration for ``MotorModel``. Every ``None`` must be set before enabling."""

    enabled: bool = True

    # Current-loop time constant [s]. 0 disables the lag only.
    torque_time_constant_s: float | None = None

    # Peak joint torque [N*m].
    peak_torque_nm: tuple[float, float, float] | None = None

    # Joint speed where torque starts to drop [rad/s].
    corner_speed_rad_s: tuple[float, float, float] | None = None

    # Joint speed where available torque reaches 0 [rad/s].
    no_load_speed_rad_s: tuple[float, float, float] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path = GO2_MOTOR_PARAMS, *, enabled: bool | None = None) -> "MotorModelConfig":
        """Unknown keys in the file raise, so a typo cannot silently leave a value unset.

        ``enabled=None`` keeps the dataclass default.
        """
        params = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
        if enabled is not None:
            params["enabled"] = enabled
        return cls(**params)


motor_model_config = MotorModelConfig.from_yaml()
