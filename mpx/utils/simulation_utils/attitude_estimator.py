"""
Attitude estimation from the IMU alone — the roll/pitch a real robot actually has.

Why this exists
---------------
A gravity-aligned foot vector needs to know which way is down. In simulation the
answer sits in ``base_quat``, but that is ground truth and is not available on
hardware, so a channel built from it is privileged no matter what its name says.
:class:`ComplementaryAttitudeFilter` produces the same quantity from
``imu_acc_body`` and ``imu_gyro_body`` only, which is what an onboard estimator
would have.

How it works
------------
A complementary filter, the standard cheap attitude estimator:

* the **gyro** integrates well over short horizons but drifts,
* the **accelerometer** gives an absolute gravity direction but is corrupted by
  the robot's own acceleration,

so the gyro supplies the high-frequency estimate and the accelerometer slowly
pulls roll and pitch back toward gravity. The blend constant ``alpha`` is the
gyro's share per step.

**Yaw is not observable** from an accelerometer — gravity says nothing about
heading — so the yaw this filter reports drifts freely. That is fine and is the
reason it is only ever used to build *gravity-aligned* (yaw-removed) frames,
where the yaw cancels out.

The accelerometer correction is gated on the specific-force magnitude being near
1 g: while the robot is accelerating hard, or airborne, the accelerometer is not
measuring gravity and trusting it would tilt the estimate toward the acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY_MAGNITUDE = 9.81


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """wxyz quaternion from intrinsic roll-pitch-yaw (Z-Y-X)."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


@dataclass
class ComplementaryAttitudeFilter:
    """
    Roll/pitch/yaw from body-frame specific force and angular rate.

    Feed it the **measured** IMU channels, the ones that already carry noise and
    bias, so the estimate degrades exactly as it would on the robot.
    """

    dt: float

    # Gyro share per step. 0.98 at 50 Hz gives a ~1 s correction time constant:
    # fast enough to track a real tilt, slow enough that a footfall's acceleration
    # spike does not drag the estimate with it.
    alpha: float = 0.98

    # Accept the accelerometer as a gravity reference only while its magnitude is
    # within this fraction of 1 g. Outside it the robot's own acceleration
    # dominates and the vector no longer points down.
    gravity_tolerance: float = 0.20

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    _initialized: bool = field(default=False, init=False, repr=False)

    def reset(self) -> None:
        """Start a new episode; the next sample seeds roll and pitch."""
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._initialized = False

    def update(self, acc_body: np.ndarray, gyro_body: np.ndarray) -> np.ndarray:
        """
        Advance one control step and return the estimated attitude as wxyz.

        ``acc_body`` is specific force in the IMU frame including gravity;
        ``gyro_body`` is the angular rate in the same frame.
        """
        acc = np.asarray(acc_body, dtype=np.float64).reshape(3)
        gyro = np.asarray(gyro_body, dtype=np.float64).reshape(3)

        # Roll/pitch implied by treating the measured specific force as gravity.
        magnitude = float(np.linalg.norm(acc))
        acc_trustworthy = (
            magnitude > 1e-6
            and abs(magnitude - GRAVITY_MAGNITUDE)
            < self.gravity_tolerance * GRAVITY_MAGNITUDE
        )
        if acc_trustworthy:
            acc_roll = float(np.arctan2(acc[1], acc[2]))
            acc_pitch = float(
                np.arctan2(-acc[0], np.sqrt(acc[1] ** 2 + acc[2] ** 2))
            )
        else:
            acc_roll = acc_pitch = None  # type: ignore[assignment]

        if not self._initialized:
            # Seed from the accelerometer so the filter does not spend its first
            # second ramping up from a level assumption.
            if acc_trustworthy:
                self.roll, self.pitch = acc_roll, acc_pitch
            self._initialized = True
            return self.quaternion()

        # Gyro integration (small-angle body rates into Euler rates).
        sin_r, cos_r = np.sin(self.roll), np.cos(self.roll)
        tan_p = np.tan(np.clip(self.pitch, -1.5, 1.5))
        cos_p = max(np.cos(self.pitch), 1e-6)

        roll_rate = gyro[0] + sin_r * tan_p * gyro[1] + cos_r * tan_p * gyro[2]
        pitch_rate = cos_r * gyro[1] - sin_r * gyro[2]
        yaw_rate = (sin_r / cos_p) * gyro[1] + (cos_r / cos_p) * gyro[2]

        roll = self.roll + roll_rate * self.dt
        pitch = self.pitch + pitch_rate * self.dt
        self.yaw += yaw_rate * self.dt      # unobservable, allowed to drift

        if acc_trustworthy:
            self.roll = self.alpha * roll + (1.0 - self.alpha) * acc_roll
            self.pitch = self.alpha * pitch + (1.0 - self.alpha) * acc_pitch
        else:
            # Gyro-only while the accelerometer is not measuring gravity.
            self.roll, self.pitch = roll, pitch

        return self.quaternion()

    def quaternion(self) -> np.ndarray:
        """Current estimate as a wxyz quaternion."""
        return _quat_from_rpy(self.roll, self.pitch, self.yaw)

    def to_metadata(self) -> dict:
        return {
            "type": "complementary",
            "alpha": float(self.alpha),
            "dt": float(self.dt),
            "gravity_tolerance": float(self.gravity_tolerance),
            "inputs": ["imu_acc_body", "imu_gyro_body"],
            "note": (
                "Yaw is unobservable from an accelerometer and drifts; the "
                "estimate is only used to build gravity-aligned frames, where "
                "yaw cancels."
            ),
        }
