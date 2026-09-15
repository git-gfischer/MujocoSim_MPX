"""
Collection-time observation noise for joint encoders, actuator current and IMU.

What it models
--------------
- **Joint position**: additive Gaussian, then encoder quantisation (in that
  order — quantising first and then adding noise would not produce the staircase
  a real encoder shows).
- **Joint velocity**: additive Gaussian on the raw differentiated signal, then a
  low-pass filter, so the filter shapes the noise the way a motor driver does.
  Real velocity is differentiated, not measured, and is far noisier than position.
- **Joint torque**: additive Gaussian plus a multiplicative scale error drawn
  once per episode — torque is inferred from motor current, so a per-joint gain
  error persists for a whole run rather than averaging out.
- **IMU**: additive Gaussian plus a bias that starts at a random offset and
  random-walks through the episode. The realised bias is returned so it can be
  logged as a privileged channel and drift effects analysed directly.
- **Transport**: whole-sample latency per sensor group, and packet dropout that
  holds the previous sample and raises a staleness flag.

The simulator state and the MPC are never touched: this operates on copied
observation arrays only. ``reset()`` starts a new episode — it redraws the
initial IMU bias and the torque scale error and clears the filter and delay
lines.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict

import numpy as np

from mpx.config.sim_config.config_sensor_noise import (
    JointVelFilterConfig,
    LatencyConfig,
    SensorNoiseConfig,
    sensor_noise_config,
)


def _butter_lowpass(cutoff_hz: float, dt: float, order: int):
    """
    Butterworth low-pass coefficients, or ``None`` when SciPy is unavailable.

    Falls back to a first-order IIR in :class:`SensorNoise` so collection never
    depends on SciPy being installed.
    """
    try:
        from scipy.signal import butter  # noqa: PLC0415  (optional dependency)
    except ImportError:
        return None
    nyquist = 0.5 / float(dt)
    normalized = min(float(cutoff_hz) / nyquist, 0.99)
    return butter(int(order), normalized, btype="low", output="ba")


@dataclass
class SensorReading:
    """One control step as the robot's own sensors would report it."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_torque: np.ndarray
    imu_acc: np.ndarray
    imu_gyro: np.ndarray
    imu_acc_bias: np.ndarray
    imu_gyro_bias: np.ndarray
    stale: bool


@dataclass
class SensorNoise:
    """Corrupt proprioceptive observations; never mutate simulator state."""

    rng: np.random.Generator
    dt: float
    enabled: bool = sensor_noise_config.enabled
    joint_pos_std: float = sensor_noise_config.joint_pos_std
    joint_pos_quantization_rad: float = sensor_noise_config.joint_pos_quantization_rad
    joint_vel_std: float = sensor_noise_config.joint_vel_std
    joint_vel_filter: JointVelFilterConfig = field(
        default_factory=lambda: sensor_noise_config.joint_vel_filter
    )
    joint_torque_std: float = sensor_noise_config.joint_torque_std
    joint_torque_scale_err_std: float = sensor_noise_config.joint_torque_scale_err_std
    imu_acc_std: float = sensor_noise_config.imu_acc_std
    imu_gyro_std: float = sensor_noise_config.imu_gyro_std
    imu_random_walk: bool = sensor_noise_config.imu_random_walk
    imu_acc_bias_rw_std: float = sensor_noise_config.imu_acc_bias_rw_std
    imu_gyro_bias_rw_std: float = sensor_noise_config.imu_gyro_bias_rw_std
    imu_acc_bias_init_std: float = sensor_noise_config.imu_acc_bias_init_std
    imu_gyro_bias_init_std: float = sensor_noise_config.imu_gyro_bias_init_std
    latency_steps: LatencyConfig = field(
        default_factory=lambda: sensor_noise_config.latency_steps
    )
    dropout_prob: float = sensor_noise_config.dropout_prob

    _acc_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64), init=False, repr=False
    )
    _gyro_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64), init=False, repr=False
    )
    _torque_scale: np.ndarray | None = field(default=None, init=False, repr=False)
    _vel_filter_state: Dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _joint_delay: Deque[Any] = field(default_factory=deque, init=False, repr=False)
    _imu_delay: Deque[Any] = field(default_factory=deque, init=False, repr=False)
    _last_reading: SensorReading | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        dt: float,
        cfg: SensorNoiseConfig = sensor_noise_config,
        rng: np.random.Generator | None = None,
    ) -> "SensorNoise":
        noise = cls(
            rng=rng if rng is not None else np.random.default_rng(cfg.rng_seed),
            dt=float(dt),
            enabled=bool(cfg.enabled),
            joint_pos_std=float(cfg.joint_pos_std),
            joint_pos_quantization_rad=float(cfg.joint_pos_quantization_rad),
            joint_vel_std=float(cfg.joint_vel_std),
            joint_vel_filter=cfg.joint_vel_filter,
            joint_torque_std=float(cfg.joint_torque_std),
            joint_torque_scale_err_std=float(cfg.joint_torque_scale_err_std),
            imu_acc_std=float(cfg.imu_acc_std),
            imu_gyro_std=float(cfg.imu_gyro_std),
            imu_random_walk=bool(cfg.imu_random_walk),
            imu_acc_bias_rw_std=float(cfg.imu_acc_bias_rw_std),
            imu_gyro_bias_rw_std=float(cfg.imu_gyro_bias_rw_std),
            imu_acc_bias_init_std=float(cfg.imu_acc_bias_init_std),
            imu_gyro_bias_init_std=float(cfg.imu_gyro_bias_init_std),
            latency_steps=cfg.latency_steps,
            dropout_prob=float(cfg.dropout_prob),
        )
        noise.reset()
        return noise

    # ── episode lifecycle ────────────────────────────────────────────────────

    def reset(self, n_joints: int = 12) -> None:
        """
        Start a new episode: redraw per-episode errors, clear filters and delays.

        The initial IMU bias and the torque scale error are episode-constant, so
        a network cannot learn to average them away within one episode — which is
        the point of modelling them at all.
        """
        if self.enabled and self.imu_random_walk:
            self._acc_bias = self.rng.normal(0.0, self.imu_acc_bias_init_std, size=3)
            self._gyro_bias = self.rng.normal(0.0, self.imu_gyro_bias_init_std, size=3)
        else:
            self._acc_bias = np.zeros(3, dtype=np.float64)
            self._gyro_bias = np.zeros(3, dtype=np.float64)

        if self.enabled and self.joint_torque_scale_err_std > 0.0:
            self._torque_scale = 1.0 + self.rng.normal(
                0.0, self.joint_torque_scale_err_std, size=n_joints
            )
        else:
            self._torque_scale = np.ones(n_joints, dtype=np.float64)

        self._vel_filter_state = {}
        self._joint_delay = deque()
        self._imu_delay = deque()
        self._last_reading = None

    # ── per-step corruption ──────────────────────────────────────────────────

    def apply(
        self,
        *,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_torque: np.ndarray,
        imu_acc: np.ndarray,
        imu_gyro: np.ndarray,
    ) -> SensorReading:
        """
        Corrupt one control step's observations.

        Returns a :class:`SensorReading`; the realised IMU biases travel with it
        so the recorder can log them as privileged channels.
        """
        q = np.asarray(joint_pos, dtype=np.float64).reshape(-1).copy()
        dq = np.asarray(joint_vel, dtype=np.float64).reshape(-1).copy()
        tau = np.asarray(joint_torque, dtype=np.float64).reshape(-1).copy()
        acc = np.asarray(imu_acc, dtype=np.float64).reshape(3).copy()
        gyro = np.asarray(imu_gyro, dtype=np.float64).reshape(3).copy()

        if not self.enabled:
            return SensorReading(
                joint_pos=q.astype(np.float32),
                joint_vel=dq.astype(np.float32),
                joint_torque=tau.astype(np.float32),
                imu_acc=acc.astype(np.float32),
                imu_gyro=gyro.astype(np.float32),
                imu_acc_bias=np.zeros(3, dtype=np.float32),
                imu_gyro_bias=np.zeros(3, dtype=np.float32),
                stale=False,
            )

        q = self._corrupt_joint_pos(q)
        dq = self._corrupt_joint_vel(dq)
        tau = self._corrupt_torque(tau)
        acc, gyro = self._corrupt_imu(acc, gyro)

        q, dq, tau = self._delay_joint(q, dq, tau)
        acc, gyro = self._delay_imu(acc, gyro)

        reading = SensorReading(
            joint_pos=q.astype(np.float32),
            joint_vel=dq.astype(np.float32),
            joint_torque=tau.astype(np.float32),
            imu_acc=acc.astype(np.float32),
            imu_gyro=gyro.astype(np.float32),
            imu_acc_bias=self._acc_bias.astype(np.float32).copy(),
            imu_gyro_bias=self._gyro_bias.astype(np.float32).copy(),
            stale=False,
        )

        # Packet dropout: hold the previous sample and say so.
        if (
            self._last_reading is not None
            and self.dropout_prob > 0.0
            and self.rng.random() < self.dropout_prob
        ):
            held = self._last_reading
            reading = SensorReading(
                joint_pos=held.joint_pos,
                joint_vel=held.joint_vel,
                joint_torque=held.joint_torque,
                imu_acc=held.imu_acc,
                imu_gyro=held.imu_gyro,
                imu_acc_bias=reading.imu_acc_bias,
                imu_gyro_bias=reading.imu_gyro_bias,
                stale=True,
            )

        self._last_reading = reading
        return reading

    def _corrupt_joint_pos(self, q: np.ndarray) -> np.ndarray:
        if self.joint_pos_std > 0.0:
            q = q + self.rng.normal(0.0, self.joint_pos_std, size=q.shape)
        step = self.joint_pos_quantization_rad
        if step > 0.0:
            q = np.round(q / step) * step
        return q

    def _corrupt_joint_vel(self, dq: np.ndarray) -> np.ndarray:
        if self.joint_vel_std > 0.0:
            dq = dq + self.rng.normal(0.0, self.joint_vel_std, size=dq.shape)
        return self._filter_velocity(dq)

    def _filter_velocity(self, dq: np.ndarray) -> np.ndarray:
        """Low-pass the noisy velocity, as a motor driver would."""
        cfg = self.joint_vel_filter
        if cfg is None or cfg.cutoff_hz <= 0.0:
            return dq

        state = self._vel_filter_state
        if "kind" not in state:
            coefficients = _butter_lowpass(cfg.cutoff_hz, self.dt, cfg.order)
            if coefficients is None:
                # First-order IIR fallback, matched to the same cutoff.
                alpha = 1.0 - np.exp(-2.0 * np.pi * cfg.cutoff_hz * self.dt)
                state.update(kind="iir", alpha=float(alpha), y=dq.copy())
                return state["y"]
            b, a = coefficients
            state.update(
                kind="butter",
                b=np.asarray(b, dtype=np.float64),
                a=np.asarray(a, dtype=np.float64),
                # One delay line per joint, initialised to the first sample so the
                # filter does not ramp up from zero at episode start.
                zi=[np.zeros(max(len(a), len(b)) - 1) for _ in range(dq.size)],
                primed=False,
            )

        if state["kind"] == "iir":
            state["y"] = state["y"] + state["alpha"] * (dq - state["y"])
            return state["y"].copy()

        from scipy.signal import lfilter, lfilter_zi  # noqa: PLC0415

        b, a = state["b"], state["a"]
        if not state["primed"]:
            base = lfilter_zi(b, a)
            state["zi"] = [base * float(value) for value in dq]
            state["primed"] = True
        out = np.empty_like(dq)
        for i, value in enumerate(dq):
            filtered, state["zi"][i] = lfilter(b, a, [value], zi=state["zi"][i])
            out[i] = filtered[0]
        return out

    def _corrupt_torque(self, tau: np.ndarray) -> np.ndarray:
        scale = self._torque_scale
        if scale is None or scale.size != tau.size:
            scale = np.ones(tau.size, dtype=np.float64)
            self._torque_scale = scale
        tau = tau * scale
        if self.joint_torque_std > 0.0:
            tau = tau + self.rng.normal(0.0, self.joint_torque_std, size=tau.shape)
        return tau

    def _corrupt_imu(self, acc: np.ndarray, gyro: np.ndarray):
        if self.imu_random_walk:
            sqrt_dt = float(np.sqrt(self.dt))
            if self.imu_acc_bias_rw_std > 0.0:
                self._acc_bias = self._acc_bias + self.rng.normal(
                    0.0, self.imu_acc_bias_rw_std * sqrt_dt, size=3
                )
            if self.imu_gyro_bias_rw_std > 0.0:
                self._gyro_bias = self._gyro_bias + self.rng.normal(
                    0.0, self.imu_gyro_bias_rw_std * sqrt_dt, size=3
                )
        if self.imu_acc_std > 0.0:
            acc = acc + self.rng.normal(0.0, self.imu_acc_std, size=3)
        if self.imu_gyro_std > 0.0:
            gyro = gyro + self.rng.normal(0.0, self.imu_gyro_std, size=3)
        return acc + self._acc_bias, gyro + self._gyro_bias

    def _delay_joint(self, q, dq, tau):
        steps = int(getattr(self.latency_steps, "joint", 0) or 0)
        if steps <= 0:
            return q, dq, tau
        self._joint_delay.append((q, dq, tau))
        if len(self._joint_delay) <= steps:
            return self._joint_delay[0]
        return self._joint_delay.popleft()

    def _delay_imu(self, acc, gyro):
        steps = int(getattr(self.latency_steps, "imu", 0) or 0)
        if steps <= 0:
            return acc, gyro
        self._imu_delay.append((acc, gyro))
        if len(self._imu_delay) <= steps:
            return self._imu_delay[0]
        return self._imu_delay.popleft()

    # ── reporting ────────────────────────────────────────────────────────────

    def to_metadata(self) -> Dict[str, Any]:
        """The ``sensor_noise`` block written into ``run_metadata.json``."""
        cfg = self.joint_vel_filter
        return {
            "enabled": bool(self.enabled),
            "dt": float(self.dt),
            "joint_pos_std": float(self.joint_pos_std),
            "joint_pos_quantization_rad": float(self.joint_pos_quantization_rad),
            "joint_vel_std": float(self.joint_vel_std),
            "joint_vel_filter": {
                "type": getattr(cfg, "type", "none"),
                "cutoff_hz": float(getattr(cfg, "cutoff_hz", 0.0)),
                "order": int(getattr(cfg, "order", 0)),
            },
            "joint_torque_std": float(self.joint_torque_std),
            "joint_torque_scale_err_std": float(self.joint_torque_scale_err_std),
            "imu_acc_std": float(self.imu_acc_std),
            "imu_gyro_std": float(self.imu_gyro_std),
            "imu_random_walk": bool(self.imu_random_walk),
            "imu_acc_bias_rw_std": float(self.imu_acc_bias_rw_std),
            "imu_gyro_bias_rw_std": float(self.imu_gyro_bias_rw_std),
            "imu_acc_bias_init_std": float(self.imu_acc_bias_init_std),
            "imu_gyro_bias_init_std": float(self.imu_gyro_bias_init_std),
            "latency_steps": {
                "joint": int(getattr(self.latency_steps, "joint", 0)),
                "imu": int(getattr(self.latency_steps, "imu", 0)),
            },
            "dropout_prob": float(self.dropout_prob),
        }
