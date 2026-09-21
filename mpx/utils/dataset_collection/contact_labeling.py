"""
Contact labelling: per-substep force reduction, hysteresis, minimum dwell.

Why this exists
---------------
Sampling the contact solver at one instant every control step produced 31.6% of
contact runs lasting exactly one 20 ms step, and 6.2 transitions per foot per
second against an expected ~2.5 for the commanded gait. Those dropouts are
physically real — total vertical GRF fell 177 N → 106 N and the base
accelerated downward at those frames — but at 50 Hz they are unobservable from
proprioception, so as labels they are pure noise.

Three stages fix it, in this order:

1. **Reduce at sim rate.** The force is accumulated on every sim substep between
   two logged frames, and a foot counts as loaded for the frame when the majority
   of its substeps exceed the ON threshold. A single-substep bounce no longer
   flips the frame.
2. **Schmitt trigger.** Touchdown needs ON_THRESHOLD_N, release needs the force
   to fall below the lower OFF_THRESHOLD_N. A force hovering near one threshold
   cannot oscillate.
3. **Minimum dwell.** A state must hold for MIN_DWELL_STEPS before it may change
   again, which bounds the transition rate no matter what the force does.

Every stage is **causal** — it uses only the current and past samples — so the
labels remain reproducible online and no future information leaks backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

N_FEET = 4


@dataclass
class ContactLabelConfig:
    """Thresholds and dwell for :class:`ContactDebouncer`."""

    # Touchdown threshold [N] — roughly 8% of the Go2's 176 N body weight.
    on_threshold_n: float = 15.0

    # Release threshold [N] — the v3 single threshold, kept as the lower rail.
    off_threshold_n: float = 5.0

    # Minimum steps a state must hold before it may change again (3 = 60 ms @ 50 Hz).
    min_dwell_steps: int = 3

    # Fraction of a control interval's substeps that must exceed ``on_threshold_n``
    # for the raw label to read "loaded".
    substep_majority: float = 0.5

    def __post_init__(self) -> None:
        if self.off_threshold_n > self.on_threshold_n:
            raise ValueError(
                f"off_threshold_n ({self.off_threshold_n}) must not exceed "
                f"on_threshold_n ({self.on_threshold_n}) — hysteresis needs a "
                f"lower release rail than trigger rail"
            )
        if self.min_dwell_steps < 1:
            raise ValueError("min_dwell_steps must be >= 1")
        if not 0.0 < self.substep_majority <= 1.0:
            raise ValueError("substep_majority must be in (0, 1]")

    def to_metadata(self, substeps_per_control: int, body_weight_n: float) -> Dict[str, Any]:
        """The ``contact_labeling`` block written into ``run_metadata.json``."""
        return {
            "method": "substep_majority + schmitt + min_dwell",
            "on_threshold_n": float(self.on_threshold_n),
            "off_threshold_n": float(self.off_threshold_n),
            "min_dwell_steps": int(self.min_dwell_steps),
            "substep_majority": float(self.substep_majority),
            "substeps_per_control": int(substeps_per_control),
            "body_weight_n": float(body_weight_n),
        }


@dataclass
class SubstepForceAccumulator:
    """
    Collects the per-foot GRF **vector** across the substeps of one control interval.

    The recorder pushes one sample per sim step and reduces once per control
    step, so both the label and the regression target see the whole interval
    instead of the single instant the control step happens to land on.

    Vectors, not magnitudes: the contact label only needs ``|f|``, but the GRF
    regression target needs the direction too, and averaging magnitudes after
    the fact cannot recover it (DATASET_FIX_TASKS_R3, Task R3-2).
    """

    config: ContactLabelConfig = field(default_factory=ContactLabelConfig)
    _samples: List[np.ndarray] = field(default_factory=list)

    def push(self, grf_world: np.ndarray) -> None:
        """
        Record one substep's per-foot GRF.

        Accepts the ``(4, 3)`` world-frame force vectors, or a ``(4,)`` vector of
        magnitudes for callers that only have those (the magnitudes are then
        placed on +Z, which is correct on flat ground and is only ever used for
        the scalar statistics).
        """
        values = np.asarray(grf_world, dtype=np.float64)
        if values.ndim == 1:
            vectors = np.zeros((N_FEET, 3), dtype=np.float64)
            vectors[:, 2] = values.reshape(N_FEET)
        else:
            vectors = values.reshape(N_FEET, 3).copy()
        self._samples.append(vectors)

    def clear(self) -> None:
        self._samples.clear()

    @property
    def n_substeps(self) -> int:
        return len(self._samples)

    def reduce(self) -> Dict[str, np.ndarray]:
        """
        Collapse the buffered substeps into one control-step row and clear.

        Returns
        -------
        contact_raw : (4,) uint8
            Majority threshold over the interval's substeps, no hysteresis.
        mean : (4,) float
            Mean per-foot force MAGNITUDE over the interval. This is what the
            debouncer sees, and it is the mean of ``|f|``, not ``|mean f|``: a
            foot that is loaded throughout must not read low because the contact
            normal swung during the interval.
        max : (4,) float
            Largest per-foot magnitude within the interval (touchdown transient).
        mean_vector : (4, 3) float
            Mean per-foot force VECTOR, world frame. The regression target
            (R3-2); it keeps the direction that ``mean`` throws away.
        """
        if not self._samples:
            zeros = np.zeros(N_FEET, dtype=np.float64)
            return {
                "contact_raw": np.zeros(N_FEET, dtype=np.uint8),
                "mean": zeros,
                "max": zeros.copy(),
                "mean_vector": np.zeros((N_FEET, 3), dtype=np.float64),
            }

        stacked = np.stack(self._samples, axis=0)          # (substeps, 4, 3)
        magnitude = np.linalg.norm(stacked, axis=2)        # (substeps, 4)
        loaded_fraction = (magnitude >= self.config.on_threshold_n).mean(axis=0)
        result = {
            "contact_raw": (
                loaded_fraction >= self.config.substep_majority
            ).astype(np.uint8),
            "mean": magnitude.mean(axis=0),
            "max": magnitude.max(axis=0),
            "mean_vector": stacked.mean(axis=0),
        }
        self.clear()
        return result


class ContactDebouncer:
    """
    Hysteresis plus minimum dwell over the per-foot force. Causal.

    Feed it the reduced per-interval force (the accumulator's ``mean``) once per
    control step. It returns the debounced 4-bit contact vector that becomes the
    primary training label.
    """

    def __init__(self, config: ContactLabelConfig | None = None, n_feet: int = N_FEET):
        self.config = config if config is not None else ContactLabelConfig()
        self.n_feet = int(n_feet)
        self.reset()

    def reset(self, initial_contact: int = 1) -> None:
        """
        Start a fresh episode.

        Feet start in stance and already past the dwell window, so the first real
        liftoff is not delayed by an artefact of initialisation.
        """
        self.state = np.full(self.n_feet, int(initial_contact), dtype=np.uint8)
        self.dwell = np.full(self.n_feet, self.config.min_dwell_steps, dtype=np.int32)

    def step(self, force_n: np.ndarray) -> np.ndarray:
        """Advance one control step with per-foot force ``(4,)`` [N]."""
        force = np.asarray(force_n, dtype=np.float64).reshape(self.n_feet)
        on, off = self.config.on_threshold_n, self.config.off_threshold_n
        min_dwell = self.config.min_dwell_steps

        for i in range(self.n_feet):
            want = self.state[i]
            if self.state[i] == 0 and force[i] >= on:
                want = 1
            elif self.state[i] == 1 and force[i] < off:
                want = 0

            if want != self.state[i] and self.dwell[i] >= min_dwell:
                self.state[i] = want
                self.dwell[i] = 0
            else:
                self.dwell[i] += 1

        return self.state.copy()


def debounce_sequence(
    force_n: np.ndarray,
    config: ContactLabelConfig | None = None,
) -> np.ndarray:
    """
    Run :class:`ContactDebouncer` over a whole ``(T, 4)`` force trace.

    Offline convenience for re-labelling stored episodes and for tests; the
    recorder uses the streaming class.
    """
    force = np.asarray(force_n, dtype=np.float64).reshape(-1, N_FEET)
    debouncer = ContactDebouncer(config)
    return np.stack([debouncer.step(row) for row in force], axis=0).astype(np.uint8)


def contact_run_statistics(contact: np.ndarray, control_hz: float) -> Dict[str, float]:
    """
    Chatter diagnostics for a ``(T, 4)`` contact trace.

    ``fraction_runs_length_1`` and ``transitions_per_foot_per_s`` are the two
    numbers the v4 acceptance check gates on (< 0.02 and < 3.5 respectively;
    v3 measured 0.316 and 6.23).
    """
    bits = np.asarray(contact, dtype=np.int64).reshape(-1, N_FEET)
    n_steps = bits.shape[0]
    if n_steps == 0:
        return {
            "runs": 0,
            "steps": 0,
            "transitions": 0,
            "fraction_runs_length_1": 0.0,
            "transitions_per_foot_per_s": 0.0,
            "median_stance_steps": 0.0,
            "median_swing_steps": 0.0,
        }

    run_lengths: List[int] = []
    stance_lengths: List[int] = []
    swing_lengths: List[int] = []
    transitions = 0
    for foot in range(N_FEET):
        series = bits[:, foot]
        changes = np.flatnonzero(np.diff(series)) + 1
        transitions += len(changes)
        for segment in np.split(series, changes):
            run_lengths.append(len(segment))
            (stance_lengths if segment[0] == 1 else swing_lengths).append(len(segment))

    lengths = np.asarray(run_lengths)
    return {
        "runs": int(len(lengths)),
        "steps": int(n_steps),
        "transitions": int(transitions),
        "fraction_runs_length_1": float((lengths == 1).mean()),
        "transitions_per_foot_per_s": float(
            transitions / N_FEET / (n_steps / float(control_hz))
        ),
        "median_stance_steps": float(np.median(stance_lengths)) if stance_lengths else 0.0,
        "median_swing_steps": float(np.median(swing_lengths)) if swing_lengths else 0.0,
    }
