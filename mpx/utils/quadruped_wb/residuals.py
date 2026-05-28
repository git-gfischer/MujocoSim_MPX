"""Residual masks for whole-body QP foot tracking and GRF matching.

Maps are driven by ``swing_tracking`` (from config):

- Locomotion: ``swing_tracking=True`` — feet and GRF residuals are always stacked (scheduler
  in the reference still sets contact/friction penalties).
- Balance: ``swing_tracking=False`` — residuals for feet and GRF on nominal swing legs
  are gated off so stance legs dominate the QP (fixed contact masks from reference).
"""

from __future__ import annotations

import jax.numpy as jnp


def quadruped_wb_residual_masks(swing_tracking: bool, contact, n_contact: int):
    """Length ``3 * n_contact`` masks for stacked foot XYZ and GRF components."""
    if swing_tracking:
        cm = jnp.ones(3 * n_contact, dtype=jnp.float32)
        return cm, cm
    mask = jnp.repeat(contact, 3).astype(jnp.float32)
    return mask, mask
