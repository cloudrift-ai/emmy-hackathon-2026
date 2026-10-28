"""Shared result-normalization helpers for local and hosted validation."""

from __future__ import annotations

import numpy as np


def result_array(values: list, reference: np.ndarray) -> np.ndarray:
    """Shape runner output without discarding the reference result's dtype."""
    return np.asarray(values, dtype=reference.dtype).reshape(reference.shape)
