"""Data generation from River's FriedmanDrift benchmark."""

import numpy as np
from river.datasets.synth import FriedmanDrift

from config import Config


def generate_stream(
    cfg,
    drift_type,
    transition_window,
    seed,
):
    """Materialise a FriedmanDrift stream into (X, y) numpy arrays.

    Parameters
    ----------
    cfg : Config
        Experiment configuration (defines stream length and drift position).
    drift_type : str
        One of 'gra', 'gsg', 'lea'.
    transition_window : int
        Gradual-drift window width (0 = abrupt).
    seed : int
        Random seed for the stream generator.

    Returns
    -------
    X : ndarray of shape (n_total, 10)
    y : ndarray of shape (n_total,)
    """
    positions = cfg.positions_for(drift_type)
    dataset = FriedmanDrift(
        drift_type=drift_type,
        position=positions,
        transition_window=transition_window,
        seed=seed,
    )

    n = cfg.n_total
    X = np.empty((n, 10), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)

    for i, (x_dict, yi) in enumerate(dataset.take(n)):
        X[i] = list(x_dict.values())
        y[i] = yi

    return X, y
