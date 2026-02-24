"""Data generation from River's FriedmanDrift benchmark.

FriedmanDrift streams simulate a regression problem with 10 input features
(``x0``–``x9``) and a continuous target.  Drift is introduced at a
configurable position and can be abrupt or gradual.  Only features ``x0``–``x4``
and ``x9`` appear in the true function; the remaining features are noise.

Supported drift types
---------------------
``'gra'``  – Global Recurring Abrupt: all relevant features change simultaneously.
``'gsg'``  – Global Slow Gradual: change spreads linearly over a transition window.
``'lea'``  – Local Expanding Abrupt: drift starts on a subset of features and
             expands to include more.
"""

from __future__ import annotations

import numpy as np
from river.datasets.synth import FriedmanDrift

from config import Config


def generate_stream(
    cfg: Config,
    drift_type: str,
    transition_window: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a FriedmanDrift stream into (X, y) numpy arrays.

    The stream is consumed up to ``cfg.n_total`` samples.  Samples 0 to
    ``cfg.n_train - 1`` are pre-drift training data; samples ``cfg.n_train``
    onwards form the monitoring window.

    Parameters
    ----------
    cfg : Config
        Experiment configuration (determines stream length and drift position).
    drift_type : str
        One of ``'gra'``, ``'gsg'``, ``'lea'``.
    transition_window : int
        Width of the gradual drift transition in samples (0 = abrupt).
    seed : int
        Random seed for the stream generator; controls both feature noise and
        the label noise.

    Returns
    -------
    X : ndarray, shape (cfg.n_total, 10)
        Feature matrix; columns correspond to ``x0`` … ``x9``.
    y : ndarray, shape (cfg.n_total,)
        Regression targets.
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
