"""Model training and PIT computation.

Approach:
    1. Train a GradientBoostingRegressor on the training split.
    2. For each monitoring sample, compute PIT.

This gives a clean, standard distributional assumption whose quality is
itself part of what PITMonitor is designed to detect changes in.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cal_frac: float,
    seed: int,
) -> tuple[GradientBoostingRegressor, float]:
    """Train a GBR and estimate calibration residual std.

    Returns
    -------
    model : GradientBoostingRegressor
        Fitted model.
    sigma_hat : float
        Standard deviation of calibration-set residuals.
    """
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=cal_frac,
        random_state=seed,
    )
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X_fit, y_fit)

    residuals = y_cal - model.predict(X_cal)
    sigma_hat = float(np.std(residuals))
    if sigma_hat < 1e-12:
        sigma_hat = 1.0  # safety fallback
    return model, sigma_hat


def compute_pits(
    model: GradientBoostingRegressor,
    sigma_hat: float,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute PIT values F̂(y | x) = Φ((y − ŷ) / σ̂).

    Returns array of shape (n,) with values in (0, 1).
    """
    y_hat = model.predict(X)
    z = (y - y_hat) / sigma_hat
    pits = stats.norm.cdf(z)
    # Clip away from exact 0/1 for numerical safety.
    return np.clip(pits, 1e-8, 1 - 1e-8)


def compute_residuals(
    model: GradientBoostingRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute raw residuals (y − ŷ)."""
    return y - model.predict(X)


###################

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class ProbabilisticMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        mu = self.mean_head(features)
        log_var = self.logvar_head(features)
        return mu, log_var


def train_model(X, y, epochs=100, lr=1e-3):
    """Trains a Probabilistic MLP using Gaussian NLL loss."""
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    model = ProbabilisticMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.GaussianNLLLoss(eps=1e-6)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        mu, log_var = model(X_tensor)
        loss = criterion(mu, y_tensor, torch.exp(log_var))
        loss.backward()
        optimizer.step()
    return model


def compute_pits(model, X, y):
    """Calculates PIT values."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        mu, log_var = model(X_t)
        sigma = torch.exp(0.5 * log_var)

        dist = Normal(0, 1)
        z_scores = (y_t - mu) / sigma
        pits = dist.cdf(z_scores).numpy().flatten()
    return pits


def compute_residuals(model, X, y):
    """Computes residuals as r = y - mu for detector compatibility."""
    model.eval()
    with torch.no_grad():
        mu, _ = model(torch.tensor(X.values, dtype=torch.float32))
        return (y.values.reshape(-1, 1) - mu.numpy()).flatten()
