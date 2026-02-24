"""Model training and PIT computation.

Approach:
    1. Train a ProbabilisticMLP on the training split.
    2. For each monitoring sample, compute PIT.

This gives a clean, standard distributional assumption whose quality is
itself part of what PITMonitor is designed to detect changes in.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


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


def train_model(X_train, y_train, epochs=100, lr=1e-3):
    """Trains a Probabilistic MLP using Gaussian NLL loss."""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = ProbabilisticMLP(input_dim=X_train.shape[1])
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
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        mu, log_var = model(X_t)
        sigma = torch.exp(0.5 * log_var)

        dist = Normal(0, 1)
        z_scores = (y_t - mu) / sigma
        pits = dist.cdf(z_scores).numpy().flatten()
    return np.clip(pits, 1e-8, 1 - 1e-8)  # Clip for numerical safety.


def compute_residuals(model, X, y):
    """Computes residuals."""
    model.eval()
    with torch.no_grad():
        mu, _ = model(torch.tensor(X, dtype=torch.float32))
        return (y.reshape(-1, 1) - mu.numpy()).flatten()
