"""Model training and PIT computation.

Approach:
    1. Train a ProbabilisticMLP on the training split.
    2. For each monitoring sample, compute PIT values via the predicted
       Gaussian CDF (standard normal evaluated at the z-score).

The model outputs a predictive mean and log-variance for each input.
Under the null (stable calibration), PITs should be approximately U[0,1],
which is what PITMonitor is designed to test.

Normalization:
    Inputs and targets are standardized before training and the scaler
    statistics are bundled with the model so that inference is consistent
    without re-fitting.

Persistence:
    Use ``save_bundle`` / ``load_bundle`` to cache trained weights and avoid
    retraining on every experiment run.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset


# ─── Scaler ─────────────────────────────────────────────────────────


@dataclass
class StandardScaler:
    """Mean/std scaler fitted on training data.

    Parameters
    ----------
    mean_ : ndarray
        Per-feature (or scalar) mean from the training set.
    std_ : ndarray
        Per-feature (or scalar) standard deviation from the training set.
        Any component smaller than 1e-8 is clamped to 1 to avoid division
        by zero on constant features.
    """

    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, arr: np.ndarray) -> "StandardScaler":
        """Fit a new scaler to *arr* (shape (N,) or (N, D))."""
        mean_ = arr.mean(axis=0)
        std_ = arr.std(axis=0)
        std_ = np.where(std_ < 1e-8, 1.0, std_)
        return cls(mean_, std_)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """Return standardized copy of *arr* using the fitted statistics."""
        return (arr - self.mean_) / self.std_

    def inverse_transform_mean(self, arr: np.ndarray) -> np.ndarray:
        """Invert standardization on a mean prediction."""
        return arr * self.std_ + self.mean_

    def inverse_transform_std(self, std: np.ndarray) -> np.ndarray:
        """Scale a standard deviation back to original units."""
        return std * self.std_


# ─── Architecture ────────────────────────────────────────────────────


class ProbabilisticMLP(nn.Module):
    """Feed-forward network that outputs a Gaussian predictive distribution.

    The network shares a multi-layer backbone with SiLU activations and
    branches into separate linear heads for the predictive mean and
    log-variance.  Inputs and targets must be standardized before passing to
    ``forward``; the matching ``StandardScaler`` objects are stored on the
    ``ModelBundle`` wrapper returned by ``train_model``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the (normalized) input features.
    hidden_dim : int, default=128
        Width of each hidden layer.
    n_layers : int, default=3
        Number of hidden layers (minimum 1).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict Gaussian parameters for standardized inputs.

        Parameters
        ----------
        x : Tensor, shape (N, input_dim)
            Standardized input features.

        Returns
        -------
        mu : Tensor, shape (N, 1)
            Predicted mean in standardized target space.
        log_var : Tensor, shape (N, 1)
            Predicted log-variance in standardized target space.
        """
        features = self.backbone(x)
        mu = self.mean_head(features)
        log_var = self.logvar_head(features)
        return mu, log_var


# ─── Bundle: model + scalers ─────────────────────────────────────────


@dataclass
class ModelBundle:
    """Trained model together with its normalization statistics.

    Attributes
    ----------
    model : ProbabilisticMLP
        The trained network (weights set to eval mode after training).
    x_scaler : StandardScaler
        Fitted on training features; applied before every forward pass.
    y_scaler : StandardScaler
        Fitted on training targets; used to convert predictions back to
        the original scale and to normalize targets at inference time.
    """

    model: ProbabilisticMLP
    x_scaler: StandardScaler
    y_scaler: StandardScaler


# ─── Training ────────────────────────────────────────────────────────


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 500,
    lr: float = 3e-4,
    batch_size: int = 256,
    hidden_dim: int = 128,
    n_layers: int = 3,
    seed: int = 42,
) -> ModelBundle:
    """Train a ProbabilisticMLP and return a ``ModelBundle``.

    Steps:
    1. Fit ``StandardScaler`` on ``X_train`` and ``y_train``.
    2. Train with mini-batch Gaussian NLL loss and a cosine LR schedule.
    3. Return the bundle with the model in evaluation mode.

    Parameters
    ----------
    X_train : ndarray, shape (N, D)
        Raw (unnormalized) training features.
    y_train : ndarray, shape (N,)
        Raw training targets.
    epochs : int, default=200
        Number of full passes over the training data.
    lr : float, default=3e-4
        Initial Adam learning rate.
    batch_size : int, default=256
        Mini-batch size.
    hidden_dim : int, default=128
        Width of each hidden layer.
    n_layers : int, default=3
        Number of hidden layers.
    seed : int, default=42
        Torch manual seed for weight initialization reproducibility.

    Returns
    -------
    ModelBundle
        Trained model with ``x_scaler`` and ``y_scaler`` attached.
    """
    torch.manual_seed(seed)

    # Fit and apply scalers
    x_scaler = StandardScaler.fit(X_train)
    y_scaler = StandardScaler.fit(y_train.reshape(-1))

    X_norm = x_scaler.transform(X_train).astype(np.float32)
    y_norm = y_scaler.transform(y_train.reshape(-1)).astype(np.float32)

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y_norm).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ProbabilisticMLP(
        input_dim=X_train.shape[1], hidden_dim=hidden_dim, n_layers=n_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.GaussianNLLLoss(eps=1e-6)

    model.train()
    for _ in range(epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            mu, log_var = model(X_b)
            var = torch.exp(log_var).clamp(min=1e-6)
            loss = criterion(mu, y_b, var)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return ModelBundle(model=model, x_scaler=x_scaler, y_scaler=y_scaler)


# ─── Inference ───────────────────────────────────────────────────────


def compute_pits(bundle: ModelBundle, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute probability integral transform (PIT) values.

    For each observation (x_i, y_i), the PIT is:
        F_i(y_i) = Phi((y_i - mu_i) / sigma_i)
    where Phi is the standard normal CDF, mu_i and sigma_i are the model's
    predictive mean and std in original units.  Under perfect calibration
    PITs are i.i.d. U[0, 1].

    Parameters
    ----------
    bundle : ModelBundle
        Trained model with normalization scalers.
    X : ndarray, shape (N, D)
        Raw features.
    y : ndarray, shape (N,)
        Raw observed targets.

    Returns
    -------
    pits : ndarray, shape (N,)
        PIT values, clipped to (1e-8, 1 - 1e-8) for numerical safety.
    """
    model = bundle.model
    model.eval()
    with torch.no_grad():
        X_norm = bundle.x_scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_norm)
        mu_norm, log_var = model(X_t)
        sigma_norm = torch.exp(0.5 * log_var).clamp(min=1e-6)

        # Compute z-score in normalized space (avoids unit mismatches)
        y_norm = bundle.y_scaler.transform(y.reshape(-1)).astype(np.float32)
        y_t = torch.from_numpy(y_norm).unsqueeze(1)

        z = (y_t - mu_norm) / sigma_norm
        pits = Normal(0.0, 1.0).cdf(z).numpy().flatten()

    return np.clip(pits, 1e-8, 1 - 1e-8)


def compute_predictions(
    bundle: ModelBundle, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictive mean and std in the original (un-normalized) scale.

    Parameters
    ----------
    bundle : ModelBundle
    X : ndarray, shape (N, D)

    Returns
    -------
    mu : ndarray, shape (N,)
        Predictive mean in original target units.
    sigma : ndarray, shape (N,)
        Predictive standard deviation in original target units (always > 0).
    """
    model = bundle.model
    model.eval()
    with torch.no_grad():
        X_norm = bundle.x_scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_norm)
        mu_norm, log_var = model(X_t)
        sigma_norm = torch.exp(0.5 * log_var)

    mu = bundle.y_scaler.inverse_transform_mean(mu_norm.numpy().flatten())
    sigma = bundle.y_scaler.inverse_transform_std(sigma_norm.numpy().flatten())
    return mu, np.abs(sigma)


def compute_residuals(bundle: ModelBundle, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute raw residuals (y - mu_hat) in original units.

    Parameters
    ----------
    bundle : ModelBundle
    X : ndarray, shape (N, D)
    y : ndarray, shape (N,)

    Returns
    -------
    residuals : ndarray, shape (N,)
    """
    mu, _ = compute_predictions(bundle, X)
    return y.reshape(-1) - mu


# ─── Persistence ─────────────────────────────────────────────────────


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    """Persist the full ``ModelBundle`` (weights + scalers) to *path*.

    Uses ``pickle`` so that both the torch model state and the numpy scaler
    statistics are stored in a single file.

    Parameters
    ----------
    bundle : ModelBundle
    path : Path
        Destination file; parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path: Path) -> ModelBundle:
    """Load a ``ModelBundle`` previously saved with ``save_bundle``.

    The loaded model is set to evaluation mode before returning.

    Parameters
    ----------
    path : Path

    Returns
    -------
    ModelBundle
    """
    path = Path(path)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    bundle.model.eval()
    return bundle
