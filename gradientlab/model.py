"""Polynomial regression trained with manual gradient descent.

The model is a polynomial hypothesis::

    y_hat = theta_0 + theta_1 * x + theta_2 * x^2 + ... + theta_n * x^n

fitted by minimising the mean squared error::

    J(theta) = (1 / m) * sum_i (y_hat_i - y_i) ** 2

The gradient of the loss with respect to each parameter is::

    dJ/dtheta_j = (2 / m) * sum_i (y_hat_i - y_i) * x_i ** j

and parameters are updated with plain batch gradient descent::

    theta_j := theta_j - alpha * dJ/dtheta_j

Everything below is written with vectorised NumPy operations; no autograd or
scikit-learn is used for the optimisation.
"""

from __future__ import annotations

import numpy as np

from .data import standardize


def design_matrix(x, degree):
    """Build the polynomial design matrix ``[x^0, x^1, ..., x^degree]``.

    Parameters
    ----------
    x : array-like, shape (m,)
        Input feature values.
    degree : int
        Highest polynomial power (>= 1).

    Returns
    -------
    numpy.ndarray, shape (m, degree + 1)
        Column ``j`` holds ``x ** j``.
    """
    x = np.asarray(x, dtype=float).ravel()
    # increasing=True -> columns run x^0, x^1, ... x^degree (Vandermonde matrix).
    return np.vander(x, N=degree + 1, increasing=True)


def predict(design, theta):
    """Return predictions ``y_hat = design @ theta`` for a prebuilt matrix."""
    return np.asarray(design, dtype=float) @ np.asarray(theta, dtype=float)


def mse(y_pred, y_true):
    """Mean squared error ``(1/m) * sum (y_hat - y)^2``."""
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    # A diverged model produces huge predictions; overflowing to inf is the
    # correct answer, so silence the redundant FP warning.
    with np.errstate(over="ignore", invalid="ignore"):
        error = y_pred - y_true
        return float(np.mean(error ** 2))


def r_squared(y_pred, y_true):
    """Coefficient of determination ``R^2 = 1 - SS_res / SS_tot``.

    Returns ``nan`` when the targets have zero variance (R^2 is undefined).
    """
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    with np.errstate(over="ignore", invalid="ignore"):
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def gradient(design, theta, y_true):
    """Gradient of the MSE loss: ``(2/m) * design.T @ (y_hat - y)``."""
    design = np.asarray(design, dtype=float)
    y_true = np.asarray(y_true, dtype=float).ravel()
    m = design.shape[0]
    error = design @ np.asarray(theta, dtype=float) - y_true
    return (2.0 / m) * (design.T @ error)


class PolynomialRegression:
    """Polynomial regression fitted with batch gradient descent.

    Parameters
    ----------
    degree : int
        Degree of the polynomial (>= 1).
    learning_rate : float
        Step size ``alpha`` used in the parameter update.
    epochs : int
        Maximum number of gradient-descent iterations.
    tolerance : float
        Stop early once ``abs(previous_loss - current_loss) < tolerance``.
    normalize : bool
        Standardise ``x`` (zero mean, unit variance) before building the
        polynomial features. This dramatically improves conditioning for
        higher-degree fits.
    """

    def __init__(self, degree=2, learning_rate=0.01, epochs=10000,
                 tolerance=1e-9, normalize=False):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        self.degree = int(degree)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.tolerance = float(tolerance)
        self.normalize = bool(normalize)

        # Learned state (populated by ``fit``).
        self.theta = None
        self.loss_history = []
        self.converged = False
        self.diverged = False
        self.n_iterations = 0
        self._x_mean = 0.0
        self._x_std = 1.0

    # -- feature handling ---------------------------------------------------
    def _transform(self, x):
        """Apply the same x-scaling used at fit time, then build features."""
        x = np.asarray(x, dtype=float).ravel()
        if self.normalize:
            x = (x - self._x_mean) / self._x_std
        return design_matrix(x, self.degree)

    # -- training -----------------------------------------------------------
    def fit(self, x, y):
        """Fit the model to ``(x, y)`` with gradient descent."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit an empty dataset")

        if self.normalize:
            _, self._x_mean, self._x_std = standardize(x)

        design = self._transform(x)
        m = design.shape[0]

        self.theta = np.zeros(design.shape[1])
        self.loss_history = []
        self.converged = False
        self.diverged = False
        prev_loss = None

        # A diverging run overflows to inf/nan; we detect that explicitly below,
        # so silence NumPy's overflow chatter instead of spamming the console.
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(self.epochs):
                # One matrix-vector product serves both loss and gradient.
                error = design @ self.theta - y
                loss = float(np.mean(error ** 2))
                self.loss_history.append(loss)

                if not np.isfinite(loss):
                    self.diverged = True
                    break
                if prev_loss is not None and abs(prev_loss - loss) < self.tolerance:
                    self.converged = True
                    break
                prev_loss = loss

                # theta := theta - alpha * (2/m) * X^T (y_hat - y)
                grad = (2.0 / m) * (design.T @ error)
                self.theta = self.theta - self.learning_rate * grad

        self.n_iterations = len(self.loss_history)
        return self

    # -- inference ----------------------------------------------------------
    def predict(self, x):
        """Predict targets for new ``x`` values."""
        if self.theta is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self._transform(x) @ self.theta

    def loss(self, x, y):
        """Mean squared error of the fitted model on ``(x, y)``."""
        return mse(self.predict(x), y)

    def score(self, x, y):
        """R^2 score of the fitted model on ``(x, y)``."""
        return r_squared(self.predict(x), y)

    # -- reporting ----------------------------------------------------------
    def coefficients(self):
        """Polynomial coefficients in terms of the *raw* ``x`` (increasing order).

        When ``normalize`` is on, ``theta`` is expressed in the standardized
        variable ``z = (x - mean) / std``. Here we substitute that back in and
        collect the powers of ``x`` so the reported equation is directly usable.
        """
        if self.theta is None:
            raise RuntimeError("model is not fitted; call fit() first")
        if not self.normalize:
            return self.theta.copy()

        # Compose sum_j theta_j * z^j with z = (x - mean) / std.
        from numpy.polynomial import Polynomial

        z = Polynomial([-self._x_mean / self._x_std, 1.0 / self._x_std])
        poly = Polynomial([0.0])
        z_power = Polynomial([1.0])
        for coef in self.theta:
            poly = poly + coef * z_power
            z_power = z_power * z

        coeffs = np.zeros(self.degree + 1)
        coeffs[: poly.coef.shape[0]] = poly.coef
        return coeffs

    def equation(self, precision=4):
        """Human-readable ``y_hat = ...`` string in terms of raw ``x``."""
        coeffs = self.coefficients()
        parts = []
        for power, c in enumerate(coeffs):
            value = round(float(c), precision)
            if power == 0:
                parts.append(f"{value:g}")
                continue
            token = "x" if power == 1 else f"x^{power}"
            sign = "-" if value < 0 else "+"
            parts.append(f" {sign} {abs(value):g}*{token}")
        return "y_hat = " + "".join(parts)
