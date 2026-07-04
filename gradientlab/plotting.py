"""Plotting helpers for GradientLab.

matplotlib is imported lazily inside the function so that the core package
(training, prediction, tests) has no hard dependency on it.
"""

from __future__ import annotations

import numpy as np


def plot_results(model, x, y, show=True, save_path=None):
    """Plot the data with the fitted curve, plus the training loss curve.

    Parameters
    ----------
    model : PolynomialRegression
        A fitted model (uses ``model.predict`` and ``model.loss_history``).
    x, y : array-like
        The training data.
    show : bool
        Call ``plt.show()`` when True.
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: data points and the fitted polynomial curve.
    ax_fit.scatter(x, y, s=25, alpha=0.7, color="#1f77b4", label="data")
    x_line = np.linspace(x.min(), x.max(), 300)
    ax_fit.plot(x_line, model.predict(x_line), color="#d62728", lw=2,
                label=f"degree-{model.degree} fit")
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    ax_fit.set_title("Data and fitted curve")
    ax_fit.legend()

    # Right: loss versus iteration (log scale reads better across magnitudes).
    ax_loss.plot(model.loss_history, color="#2ca02c")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("Training loss")
    if len(model.loss_history) > 1 and min(model.loss_history) > 0:
        ax_loss.set_yscale("log")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig
