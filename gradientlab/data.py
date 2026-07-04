"""Data helpers: CSV loading and feature standardization."""

from __future__ import annotations

import csv

import numpy as np


def load_csv(path):
    """Load a two-column ``x,y`` CSV file into NumPy arrays.

    A header row (e.g. ``x,y``) is optional and detected automatically: any row
    whose first two fields are not both numbers is skipped. This keeps the
    loader forgiving of headers, blank lines and stray comments.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The ``x`` and ``y`` columns as 1-D float arrays.

    Raises
    ------
    ValueError
        If no numeric ``(x, y)`` rows are found.
    """
    xs, ys = [], []
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                x_value = float(row[0])
                y_value = float(row[1])
            except ValueError:
                # Header row or non-numeric line -> skip.
                continue
            xs.append(x_value)
            ys.append(y_value)

    if not xs:
        raise ValueError(f"no numeric 'x,y' rows found in {path!r}")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def standardize(x):
    """Standardize ``x`` to zero mean and unit variance.

    Returns
    -------
    (numpy.ndarray, float, float)
        The transformed values ``(x - mean) / std`` together with the ``mean``
        and ``std`` used, so the same transform can be reapplied later. A
        constant feature (``std == 0``) falls back to ``std = 1`` to avoid
        division by zero.
    """
    x = np.asarray(x, dtype=float).ravel()
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std == 0.0:
        std = 1.0
    return (x - mean) / std, mean, std
