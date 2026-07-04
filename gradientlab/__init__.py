"""GradientLab: polynomial regression via manual gradient descent."""

from .data import load_csv, standardize
from .model import (
    PolynomialRegression,
    design_matrix,
    gradient,
    mse,
    predict,
    r_squared,
)

__version__ = "0.1.0"

__all__ = [
    "PolynomialRegression",
    "design_matrix",
    "predict",
    "mse",
    "r_squared",
    "gradient",
    "load_csv",
    "standardize",
]
