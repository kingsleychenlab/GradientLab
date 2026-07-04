"""Tests for GradientLab's math and training loop."""

import numpy as np
import pytest

from gradientlab.data import load_csv, standardize
from gradientlab.model import (
    PolynomialRegression,
    design_matrix,
    gradient,
    mse,
    predict,
    r_squared,
)


# --- prediction ------------------------------------------------------------
def test_prediction_matches_manual_polynomial():
    # y_hat = 1 + 2x + 3x^2  ->  x=0:1, x=1:6, x=2:17
    theta = np.array([1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0])
    y_hat = predict(design_matrix(x, degree=2), theta)
    assert np.allclose(y_hat, [1.0, 6.0, 17.0])


def test_design_matrix_shape_and_columns():
    x = np.array([2.0, 3.0])
    design = design_matrix(x, degree=3)
    assert design.shape == (2, 4)
    # Columns are x^0, x^1, x^2, x^3.
    assert np.allclose(design[0], [1.0, 2.0, 4.0, 8.0])


# --- MSE -------------------------------------------------------------------
def test_mse_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_mse_known_value():
    # mean of (3-1)^2 and (5-2)^2 = mean(4, 9) = 6.5
    assert mse([3.0, 5.0], [1.0, 2.0]) == pytest.approx(6.5)


# --- R^2 -------------------------------------------------------------------
def test_r_squared_perfect_fit_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_mean_predictor_is_zero():
    # Predicting the mean everywhere gives R^2 = 0 by definition.
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.full_like(y, y.mean())
    assert r_squared(y_pred, y) == pytest.approx(0.0)


def test_r_squared_undefined_when_no_variance():
    assert np.isnan(r_squared([5.0, 5.0], [5.0, 5.0]))


# --- gradient --------------------------------------------------------------
def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=25)
    y = rng.uniform(-2, 2, size=25)
    design = design_matrix(x, degree=3)
    theta = rng.normal(size=4)

    analytic = gradient(design, theta, y)

    eps = 1e-6
    numeric = np.zeros_like(theta)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = eps
        loss_plus = mse(predict(design, theta + step), y)
        loss_minus = mse(predict(design, theta - step), y)
        numeric[j] = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(analytic, numeric, atol=1e-5)


# --- training --------------------------------------------------------------
def test_loss_decreases_during_training():
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 50)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, size=x.size)

    model = PolynomialRegression(degree=1, learning_rate=0.05, epochs=2000)
    model.fit(x, y)

    history = model.loss_history
    assert history[-1] < history[0]
    # Batch gradient descent with a stable step is monotonically non-increasing.
    diffs = np.diff(history)
    assert np.all(diffs <= 1e-9)


def test_fit_recovers_known_line():
    x = np.linspace(0, 10, 60)
    y = 3.0 * x - 2.0  # exact line, no noise

    model = PolynomialRegression(degree=1, learning_rate=0.1, epochs=50000,
                                 tolerance=1e-12, normalize=True)
    model.fit(x, y)

    assert model.score(x, y) == pytest.approx(1.0, abs=1e-6)
    # Coefficients are reported in raw-x space: intercept -2, slope 3.
    assert np.allclose(model.coefficients(), [-2.0, 3.0], atol=1e-3)


def test_convergence_flag_set_before_max_epochs():
    x = np.linspace(-2, 2, 30)
    y = x ** 2
    model = PolynomialRegression(degree=2, learning_rate=0.1, epochs=100000,
                                 tolerance=1e-10, normalize=True)
    model.fit(x, y)
    assert model.converged
    assert model.n_iterations < model.epochs


def test_equation_coefficients_match_predictions_when_normalized():
    rng = np.random.default_rng(7)
    x = np.linspace(-5, 5, 40)
    y = 0.5 * x ** 2 - x + 2 + rng.normal(0, 0.3, size=x.size)

    model = PolynomialRegression(degree=2, learning_rate=0.1, epochs=20000,
                                 normalize=True)
    model.fit(x, y)

    # Evaluating the reported raw-x coefficients must reproduce predict().
    x_test = np.linspace(-5, 5, 11)
    from_coeffs = np.polynomial.polynomial.polyval(x_test, model.coefficients())
    assert np.allclose(from_coeffs, model.predict(x_test), atol=1e-6)


def test_diverges_with_large_learning_rate():
    x = np.linspace(-6, 6, 40)
    y = x ** 2
    model = PolynomialRegression(degree=2, learning_rate=0.05, epochs=500)
    model.fit(x, y)
    assert model.diverged
    assert not model.converged


# --- data ------------------------------------------------------------------
def test_standardize_zero_mean_unit_std():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z, mean, std = standardize(x)
    assert mean == pytest.approx(3.0)
    assert np.mean(z) == pytest.approx(0.0)
    assert np.std(z) == pytest.approx(1.0)


def test_standardize_constant_feature_does_not_divide_by_zero():
    z, mean, std = standardize([7.0, 7.0, 7.0])
    assert std == 1.0
    assert np.allclose(z, 0.0)


def test_load_csv_with_header(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n0,1\n1,3\n2,5\n")
    x, y = load_csv(path)
    assert np.allclose(x, [0, 1, 2])
    assert np.allclose(y, [1, 3, 5])


def test_load_csv_without_header(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("0.0,1.0\n1.0,3.0\n")
    x, y = load_csv(path)
    assert np.allclose(x, [0.0, 1.0])
    assert np.allclose(y, [1.0, 3.0])


def test_load_csv_empty_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("x,y\n")
    with pytest.raises(ValueError):
        load_csv(path)
