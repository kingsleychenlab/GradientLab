# GradientLab

A small, honest educational tool for **polynomial regression trained with
gradient descent**. Every number it prints comes from real math — the gradient
is derived and implemented by hand, and the only heavy lifting delegated to
NumPy is vectorised linear algebra. No scikit-learn, no autograd, no fake
outputs, no hardcoded equations.

---

## What it does

Given a CSV of `x,y` points, GradientLab fits a degree-`n` polynomial

```
y_hat = θ₀ + θ₁x + θ₂x² + ... + θₙxⁿ
```

by minimising the mean squared error with batch gradient descent, then reports
the fitted equation, the final loss, the R² score, the number of iterations,
and whether it converged. Optionally it plots the fit and the loss curve.

---

## The math

**Model (hypothesis).** With parameters `θ = [θ₀, ..., θₙ]` and the design
matrix `X` whose row `i` is `[1, xᵢ, xᵢ², ..., xᵢⁿ]`:

```
y_hat = X · θ
```

**Loss — mean squared error** over `m` samples:

```
J(θ) = (1/m) · Σᵢ (y_hatᵢ − yᵢ)²
```

**Gradient** (derived by hand, since `∂y_hatᵢ/∂θⱼ = xᵢʲ`):

```
∂J/∂θⱼ = (2/m) · Σᵢ (y_hatᵢ − yᵢ) · xᵢʲ
```

which vectorises to `∇J = (2/m) · Xᵀ (X·θ − y)`.

**Parameter update** (step size / learning rate `α`):

```
θⱼ := θⱼ − α · ∂J/∂θⱼ
```

**R² (coefficient of determination)** reported after training:

```
R² = 1 − SS_res / SS_tot
   = 1 − Σ(yᵢ − y_hatᵢ)² / Σ(yᵢ − ȳ)²
```

**Feature normalization (why it matters).** For a degree-`n` fit the columns of
`X` span `x⁰ … xⁿ`, whose magnitudes differ by orders of magnitude, so the loss
surface becomes badly conditioned and gradient descent either crawls or blows
up. With `--normalize`, `x` is standardized to zero mean and unit variance,
`z = (x − μ) / σ`, before the powers are formed. The model stores `μ, σ` and
applies the same transform at prediction time; the reported equation is then
**algebraically expanded back into raw `x`**, so what you read is directly
usable. Same answer, far fewer iterations — and high degrees stop diverging.

---

## Project structure

```
GradientLab/
├── README.md
├── requirements.txt
├── data/
│   ├── sample_linear.csv       # y = 2x + 1 + noise
│   └── sample_quadratic.csv    # y = 0.5x² − 2x + 3 + noise
├── gradientlab/
│   ├── __init__.py
│   ├── model.py                # design matrix, MSE, gradient, R², training loop
│   ├── data.py                 # CSV loading, standardization
│   ├── plotting.py             # data + fit + loss-curve plots (matplotlib)
│   └── cli.py                  # argparse command-line interface
└── tests/
    └── test_model.py
```

---

## How to use it

### Step 1 — Install (once)

From the project root, create a virtual environment and install the
dependencies. Only NumPy is needed to train; matplotlib is used for `--plot`
and pytest for the tests.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Run it on the included data

The command is always `python -m gradientlab` followed by `--file` and any
options. Try the quadratic sample:

```bash
python -m gradientlab --file data/sample_quadratic.csv --degree 2 --normalize --learning-rate 0.1 --plot
```

This trains a degree-2 polynomial, prints the fitted equation, final loss, R²,
iteration count and whether it converged, and — because of `--plot` — opens a
window showing the data with the fitted curve and the training loss curve.

### Step 3 — Run it on your own data

Point `--file` at any CSV with two columns named `x` and `y` (the header row is
optional):

```
x,y
0,1.2
1,3.1
2,5.4
3,9.0
```

```bash
python -m gradientlab --file yourdata.csv --degree 3 --normalize
```

**Rule of thumb:** for any `--degree` of 2 or higher, add `--normalize`.
Without it the high-power terms blow up the gradient and training diverges
(you'll get a warning telling you exactly that).

### All options

Run `python -m gradientlab --help` to see these at any time.

| Flag | Default | Meaning |
|------|---------|---------|
| `--file` | *(required)* | CSV file with columns `x,y` (a header row is optional) |
| `--degree` | `2` | Polynomial degree `n` |
| `--learning-rate` | `0.01` | Step size `α` (smaller = safer but slower) |
| `--epochs` | `10000` | Maximum number of iterations |
| `--tolerance` | `1e-9` | Stop early when `abs(prev_loss − loss) < tolerance` |
| `--normalize` | off | Standardize `x` before fitting (recommended for degree ≥ 2) |
| `--plot` | off | Show the data + fitted curve and the loss curve |

### More examples

```bash
# Straight line — converges fine without normalization
python -m gradientlab --file data/sample_linear.csv --degree 1

# Same line fit, but normalization gets there in far fewer iterations
python -m gradientlab --file data/sample_linear.csv --degree 1 --normalize --learning-rate 0.1

# Quadratic fit, no plot, tighter stopping tolerance
python -m gradientlab --file data/sample_quadratic.csv --degree 2 --normalize --tolerance 1e-12
```

---

## Sample output

Fitting the quadratic sample (true curve `0.5x² − 2x + 3`):

```
$ python -m gradientlab --file data/sample_quadratic.csv --degree 2 --normalize --learning-rate 0.1
y_hat = 2.7341 - 2.0333*x + 0.5215*x^2
final loss (MSE): 3.38612
R^2 score:        0.962421
iterations:       139
converged:        yes
```

The recovered coefficients (`0.52, −2.03, 2.73`) closely match the underlying
`0.5, −2, 3` — the residual loss is just the noise that was baked into the data.

Running the **same fit without `--normalize`** shows why normalization exists:

```
$ python -m gradientlab --file data/sample_quadratic.csv --degree 2
y_hat = -1.22276e+151 + 2.14598e+135*x - 2.76999e+152*x^2
final loss (MSE): inf
R^2 score:        -inf
iterations:       227
converged:        no

WARNING: training diverged (loss became non-finite).
Try adding --normalize or lowering --learning-rate.
```

The unnormalized `x²` column overwhelms the step size and the parameters blow
up — GradientLab detects the non-finite loss and tells you how to fix it.

---

## Running the tests

```bash
python -m pytest
```

The suite checks the pieces that have to be correct: prediction, MSE, R², the
analytic gradient (against a finite-difference approximation), that the loss
decreases monotonically during training, that a known line is recovered, that
the reported equation reproduces the model's predictions, and CSV loading.

---

## Limitations

- **Batch gradient descent only.** No stochastic/mini-batch variants, momentum,
  or adaptive learning rates — this is a teaching tool, kept deliberately simple.
- **No regularization.** High degrees on small/noisy data will overfit; there is
  no L1/L2 penalty. Prefer the lowest degree that fits.
- **Single feature.** One input `x` and one output `y` (with polynomial terms
  derived from `x`); it is not a general multivariate regressor.
- **Learning rate is manual.** There is no line search; for degree ≥ 2 you will
  almost always want `--normalize`, and very high degrees can still be poorly
  conditioned.
- **Closed form exists.** For plain least squares the normal equations solve this
  exactly; gradient descent is used here because *the point is to show the
  gradient-descent math*, not to be the fastest solver.
