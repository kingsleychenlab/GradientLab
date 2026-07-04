"""Command-line interface for GradientLab.

Example
-------
    python -m gradientlab --file data/sample_quadratic.csv --degree 2 \\
        --learning-rate 0.1 --normalize --plot
"""

from __future__ import annotations

import argparse
import sys

from . import data
from .model import PolynomialRegression


def build_parser():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="gradientlab",
        description="Fit a polynomial to x,y data using gradient descent.",
    )
    parser.add_argument("--file", required=True,
                        help="CSV file with columns x,y")
    parser.add_argument("--degree", type=int, default=2,
                        help="polynomial degree (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="step size alpha (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="maximum iterations (default: 10000)")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="stop when abs(prev_loss - loss) < tolerance "
                             "(default: 1e-9)")
    parser.add_argument("--normalize", action="store_true",
                        help="standardize x before fitting (recommended for "
                             "degree >= 2)")
    parser.add_argument("--plot", action="store_true",
                        help="show data, fitted curve and loss curve")
    return parser


def main(argv=None):
    """Entry point. Returns the fitted model (handy for tests)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        x, y = data.load_csv(args.file)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    model = PolynomialRegression(
        degree=args.degree,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        tolerance=args.tolerance,
        normalize=args.normalize,
    )
    model.fit(x, y)

    final_loss = model.loss(x, y)
    r2 = model.score(x, y)

    print(model.equation())
    print(f"final loss (MSE): {final_loss:.6g}")
    print(f"R^2 score:        {r2:.6f}")
    print(f"iterations:       {model.n_iterations}")
    print(f"converged:        {'yes' if model.converged else 'no'}")
    if model.diverged:
        print(
            "\nWARNING: training diverged (loss became non-finite).\n"
            "Try adding --normalize or lowering --learning-rate.",
            file=sys.stderr,
        )

    if args.plot:
        from . import plotting
        plotting.plot_results(model, x, y)

    return model


if __name__ == "__main__":
    main()
