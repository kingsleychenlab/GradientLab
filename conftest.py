"""Put the repository root on sys.path so ``import gradientlab`` works
without installing the package (handy for running the test suite directly)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
