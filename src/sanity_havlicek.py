"""
Sanity check: ZZFeatureMap should separate make_circles.

Havlíček et al. (2019) showed that ZZFeatureMap creates a feature space
that can separate certain datasets classical kernels can't. Their exact
dataset isn't public, but make_circles(factor=0.3) is a standard
stand-in — it's non-linearly separable in input space but becomes
separable under ZZ's kernel. If this test fails it means the quantum
kernel circuit or the QSVC wrapper is broken, not that quantum is bad.

Reference:
  Havlíček, V. et al. (2019) 'Supervised learning with quantum-enhanced
  feature spaces', Nature, 567, pp. 209-212. doi:10.1038/s41586-019-0980-2.
"""

import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.qkernel import FQKernel, ZZ
from src.qsvc   import trainQSVC, evalQSVC

ACCURACY_THRESHOLD = 0.85


def run_havlicek_check(seed: int = 42, verbose: bool = True) -> float:
    """Trains QSVC on make_circles and returns test balanced accuracy."""

    X, y = make_circles(n_samples=200, noise=0.05, factor=0.3,
                        random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )

    # Scale to [0, π] — same preprocessing as the main pipeline
    scaler  = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    train_data = {"features": X_train, "targets": y_train}
    test_data  = {"features": X_test,  "targets": y_test}

    kernel = FQKernel(qubits=2, reps=2, map=ZZ)
    model  = trainQSVC(kernel, train_data)
    _, bacc, _, _, _, _ = evalQSVC(model, test_data)

    if verbose:
        status = "PASS" if bacc >= ACCURACY_THRESHOLD else "FAIL"
        print(f"  Havlíček sanity [{status}]  balanced accuracy = {bacc:.3f}"
              f"  (threshold {ACCURACY_THRESHOLD})")
        if bacc < ACCURACY_THRESHOLD:
            print(
                "\n  PIPELINE BROKEN — ZZFeatureMap not separating known-separable"
                " data.\n  Check FQKernel, QSVC wrapper, and quantum scaling.\n"
            )

    return bacc


if __name__ == "__main__":
    bacc = run_havlicek_check(verbose=True)
    raise SystemExit(0 if bacc >= ACCURACY_THRESHOLD else 1)
