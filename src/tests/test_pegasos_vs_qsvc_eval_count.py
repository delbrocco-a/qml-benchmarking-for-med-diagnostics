"""PegasosQSVC must use strictly fewer kernel evaluations than QSVC.

PegasosQSVC is a stochastic subgradient algorithm: each step selects one
training point, so the number of kernel evaluations scales as O(tau * n)
rather than O(n^2) for the full kernel matrix QSVC needs.
"""
import numpy as np
import pytest
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from unittest.mock import patch

from src.qkernel import FQKernel, ZZ
from src.qsvc import trainQSVC, trainPQSVC
from src.CONST import TRAINING, FEATURES, TARGETS


class _CountingKernel:
    """Wraps FidelityQuantumKernel and tallies kernel-pair evaluations.

    QSVC builds the full n×n Gram matrix in one batched evaluate() call.
    PegasosQSVC makes one call per step, each touching a small slice.
    Counting result.size (total matrix elements) captures actual pairs,
    not just the number of invocations.
    """

    def __init__(self, inner):
        self._inner = inner
        self.pair_count = 0

    def evaluate(self, x_vec, y_vec=None):
        result = self._inner.evaluate(x_vec, y_vec)
        self.pair_count += result.size
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(scope="module")
def small_circles():
    X, y = make_circles(n_samples=40, noise=0.05, factor=0.3, random_state=42)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)
    return {FEATURES: X, TARGETS: y}


def test_pegasos_fewer_kernel_evals_than_qsvc(small_circles):
    inner = FQKernel(qubits=2, reps=2, map=ZZ)

    qsvc_kernel    = _CountingKernel(inner)
    pegasos_kernel = _CountingKernel(inner)

    n_train = len(small_circles[TARGETS])
    # tau << n_train so Pegasos touches far fewer pairs than QSVC's full n×n matrix.
    tau = max(1, n_train // 8)

    trainQSVC(qsvc_kernel, small_circles)
    trainPQSVC(pegasos_kernel, small_circles, C=1.0, tau=tau)

    assert pegasos_kernel.pair_count < qsvc_kernel.pair_count, (
        f"Expected PegasosQSVC ({pegasos_kernel.pair_count} pairs) < "
        f"QSVC ({qsvc_kernel.pair_count} pairs) kernel evaluations"
    )
