import numpy as np
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import QuantumCircuit

from src.qkernel import FQKernel, ZZ, Z, PAULI


def test_quantum_kernel_matrix_is_symmetric_unit_diagonal_psd_and_bounded():
    """Validates mathematical properties guaranteed by quantum mechanics.

    A fidelity kernel K(x,y) = |<ψ(x)|ψ(y)>|² must satisfy:
      - Symmetry:   K[i,j] = K[j,i]  (fidelity is symmetric)
      - Unit diag:  K[i,i] = 1       (state fidelity with itself = 1)
      - Bounded:    K[i,j] ∈ [0,1]   (fidelity is a probability)
      - PSD:        all eigenvalues ≥ 0  (Gram matrix construction)

    Uses ZFeatureMap (single-qubit only) with reps=1 for fast simulation.
    Statevector backend produces exact results — no shot noise tolerance needed.
    """
    rng = np.random.default_rng(42)
    # 6 samples pre-scaled to [0, π] as required by angle encoding
    X = rng.uniform(0, np.pi, size=(6, 2))

    kernel = FQKernel(qubits=2, reps=1, map=Z)
    K = kernel.evaluate(X)

    assert K.shape == (6, 6), f"Expected 6×6 kernel matrix, got {K.shape}"

    # Unit diagonal
    diag = np.diag(K)
    np.testing.assert_allclose(diag, 1.0, atol=1e-6,
        err_msg="Diagonal entries must be 1 (state fidelity with itself)")

    # Symmetry
    np.testing.assert_allclose(K, K.T, atol=1e-6,
        err_msg="Kernel matrix must be symmetric")

    # Bounded in [0, 1]
    assert K.min() >= -1e-6, f"Kernel values must be ≥0, got min={K.min():.6f}"
    assert K.max() <=  1 + 1e-6, f"Kernel values must be ≤1, got max={K.max():.6f}"

    # Positive semi-definite
    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() >= -1e-6, (
        f"Kernel matrix must be PSD; smallest eigenvalue={eigenvalues.min():.6f}"
    )


def test_fqkernel_feature_map_selection():
  cases = [
    (ZZ, ZZFeatureMap),
    (Z, ZFeatureMap),
    (PAULI, PauliFeatureMap),
  ]

  for map_name, expected_class in cases:
    kernel = FQKernel(qubits=4, reps=2, map=map_name)

    fm = kernel.feature_map
    assert isinstance(fm, QuantumCircuit)
    assert isinstance(kernel, FidelityQuantumKernel)
    assert isinstance(kernel.feature_map, expected_class)
    assert kernel.feature_map.feature_dimension == 4
    assert kernel.feature_map.reps == 2


def test_fqkernel_unknown_map_raises():
  import pytest
  with pytest.raises(ValueError, match="Unknown feature map"):
    FQKernel(qubits=4, reps=2, map="invalid")
