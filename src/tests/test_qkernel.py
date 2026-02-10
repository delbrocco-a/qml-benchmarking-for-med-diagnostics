from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import QuantumCircuit

from src.qkernel import FQKernel, ZZ, Z, PAULI

def test_fqkernel_feature_map_selection():
  cases = [
    (ZZ, ZZFeatureMap),
    (Z, ZFeatureMap),
    (PAULI, PauliFeatureMap),
    ("invalid", ZZFeatureMap),  ### Fallback behavior
    (None, ZZFeatureMap),       ### Default behavior
  ]

  for map_name, expected_class in cases:
    kernel = FQKernel(qubits=4, encodes=2, map=map_name)

    fm = kernel.feature_map
    assert isinstance(fm, QuantumCircuit)
    assert isinstance(kernel, FidelityQuantumKernel)
    assert isinstance(kernel.feature_map, expected_class)
    assert kernel.feature_map.feature_dimension == 4
    assert kernel.feature_map.reps == 2
