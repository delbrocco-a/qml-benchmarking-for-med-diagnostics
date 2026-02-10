from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from typing import Optional

ZZ    = "ZZFeatureMap"
Z     = "ZFeatureMap"
PAULI = "PauliFeatureMap"

def FQKernel(
  qubits: int, encodes: Optional[int]=2, map: Optional[str]="ZZFeatureMap"
) -> FidelityQuantumKernel:
  """Generates a Quantum Kernel using feature map & settings"""

  match(map):
    case "ZFeatureMap":
      feat_map = ZFeatureMap(feature_dimension=qubits, reps=encodes)
    case "PauliFeatureMap":
      feat_map = PauliFeatureMap(feature_dimension=qubits, reps=encodes)
    case _:
      feat_map = ZZFeatureMap(feature_dimension=qubits, reps=encodes)

  return FidelityQuantumKernel(feature_map=feat_map)