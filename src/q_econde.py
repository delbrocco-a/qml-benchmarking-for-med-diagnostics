from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap

# Common encoders for classical data → quantum states:

# 1. ZZFeatureMap (most common for QSVM)
feature_map = ZZFeatureMap(feature_dimension=4, reps=2)


# 2. PauliFeatureMap (more general)
feature_map = PauliFeatureMap(feature_dimension=4, reps=2, paulis=['Z', 'ZZ'])

# Use with QSVM:
from qiskit_machine_learning.kernels import FidelityQuantumKernel
kernel = FidelityQuantumKernel(feature_map=feature_map)

# Featy

def generate_feature_map(feature_dimension: int, reps: int) -> ZZFeatureMap:
  """Generates a ZZFeatureMap for given feature dimension and repetitions."""
  return ZZFeatureMap(feature_dimension=feature_dimension, reps=reps)