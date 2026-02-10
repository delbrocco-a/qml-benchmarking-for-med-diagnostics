from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from src.load_data import TRAINING, TESTING, FEATURES, TARGETS

def trainQSVC(qkernel: FidelityQuantumKernel, data: dict) -> QSVC:
  """Trains a QSVC model using prepared quantum kernel & data"""
  
  qsvc = QSVC(kernel=qkernel)
  qsvc.fit(data[FEATURES], data[TARGETS])

  return qsvc


def testQSVC(qsvc: QSVC, data: dict) -> float:
  """Tests a QSVC model using prepared data, returns accuracy score"""
  return qsvc.score(data[FEATURES], data[TARGETS])