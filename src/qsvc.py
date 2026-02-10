from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from src.load_data import TRAIN, TEST, FEATS, TARGS

def trainQSVC(qkernel: FidelityQuantumKernel, data: dict) -> QSVC:
  """Trains a QSVC model using prepared quantum kernel & data"""
  
  qsvc = QSVC(kernel=qkernel)
  qsvc.fit(data[FEATS], data[TARGS])

  return qsvc


def testQSVC(qsvc: QSVC, data: dict) -> float:
  """Tests a QSVC model using prepared data, returns accuracy score"""
  return qsvc.score(data[FEATS], data[TARGS])