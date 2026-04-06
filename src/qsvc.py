from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from src.load_data import TRAINING, TESTING, FEATURES, TARGETS


def trainQSVC(qkernel: FidelityQuantumKernel, data: dict) -> QSVC:
  """Trains a QSVC model using prepared quantum kernel & training data"""
  
  qsvc = QSVC(quantum_kernel=qkernel)
  qsvc.fit(data[FEATURES], data[TARGETS])

  return qsvc


def evalQSVC(qsvc: QSVC, data: dict) -> float:
  """Tests a QSVC model using testing data, returns accuracy score"""

  return qsvc.score(data[FEATURES], data[TARGETS])


def trainPQSVC(
  qkernel: FidelityQuantumKernel, data: dict, C: float, tau: int
) -> PegasosQSVC:
  """Trains a PegasosQSVC model using prepared quantum kernel & data"""
  
  pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)
  pegasos_qsvc.fit(data[FEATURES], data[TARGETS])

  return pegasos_qsvc


def evalPQSVC(pegasos_qsvc: PegasosQSVC, data: dict) -> float:
  """Tests a PegasosQSVC model using testing data, returns accuracy score"""

  return pegasos_qsvc.score(data[FEATURES], data[TARGETS])