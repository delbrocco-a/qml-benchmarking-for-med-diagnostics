from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score,
)

from src.load_data import TRAINING, TESTING, FEATURES, TARGETS


def trainQSVC(qkernel: FidelityQuantumKernel, data: dict) -> QSVC:
  """Trains a QSVC model using prepared quantum kernel & training data"""

  qsvc = QSVC(quantum_kernel=qkernel)
  qsvc.fit(data[FEATURES], data[TARGETS])

  return qsvc


def evalQSVC(qsvc: QSVC, data: dict) -> tuple[float, ...]:
  """Tests a QSVC model.

  Returns (accuracy, balanced_accuracy, f1, sensitivity, specificity, auc).
  """
  preds = qsvc.predict(data[FEATURES])
  acc   = qsvc.score(data[FEATURES], data[TARGETS])
  bacc  = balanced_accuracy_score(data[TARGETS], preds)
  f1    = f1_score(data[TARGETS], preds, average="binary", zero_division=0)

  tn, fp, fn, tp = confusion_matrix(data[TARGETS], preds, labels=[0, 1]).ravel()
  sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

  try:
      scores = qsvc.decision_function(data[FEATURES])
      auc    = roc_auc_score(data[TARGETS], scores)
  except Exception:
      auc = float("nan")

  return acc, bacc, f1, sens, spec, auc


def trainPQSVC(
  qkernel: FidelityQuantumKernel, data: dict, C: float, tau: int
) -> PegasosQSVC:
  """Trains a PegasosQSVC model using prepared quantum kernel & data"""

  pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)
  pegasos_qsvc.fit(data[FEATURES], data[TARGETS])

  return pegasos_qsvc


def evalPQSVC(pegasos_qsvc: PegasosQSVC, data: dict) -> tuple[float, ...]:
  """Tests a PegasosQSVC model.

  Returns (accuracy, balanced_accuracy, f1, sensitivity, specificity, auc).
  AUC uses decision_function; falls back to nan if unsupported in this
  Qiskit version (0.8.2 support is version-dependent).
  """
  preds = pegasos_qsvc.predict(data[FEATURES])
  acc   = pegasos_qsvc.score(data[FEATURES], data[TARGETS])
  bacc  = balanced_accuracy_score(data[TARGETS], preds)
  f1    = f1_score(data[TARGETS], preds, average="binary", zero_division=0)

  tn, fp, fn, tp = confusion_matrix(data[TARGETS], preds, labels=[0, 1]).ravel()
  sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

  try:
      scores = pegasos_qsvc.decision_function(data[FEATURES])
      auc    = roc_auc_score(data[TARGETS], scores)
  except Exception:
      auc = float("nan")

  return acc, bacc, f1, sens, spec, auc