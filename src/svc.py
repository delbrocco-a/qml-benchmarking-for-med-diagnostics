from sklearn.svm import SVC
from typing import Optional

from src.load_data import FEATURES, TARGETS

def trainSVC(data: dict, kernel: Optional[str]="rbf") -> SVC:
  """Trains a classical SVC model using training data"""
  
  svc = SVC(kernel=kernel)
  svc.fit(data[FEATURES], data[TARGETS])

  return svc


def evalSVC(svc: SVC, data: dict) -> float:
  """Tests a classical SVC model using testing data, returns accuracy score"""

  return svc.score(data[FEATURES], data[TARGETS])

