from unittest.mock import MagicMock

from sklearn.svm import SVC

from src.svc import trainSVC, evalSVC
from src.load_data import FEATURES, TARGETS


def test_svc_training_and_evaluation_flow():
  fake_svc = MagicMock(spec=SVC)
  fake_svc.score.return_value = 0.9

  # Patch SVC constructor
  import src.svc as this
  this.SVC = lambda *args, **kwargs: fake_svc

  data = {
    FEATURES: [[0.1, 0.2], [0.3, 0.4]],
    TARGETS: [0, 1],
  }

  model = trainSVC(data)
  score = evalSVC(model, data)

  fake_svc.fit.assert_called_once_with(data[FEATURES], data[TARGETS])
  fake_svc.score.assert_called_once_with(data[FEATURES], data[TARGETS])

  assert model is fake_svc
  assert isinstance(score, float)
