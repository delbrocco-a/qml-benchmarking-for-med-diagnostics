from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression

from src.log_reg import trainLogReg, evalLogReg
from src.load_data import FEATURES, TARGETS


def test_log_reg_training_and_evaluation_flow():
  fake_log_reg = MagicMock(spec=LogisticRegression)
  fake_log_reg.score.return_value = 0.85

  # Patch LogisticRegression constructor
  import src.log_reg as this
  this.LogisticRegression = lambda *args, **kwargs: fake_log_reg

  data = {
    FEATURES: [[0.1, 0.2], [0.3, 0.4]],
    TARGETS: [0, 1],
  }

  model = trainLogReg(data)
  score = evalLogReg(model, data)

  fake_log_reg.fit.assert_called_once_with(data[FEATURES], data[TARGETS])
  fake_log_reg.score.assert_called_once_with(data[FEATURES], data[TARGETS])

  assert model is fake_log_reg
  assert isinstance(score, float)