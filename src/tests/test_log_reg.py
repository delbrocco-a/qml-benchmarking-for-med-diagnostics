import numpy as np
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression

from src.log_reg import trainLogReg, evalLogReg
from src.load_data import FEATURES, TARGETS


def _separable_data():
    rng = np.random.default_rng(0)
    X0  = rng.normal(loc=-3, scale=0.5, size=(30, 2))
    X1  = rng.normal(loc=+3, scale=0.5, size=(30, 2))
    X   = np.vstack([X0, X1])
    y   = np.array([0] * 30 + [1] * 30)
    return (
        {FEATURES: X[:48], TARGETS: y[:48]},
        {FEATURES: X[48:], TARGETS: y[48:]},
    )


def test_logreg_on_separable_data_achieves_100_percent():
    """Logistic regression on linearly separable data must converge to 100%.

    With max_iter=1000 and a 6σ cluster gap, the log-loss surface has a single
    global minimum that perfectly separates the classes.
    """
    train, test = _separable_data()
    model = trainLogReg(train)
    acc, bacc, f1, sens, spec, auc = evalLogReg(model, test)
    assert acc  == 1.0, f"Expected 100% on separable data, got {acc:.2%}"
    assert bacc == 1.0
    assert f1   == 1.0


def test_log_reg_training_and_evaluation_flow(monkeypatch):
  fake_log_reg = MagicMock(spec=LogisticRegression)
  fake_log_reg.score.return_value   = 0.85
  fake_log_reg.predict.return_value = np.array([0, 1])

  monkeypatch.setattr("src.log_reg.LogisticRegression",
                      lambda *args, **kwargs: fake_log_reg)

  data = {
    FEATURES: [[0.1, 0.2], [0.3, 0.4]],
    TARGETS:  [0, 1],
  }

  model = trainLogReg(data)
  acc, bacc, f1, sens, spec, auc = evalLogReg(model, data)

  fake_log_reg.fit.assert_called_once_with(data[FEATURES], data[TARGETS])
  fake_log_reg.score.assert_called_once_with(data[FEATURES], data[TARGETS])
  fake_log_reg.predict.assert_called_once_with(data[FEATURES])

  assert model is fake_log_reg
  assert isinstance(acc,  float)
  assert isinstance(bacc, float)
  assert isinstance(f1,   float)
