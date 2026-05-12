import numpy as np
from unittest.mock import MagicMock
from sklearn.svm import SVC

from src.svc import trainSVC, evalSVC
from src.load_data import FEATURES, TARGETS


def _separable_data():
    """Two Gaussian clusters 6σ apart — any linear classifier must score 100%."""
    rng = np.random.default_rng(0)
    X0  = rng.normal(loc=-3, scale=0.5, size=(30, 2))
    X1  = rng.normal(loc=+3, scale=0.5, size=(30, 2))
    X   = np.vstack([X0, X1])
    y   = np.array([0] * 30 + [1] * 30)
    train = {FEATURES: X[:48], TARGETS: y[:48]}
    test  = {FEATURES: X[48:], TARGETS: y[48:]}
    return train, test


def test_svc_linear_kernel_on_separable_data_achieves_100_percent():
    """Reproduces the theoretical SVM guarantee: linear kernel on separable data = 100%.

    SVMs with a linear kernel find the maximum-margin hyperplane, which is
    guaranteed to correctly classify all points when the classes are linearly
    separable (Vapnik, 1995). This test pins that guarantee against our wrapper.
    """
    train, test = _separable_data()
    model = trainSVC(train, kernel="linear")
    acc, bacc, f1, sens, spec, auc = evalSVC(model, test)
    assert acc  == 1.0, f"Expected 100% on separable data, got {acc:.2%}"
    assert bacc == 1.0
    assert f1   == 1.0


def test_svc_training_and_evaluation_flow(monkeypatch):
  fake_svc = MagicMock(spec=SVC)
  fake_svc.score.return_value   = 0.9
  fake_svc.predict.return_value = np.array([0, 1])

  monkeypatch.setattr("src.svc.SVC", lambda *args, **kwargs: fake_svc)

  data = {
    FEATURES: [[0.1, 0.2], [0.3, 0.4]],
    TARGETS:  [0, 1],
  }

  model = trainSVC(data)
  acc, bacc, f1, sens, spec, auc = evalSVC(model, data)

  fake_svc.fit.assert_called_once_with(data[FEATURES], data[TARGETS])
  fake_svc.score.assert_called_once_with(data[FEATURES], data[TARGETS])
  fake_svc.predict.assert_called_once_with(data[FEATURES])

  assert model is fake_svc
  assert isinstance(acc,  float)
  assert isinstance(bacc, float)
  assert isinstance(f1,   float)
