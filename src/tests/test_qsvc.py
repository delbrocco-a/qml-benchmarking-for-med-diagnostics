import numpy as np
from unittest.mock import MagicMock
from qiskit_machine_learning.algorithms.classifiers import QSVC

from src.qsvc import trainQSVC, evalQSVC
from src.load_data import FEATURES, TARGETS


def test_qsvc_training_and_testing_flow(monkeypatch):
    # monkeypatch auto-restores src.qsvc.QSVC after this test so later tests
    # (e.g. test_sanity_havlicek) get the real class, not this mock
    fake_kernel = MagicMock()

    fake_qsvc = MagicMock(spec=QSVC)
    fake_qsvc.score.return_value   = 0.85
    fake_qsvc.predict.return_value = np.array([0, 1])

    monkeypatch.setattr("src.qsvc.QSVC", lambda *a, **kw: fake_qsvc)

    data = {
        FEATURES: [[0.1, 0.2], [0.3, 0.4]],
        TARGETS:  [0, 1],
    }

    model = trainQSVC(fake_kernel, data)
    acc, bacc, f1, sens, spec, auc = evalQSVC(model, data)

    fake_qsvc.fit.assert_called_once_with(data[FEATURES], data[TARGETS])
    fake_qsvc.score.assert_called_once_with(data[FEATURES], data[TARGETS])
    fake_qsvc.predict.assert_called_once_with(data[FEATURES])

    assert model is fake_qsvc
    assert isinstance(acc,  float)
    assert isinstance(bacc, float)
    assert isinstance(f1,   float)
