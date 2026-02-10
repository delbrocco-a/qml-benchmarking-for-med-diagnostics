from unittest.mock import MagicMock
from qiskit_machine_learning.algorithms.classifiers import QSVC

from src.qsvc import trainQSVC, evalQSVC
from src.load_data import FEATURES, TARGETS


def test_qsvc_training_and_testing_flow():
    fake_kernel = MagicMock()

    fake_qsvc = MagicMock(spec=QSVC)
    fake_qsvc.score.return_value = 0.85

    def fake_qsvc_ctor(*args, **kwargs):
        return fake_qsvc

    import src.qsvc as this
    this.QSVC = fake_qsvc_ctor

    data = {
        FEATURES: [[0.1, 0.2], [0.3, 0.4]],
        TARGETS: [0, 1],
    }

    model = trainQSVC(fake_kernel, data)

    score = evalQSVC(model, data)

    fake_qsvc.fit.assert_called_once_with(data[FEATURES], data[TARGETS])
    fake_qsvc.score.assert_called_once_with(data[FEATURES], data[TARGETS])

    assert model is fake_qsvc
    assert isinstance(score, float)
