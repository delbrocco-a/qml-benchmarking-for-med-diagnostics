import numpy as np
from unittest.mock import MagicMock

from src.feat_selector import PCASelect, QPCASelect, KBestSelector
from src.load_data import TRAINING, TESTING, FEATURES, TARGETS


def test_kbestselector_reduces_features():
  rng = np.random.default_rng(42)

  data = {
    TRAINING: {
      FEATURES: rng.normal(size=(20, 10)),
      TARGETS: rng.integers(0, 2, size=20),
      },
    TESTING: {
      FEATURES: rng.normal(size=(5, 10)),
    },
  }

  train_sel, test_sel = KBestSelector(select=4, data=data)

  assert train_sel.shape == (20, 4)
  assert test_sel.shape == (5, 4)
  assert np.isfinite(train_sel).all()
  assert np.isfinite(test_sel).all()


def test_pcaselect_pipeline_shapes():
  rng = np.random.default_rng(0)

  data = {
    TRAINING: {
      FEATURES: rng.normal(size=(30, 12)),
      TARGETS: rng.integers(0, 2, size=30),
      },
    TESTING: {
      FEATURES: rng.normal(size=(8, 12)),
    },
  }

  train_pca, test_pca = PCASelect(select=6, feats=3, data=data)

  assert train_pca.shape == (30, 3)
  assert test_pca.shape == (8, 3)


def test_qpcaselect_pipeline_shapes(monkeypatch):
  rng = np.random.default_rng(1)

  data = {
    TRAINING: {
      FEATURES: rng.normal(size=(25, 10)),
      TARGETS: rng.integers(0, 2, size=25),
    },
    TESTING: {
      FEATURES: rng.normal(size=(6, 10)),
    },
  }

  # Create fake QPCA
  fake_qpca = MagicMock()
  fake_qpca.transform.side_effect = lambda x: x[:, :3]  # fake dimensionality reduction

  monkeypatch.setattr("your_module.QPCA", lambda: fake_qpca)

  train_qpca, test_qpca = QPCASelect(select=5, feats=3, data=data)

  assert train_qpca.shape[0] == 25
  assert test_qpca.shape[0] == 6
  assert train_qpca.shape[1] == 3
  assert test_qpca.shape[1] == 3
