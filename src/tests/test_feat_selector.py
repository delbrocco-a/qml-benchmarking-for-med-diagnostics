import numpy as np

from src.feat_selector import PCASelect


def test_pceselect_basic_behavior():
  rng = np.random.default_rng(42)

  data = {
    "training": {
      "features": rng.normal(size=(20, 10)),
      "targets": rng.integers(0, 2, size=20),
    },
    "testing": {
      "features": rng.normal(size=(5, 10)),
    },
  }

  train_pca, test_pca = PCASelect(select=5, feats=3, data=data)

  assert train_pca.shape == (20, 3)
  assert test_pca.shape == (5, 3)

  assert np.isfinite(train_pca).all()
  assert np.isfinite(test_pca).all()
