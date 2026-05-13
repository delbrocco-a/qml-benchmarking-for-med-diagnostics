from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from src.CONST import TRAINING, TESTING, FEATURES, TARGETS

# Shots for QPCA tomography. The algorithm recommends >=10000 but that is
# not feasible on consumer hardware. 4096 is a practical compromise:
# higher than the original 1024 (which caused eigenvalue extraction to
# fail and return 0 components on some datasets), while still completing
# within a few minutes for n_feats=2.
QPCA_SHOTS = 4096


def PCASelect(
    select: int, feats: int, data: dict
) -> tuple[np.ndarray, np.ndarray]:
    """SelectKBest (select features), then classical PCA (feats components)."""
    feats_train_select, feats_test_select = KBestSelector(select, data)
    pca = PCA(n_components=feats)
    feats_train_pca = pca.fit_transform(feats_train_select)
    feats_test_pca = pca.transform(feats_test_select)
    return feats_train_pca, feats_test_pca



def ScaleForQuantum(
  train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Rescale PCA features to [0, π] for quantum angle encoding.

  ZZFeatureMap encodes each feature as a rotation angle — if features are
  on arbitrary scales the inner products between quantum states become
  meaningless. Scaler is fit only on training data to avoid data leakage.

  Ref: Havlíček, V. et al. (2019) 'Supervised learning with quantum-enhanced
  feature spaces', Nature, 567, pp. 209-212. doi:10.1038/s41586-019-0980-2.
  """
  scaler = MinMaxScaler(feature_range=(0, np.pi))
  return scaler.fit_transform(train), scaler.transform(test)


def KBestSelector(select: int, data: dict) -> tuple[np.ndarray, np.ndarray]:
  """Performs SelectKBest algorithm for the PCA functions"""

  selector = SelectKBest(f_classif, k=select)
  train_select = selector.fit_transform(
    data[TRAINING][FEATURES], data[TRAINING][TARGETS]
  )
  test_select = selector.transform(data[TESTING][FEATURES])

  return train_select, test_select

