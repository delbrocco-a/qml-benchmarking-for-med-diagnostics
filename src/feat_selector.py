from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
# from src.QPCA.decomposition import QPCA
import numpy as np

from src.load_data import TRAINING, TESTING, FEATURES, TARGETS

QUBITS = 8
ESHOTS = 10000


def PCASelect(
  select: int, feats: int, data: dict
) -> tuple[np.ndarray, np.ndarray]:
  """Performs SelectKBest (select), then PCA (feats) on training & testing 
  data, returning a dataset with feats features for training & testing"""
  
  feats_train_select, feats_test_select = KBestSelector(select, data)
  
  pca = PCA(n_components=feats)
  feats_train_pca = pca.fit_transform(feats_train_select)
  feats_test_pca = pca.transform(feats_test_select)
  
  return feats_train_pca, feats_test_pca


# def QPCASelect(
#     select: int, feats: int, data: dict
# ) -> tuple[np.ndarray, np.ndarray]:
#   """Performs SelectKBest (select), then a QPCA algorithm, imported from 
#   Eagle-quantum's repositiory: > https://github.com/Eagle-quantum/QuPCA
#   ~ Principle contributors usesnames and GiuliaFranco (accessed 22/02/2026)"""

#   feats_train_select, feats_test_select = KBestSelector(select, data)

#   qpca = QPCA()
#   qpca.fit(input_matrix=np.cov(feats_train_select.T), resolution=QUBITS)
#   qpca.eigenvectors_reconstruction(n_repetitions=ESHOTS)

#   feats_train_qpca = qpca.transform(feats_train_select)
#   feats_test_qpca = qpca.transform(feats_test_select)

#   return feats_train_qpca, feats_test_qpca


def KBestSelector(select: int, data: dict) -> tuple[np.ndarray, np.ndarray]:
  """Performs SelectKBest algorithm for the PCA functions"""

  selector = SelectKBest(f_classif, k=select)
  train_select = selector.fit_transform(
    data[TRAINING][FEATURES], data[TRAINING][TARGETS]
  )
  test_select = selector.transform(data[TESTING][FEATURES])

  return train_select, test_select

