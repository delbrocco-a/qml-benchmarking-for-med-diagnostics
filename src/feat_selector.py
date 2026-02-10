from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

from src.load_data import TRAINING, TESTING, FEATURES, TARGETS

def PCASelect(
  select: int, feats: int, data: dict
) -> tuple[np.ndarray, np.ndarray]:
  """Performs SelectKBest (select), then PCA (feats) on training & testing 
  data, returning a dataset with feats features for training & testing"""
  
  selector = SelectKBest(f_classif, k=select)
  feats_train_select = selector.fit_transform(
    data[TRAINING][FEATURES], data[TRAINING][TARGETS]
  )
  feats_test_select = selector.transform(data[TESTING][FEATURES]) 
  
  pca = PCA(n_components=feats)
  feats_train_pca = pca.fit_transform(feats_train_select)
  feats_test_pca = pca.transform(feats_test_select)
  
  return feats_train_pca, feats_test_pca
