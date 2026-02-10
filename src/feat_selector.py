from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

def PCASelect(
  select: int, feats: int, data: dict
) -> tuple[np.ndarray, np.ndarray]:
  
  selector = SelectKBest(f_classif, k=select)
  feats_train_select = selector.fit_transform(
    data["training"]["features"], data["training"]["targets"]
  )
  feats_test_select = selector.transform(data["testing"]["features"]) 
  
  pca = PCA(n_components=feats)
  feats_train_pca = pca.fit_transform(feats_train_select)
  feats_test_pca = pca.transform(feats_test_select)
  
  return feats_train_pca, feats_test_pca
