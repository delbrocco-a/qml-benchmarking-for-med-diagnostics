from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np

# Classical PCA (reduce to k features)
pca = PCA(n_components=4)  # Keep 4 features
X_reduced = pca.fit_transform(X)

# OR SelectKBest (keep top k features)
selector = SelectKBest(f_classif, k=4)
X_selected = selector.fit_transform(X, y)

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
