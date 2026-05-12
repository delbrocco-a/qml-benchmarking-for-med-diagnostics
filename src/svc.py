from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score,
)
from typing import Optional

from src.load_data import FEATURES, TARGETS


LIN     = "linear"
POLY    = "poly"
RBF     = "rbf"
SIGMOID = "sigmoid"

# Grid searched during nested CV — kept small to stay within runtime budget.
# gamma='scale' (1/(n_feats * X.var())) consistently outperforms 'auto' on
# small feature sets; 0.01/0.1/1 cover the low-to-mid range that matters
# for PCA-reduced medical data (2–8 features, unit-variance after PCA).
_SVC_GRID = {
    "C":     [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.1, 1],
}


def tuneSVC(data: dict) -> tuple[dict, float]:
    """3-fold inner CV to select C and gamma for SVC(rbf).

    Returns (best_params, best_inner_cv_balanced_accuracy).
    GridSearchCV uses StratifiedKFold automatically for classifiers.
    """
    grid = GridSearchCV(
        SVC(kernel=RBF),
        param_grid=_SVC_GRID,
        cv=3,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=False,  # we'll re-train with full data ourselves
    )
    grid.fit(data[FEATURES], data[TARGETS])
    return grid.best_params_, float(grid.best_score_)


def trainSVC(
    data:   dict,
    kernel: Optional[str]   = "rbf",
    C:      float           = 1.0,
    gamma:  str | float     = "scale",
) -> SVC:
    """Trains a classical SVC model using training data"""

    svc = SVC(kernel=kernel, C=C, gamma=gamma)
    svc.fit(data[FEATURES], data[TARGETS])

    return svc


def evalSVC(svc: SVC, data: dict) -> tuple[float, ...]:
    """Tests a classical SVC model.

    Returns (accuracy, balanced_accuracy, f1, sensitivity, specificity, auc).
    """
    preds = svc.predict(data[FEATURES])
    acc   = svc.score(data[FEATURES], data[TARGETS])
    bacc  = balanced_accuracy_score(data[TARGETS], preds)
    f1    = f1_score(data[TARGETS], preds, average="binary", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(data[TARGETS], preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        scores = svc.decision_function(data[FEATURES])
        auc = roc_auc_score(data[TARGETS], scores)
    except Exception:
        auc = float("nan")

    return acc, bacc, f1, sens, spec, auc
