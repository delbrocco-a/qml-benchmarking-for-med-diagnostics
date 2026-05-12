from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score,
)

from src.load_data import FEATURES, TARGETS


# Inverse regularisation strength — C=1 is the sklearn default, but
# medical datasets with PCA-reduced features often benefit from stronger
# regularisation (small C). The grid covers two decades on each side.
_LR_GRID = {"C": [0.01, 0.1, 1, 10]}


def tuneLogReg(data: dict) -> tuple[dict, float]:
    """3-fold inner CV to select C for LogisticRegression.

    Returns (best_params, best_inner_cv_balanced_accuracy).
    """
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid=_LR_GRID,
        cv=3,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=False,
    )
    grid.fit(data[FEATURES], data[TARGETS])
    return grid.best_params_, float(grid.best_score_)


def trainLogReg(data: dict, C: float = 1.0) -> LogisticRegression:
    """Trains a classical Logistic Regression model using training data"""

    log_reg = LogisticRegression(max_iter=1000, C=C)
    log_reg.fit(data[FEATURES], data[TARGETS])

    return log_reg


def evalLogReg(log_reg: LogisticRegression, data: dict) -> tuple[float, ...]:
    """Tests a classical Logistic Regression model.

    Returns (accuracy, balanced_accuracy, f1, sensitivity, specificity, auc).
    """
    preds = log_reg.predict(data[FEATURES])
    acc   = log_reg.score(data[FEATURES], data[TARGETS])
    bacc  = balanced_accuracy_score(data[TARGETS], preds)
    f1    = f1_score(data[TARGETS], preds, average="binary", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(data[TARGETS], preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        probs = log_reg.predict_proba(data[FEATURES])[:, 1]
        auc   = roc_auc_score(data[TARGETS], probs)
    except Exception:
        auc = float("nan")

    return acc, bacc, f1, sens, spec, auc
