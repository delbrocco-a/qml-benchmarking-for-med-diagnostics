"""
Integration test: runs a single train/test fold through the full pipeline.

  load_csv -> split_data -> PCASelect -> ScaleForQuantum -> trainSVC -> evalSVC

Uses a synthetic two-class dataset with known structure so the expected
accuracy and output shapes can be asserted deterministically.
"""

import numpy as np
import pandas as pd
import pytest

from src.load_data    import load_csv, split_data
from src.feat_selector import PCASelect, ScaleForQuantum
from src.svc           import trainSVC, evalSVC
from src.log_reg       import trainLogReg, evalLogReg
from src.CONST         import TRAINING, TESTING, FEATURES, TARGETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_csv(tmp_path, n=120, n_feats=6, seed=0):
    """Write a CSV with two Gaussian clusters separated by 6σ.

    The dataset is constructed so that any linear classifier should reach
    ≥90% accuracy — giving us a deterministic lower bound to assert against.
    """
    rng = np.random.default_rng(seed)
    X0  = rng.normal(loc=-3, scale=0.5, size=(n // 2, n_feats))
    X1  = rng.normal(loc=+3, scale=0.5, size=(n // 2, n_feats))
    X   = np.vstack([X0, X1])
    y   = np.array([0] * (n // 2) + [1] * (n // 2))

    cols = [f"f{i}" for i in range(n_feats)] + ["target"]
    df   = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_pipeline_svc_accuracy_above_chance(tmp_path):
    """End-to-end: CSV → load → split → PCA → scale → SVC → evaluate.

    Asserts correctness at every stage boundary and that accuracy exceeds
    random chance (0.5) on a well-separated synthetic dataset.
    """
    path     = _write_synthetic_csv(tmp_path)
    csv, feats = load_csv(path)

    assert isinstance(csv, pd.DataFrame)
    assert len(feats) == 7          # 6 features + target

    data = split_data(csv, target=feats[-1], split=0.2)

    # Feature reduction: SelectKBest(4) → PCA(2)
    train_pca, test_pca = PCASelect(select=4, feats=2, data=data)

    assert train_pca.shape[1] == 2, "PCA must reduce to 2 components"
    assert test_pca.shape[1]  == 2

    # Quantum scaling
    train_q, test_q = ScaleForQuantum(train_pca, test_pca)

    assert train_q.min() >= -1e-9,           "Scaled features must be ≥ 0"
    assert train_q.max() <=  np.pi + 1e-9,  "Scaled features must be ≤ π"
    assert test_q.min()  >= -1e-9
    # Note: test set is transformed by train's scaler so may not reach π

    q_data = {
        TRAINING: {FEATURES: train_q, TARGETS: data[TRAINING][TARGETS]},
        TESTING:  {FEATURES: test_q,  TARGETS: data[TESTING][TARGETS]},
    }

    model = trainSVC(q_data[TRAINING], kernel="linear")
    acc, bacc, f1, sens, spec, auc = evalSVC(model, q_data[TESTING])

    assert acc  > 0.5,  f"SVC accuracy {acc:.2%} not above chance on separable data"
    assert bacc > 0.5
    assert 0.0 <= f1 <= 1.0


def test_full_pipeline_logreg_accuracy_above_chance(tmp_path):
    """Same pipeline as above but with LogReg — confirms module interoperability."""
    path       = _write_synthetic_csv(tmp_path)
    csv, feats = load_csv(path)
    data       = split_data(csv, target=feats[-1], split=0.2)

    train_pca, test_pca = PCASelect(select=4, feats=2, data=data)
    train_q, test_q     = ScaleForQuantum(train_pca, test_pca)

    q_data = {
        TRAINING: {FEATURES: train_q, TARGETS: data[TRAINING][TARGETS]},
        TESTING:  {FEATURES: test_q,  TARGETS: data[TESTING][TARGETS]},
    }

    model = trainLogReg(q_data[TRAINING])
    acc, bacc, f1, sens, spec, auc = evalLogReg(model, q_data[TESTING])

    assert acc  > 0.5
    assert bacc > 0.5


def test_pipeline_output_shapes_are_consistent(tmp_path):
    """Assert that train/test sample counts are preserved through the pipeline."""
    path       = _write_synthetic_csv(tmp_path, n=100)
    csv, feats = load_csv(path)
    data       = split_data(csv, target=feats[-1], split=0.2)

    n_train_expected = data[TRAINING][FEATURES].shape[0]
    n_test_expected  = data[TESTING][FEATURES].shape[0]

    train_pca, test_pca = PCASelect(select=4, feats=2, data=data)

    assert train_pca.shape[0] == n_train_expected
    assert test_pca.shape[0]  == n_test_expected

    train_q, test_q = ScaleForQuantum(train_pca, test_pca)

    assert train_q.shape == train_pca.shape
    assert test_q.shape  == test_pca.shape
