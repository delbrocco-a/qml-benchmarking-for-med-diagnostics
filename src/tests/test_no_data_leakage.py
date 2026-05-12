"""Tests that PCASelect and ScaleForQuantum are fit only on training data."""
import numpy as np
import pytest

from src.feat_selector import PCASelect, ScaleForQuantum
from src.CONST import TRAINING, TESTING, FEATURES, TARGETS


@pytest.fixture
def shifted_data():
    """Train and test sets with deliberately different means."""
    rng = np.random.default_rng(0)
    data = {
        TRAINING: {
            FEATURES: rng.normal(loc=0.0, scale=1.0, size=(60, 8)),
            TARGETS:  rng.integers(0, 2, size=60),
        },
        TESTING: {
            FEATURES: rng.normal(loc=10.0, scale=1.0, size=(20, 8)),
        },
    }
    return data


def test_pca_fit_on_train_only(shifted_data):
    """PCA fit on train; test projections are NOT re-centred to zero mean."""
    _, test_pca = PCASelect(select=6, feats=3, data=shifted_data)
    # If PCA were fit on test data, its mean would be ~0.
    # Fit on shifted train data leaves test mean far from 0.
    assert np.abs(test_pca.mean()) > 1.0, (
        "Test PCA mean is near zero — PCA may have been re-fit on test data"
    )


def test_scaler_fit_on_train_only(shifted_data):
    """Scaler fit on train; test max can exceed π since test was not used for fitting."""
    train_pca, test_pca = PCASelect(select=6, feats=3, data=shifted_data)
    _, test_scaled = ScaleForQuantum(train_pca, test_pca)
    # Scaler maps train to [0, π]; test values shifted by +10 in original
    # space will project outside that range after PCA, so test max > π.
    assert test_scaled.max() > np.pi, (
        "Test max ≤ π — scaler may have been (re-)fit on test data"
    )
