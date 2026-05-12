"""Regression and contract tests for cv_splits."""
import numpy as np
import pandas as pd
import pytest

from src.load_data import cv_splits


def _make_df(n: int = 100, imbalance: float = 0.7, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pos = int(n * imbalance)
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    X = rng.normal(size=(n, 4))
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    return df


def test_same_seed_produces_identical_folds():
    df = _make_df()
    folds_a = list(cv_splits(df, "target", seed=42))
    folds_b = list(cv_splits(df, "target", seed=42))
    for fa, fb in zip(folds_a, folds_b):
        np.testing.assert_array_equal(
            fa["training"]["targets"], fb["training"]["targets"],
            err_msg="Same seed produced different fold assignments"
        )


def test_different_seeds_produce_different_folds():
    df = _make_df()
    folds_a = list(cv_splits(df, "target", seed=1))
    folds_b = list(cv_splits(df, "target", seed=99))
    # Compare features (unique floats), not targets (binary — can match by coincidence).
    any_diff = any(
        not np.array_equal(fa["training"]["features"], fb["training"]["features"])
        for fa, fb in zip(folds_a, folds_b)
    )
    assert any_diff, "Different seeds produced identical folds — splitter is not seeded"


def test_stratification_preserves_class_ratio():
    """Each fold's class ratio must be within 5 pp of the global ratio."""
    df = _make_df(n=200, imbalance=0.65)
    global_ratio = df["target"].mean()
    tolerance = 0.05

    for fold in cv_splits(df, "target", seed=42):
        test_targets = fold["testing"]["targets"]
        fold_ratio = test_targets.mean()
        assert abs(fold_ratio - global_ratio) <= tolerance, (
            f"Fold ratio {fold_ratio:.3f} deviates from global {global_ratio:.3f} "
            f"by more than {tolerance}"
        )
