"""
Unit tests for benchmark.py — validates that aggregate_folds correctly
collapses per-fold BenchmarkResults into mean ± std summaries.
"""

import numpy as np
import pytest

from src.benchmark import BenchmarkResult, aggregate_folds, Benchmark
from src.CONST import FEATURES, TARGETS


def _make_result(accuracy, model_name="SVC", train_time=1.0, peak_ram_mb=10.0):
    return BenchmarkResult(
        model_name        = model_name,
        train_time        = train_time,
        eval_time         = 0.1,
        accuracy          = accuracy,
        balanced_accuracy = accuracy,
        f1                = accuracy,
        peak_ram_mb       = peak_ram_mb,
    )


def test_aggregate_folds_mean_and_std_are_correct():
    """aggregate_folds must compute exact mean and std over fold accuracies.

    Hand-calculated reference:
      accs = [0.70, 0.80, 0.75, 0.85, 0.90]
      mean = 0.80
      std  = sqrt(((0.1)²+(0)²+(0.05)²+(0.05)²+(0.1)²)/5)
           = sqrt(0.025/5) = sqrt(0.005) ≈ 0.07071
    """
    accs  = [0.70, 0.80, 0.75, 0.85, 0.90]
    folds = [_make_result(a) for a in accs]
    agg   = aggregate_folds(folds)

    assert abs(agg.accuracy          - np.mean(accs)) < 1e-9
    assert abs(agg.accuracy_std      - np.std(accs))  < 1e-9
    assert abs(agg.balanced_accuracy - np.mean(accs)) < 1e-9
    assert abs(agg.f1                - np.mean(accs)) < 1e-9


def test_aggregate_folds_preserves_model_name():
    folds = [_make_result(0.8, model_name="QSVC")] * 3
    agg   = aggregate_folds(folds)
    assert agg.model_name == "QSVC"


def test_aggregate_folds_peak_ram_is_max_not_mean():
    """Peak RAM should be the worst-case across folds, not the average."""
    rams  = [10.0, 50.0, 20.0, 15.0, 30.0]
    folds = [_make_result(0.8, peak_ram_mb=r) for r in rams]
    agg   = aggregate_folds(folds)
    assert agg.peak_ram_mb == max(rams), (
        f"Expected peak RAM={max(rams)}, got {agg.peak_ram_mb}"
    )


def test_aggregate_single_fold_has_zero_std():
    """With one fold, std must be 0."""
    agg = aggregate_folds([_make_result(0.75)])
    assert agg.accuracy_std == 0.0
    assert agg.f1_std       == 0.0


def test_benchmark_run_records_result():
    """Benchmark.run must store a result with correct model name."""
    data = {FEATURES: np.array([[1, 2], [3, 4]]), TARGETS: np.array([0, 1])}

    from src.svc import trainSVC, evalSVC
    bench = Benchmark()
    bench.run(
        model_name = "SVC (test)",
        train_fn   = lambda: trainSVC(data, kernel="linear"),
        eval_fn    = lambda m: evalSVC(m, data),
        train_data = data, test_data = data,
    )

    assert len(bench.results) == 1
    r = bench.results[0]
    assert r.model_name  == "SVC (test)"
    assert 0.0 <= r.accuracy <= 1.0
    assert r.train_time  >= 0
    assert r.peak_ram_mb >= 0
