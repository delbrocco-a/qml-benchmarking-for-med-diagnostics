import time
import tracemalloc
import functools
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from src.q_estimate import estimate_kernel_runtime, DEFAULT_SHOTS
from src.load_data import TESTING, FEATURES, TARGETS


@dataclass
class BenchmarkResult:
    """Stores results from a single model training and evaluation run.

    After cross-validation, accuracy/balanced_accuracy/f1/train_time fields
    hold the fold mean; the corresponding *_std fields hold the std deviation.
    """

    model_name:            str
    train_time:            float         # wall-clock seconds (mean over folds)
    eval_time:             float         # wall-clock seconds (mean over folds)
    accuracy:              float         # raw accuracy, 0.0–1.0
    balanced_accuracy:     float = 0.0   # adjusts for class imbalance
    f1:                    float = 0.0   # binary F1 (disease class = positive)
    accuracy_std:          float = 0.0   # std dev across CV folds
    balanced_accuracy_std: float = 0.0
    balanced_accuracy_max: float = 0.0   # best single fold — supplementary, not primary result
    f1_std:                float = 0.0
    sensitivity:           float = 0.0   # TP / (TP + FN) — recall for positive class
    sensitivity_std:       float = 0.0
    sensitivity_max:       float = 0.0
    specificity:           float = 0.0   # TN / (TN + FP)
    specificity_std:       float = 0.0
    specificity_max:       float = 0.0
    auc:                   float = float("nan")  # ROC-AUC; nan if model has no score output
    auc_std:               float = 0.0
    auc_max:               float = float("nan")
    train_time_std:        float = 0.0
    cpu_time_s:            float = 0.0   # CPU time during training (mean)
    peak_ram_mb:           float = 0.0   # peak Python heap during training (max)
    extra:                 dict  = field(default_factory=dict)
    quantum_time_s:        Optional[float] = None
    quantum_time_human:    Optional[str]   = None

    def __str__(self) -> str:
        std = f" ±{self.accuracy_std * 100:.2f}%" if self.accuracy_std > 0 else ""
        bstd = f" ±{self.balanced_accuracy_std * 100:.2f}%" if self.balanced_accuracy_std > 0 else ""
        f1std = f" ±{self.f1_std:.4f}" if self.f1_std > 0 else ""
        lines = [
            f"[{self.model_name}]",
            f"  Accuracy   : {self.accuracy * 100:.2f}%{std}",
            f"  Bal. acc.  : {self.balanced_accuracy * 100:.2f}%{bstd}",
            f"  F1 (binary): {self.f1:.4f}{f1std}",
            f"  Train time : {self.train_time:.4f}s",
            f"  CPU time   : {self.cpu_time_s:.4f}s",
            f"  Peak RAM   : {self.peak_ram_mb:.2f} MB",
            f"  Eval time  : {self.eval_time:.4f}s",
        ]
        if self.extra:
            for k, v in self.extra.items():
                lines.append(f"  {k:<11}: {v}")
        if self.quantum_time_s is not None:
            lines.append(f"  Quantum est.: {self.quantum_time_human}")
        return "\n".join(lines)


class Benchmark:
    """Collects and reports BenchmarkResults across multiple model runs."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []

    def run(
        self,
        model_name: str,
        train_fn:   Callable,
        eval_fn:    Callable,
        train_data: dict,
        test_data:  dict,
        extra:      Optional[dict] = None,
    ) -> BenchmarkResult:
        """
        Trains and evaluates a model, recording timing, CPU usage, and RAM.
        eval_fn must return (accuracy, balanced_accuracy, f1).
        """
        tracemalloc.start()
        cpu_before = time.process_time()
        t0 = time.perf_counter()

        model = train_fn()

        train_time  = time.perf_counter() - t0
        cpu_time    = time.process_time() - cpu_before
        _, peak     = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = peak / (1024 * 1024)

        t1 = time.perf_counter()
        acc, bacc, f1, sens, spec, auc = eval_fn(model)
        eval_time = time.perf_counter() - t1

        result = BenchmarkResult(
            model_name        = model_name,
            train_time        = train_time,
            eval_time         = eval_time,
            accuracy          = acc,
            balanced_accuracy = bacc,
            f1                = f1,
            sensitivity       = sens,
            specificity       = spec,
            auc               = auc,
            cpu_time_s        = cpu_time,
            peak_ram_mb       = peak_ram_mb,
            extra             = extra or {},
        )
        self.results.append(result)
        return result

    def summary(self) -> str:
        """Returns a formatted summary table of all recorded results."""
        if not self.results:
            return "No benchmark results recorded."

        header = (
            f"{'Model':<30} {'Train (s)':>10} {'CPU (s)':>9} {'RAM (MB)':>9} "
            f"{'Accuracy':>10} {'Bal.Acc':>9} {'Best fold':>10} {'F1':>7} {'Quantum est.':>14}"
        )
        sep  = "-" * len(header)
        rows = [header, sep]

        for r in self.results:
            acc_col  = f"{r.accuracy * 100:.2f}%"
            if r.accuracy_std > 0:
                acc_col += f"±{r.accuracy_std * 100:.2f}"
            # Best fold only populated after aggregate_folds; show "-" for single runs.
            best_col = f"{r.balanced_accuracy_max * 100:.2f}%" if r.balanced_accuracy_max > 0 else "-"
            q_col    = r.quantum_time_human if r.quantum_time_human else "N/A"
            rows.append(
                f"{r.model_name:<30} {r.train_time:>10.4f} {r.cpu_time_s:>9.4f} "
                f"{r.peak_ram_mb:>9.2f} {acc_col:>10} "
                f"{r.balanced_accuracy * 100:>8.2f}% {best_col:>10} {r.f1:>7.4f} {q_col:>14}"
            )

        best    = max(self.results, key=lambda r: r.balanced_accuracy)
        fastest = min(self.results, key=lambda r: r.train_time)
        rows += [
            sep,
            f"Best bal.acc : {best.model_name} ({best.balanced_accuracy * 100:.2f}%)",
            f"Fastest train: {fastest.model_name} ({fastest.train_time:.4f}s)",
        ]

        # Medical metrics block — sensitivity, specificity, AUC per model.
        # Only shown when at least one model has sensitivity populated (i.e.
        # after a live run; hardcoded cached results show 0.00% / nan).
        if any(r.sensitivity > 0 or r.specificity > 0 for r in self.results):
            rows.append(f"\n{'Medical metrics (mean ± std  |  best fold)':<44}")
            rows.append(f"  {'Model':<30} {'Sensitivity':>12} {'Specificity':>12} {'AUC':>10}")
            rows.append("  " + "-" * 66)
            for r in self.results:
                s_col  = (f"{r.sensitivity*100:>6.2f}%±{r.sensitivity_std*100:.2f}"
                          f" [{r.sensitivity_max*100:.2f}%]")
                sp_col = (f"{r.specificity*100:>6.2f}%±{r.specificity_std*100:.2f}"
                          f" [{r.specificity_max*100:.2f}%]")
                if not (r.auc != r.auc):  # nan check
                    a_col = f"{r.auc:.3f}±{r.auc_std:.3f} [{r.auc_max:.3f}]"
                else:
                    a_col = "nan"
                rows.append(f"  {r.model_name:<30} {s_col:>22} {sp_col:>22} {a_col:>16}")

        return "\n".join(rows)

    def clear(self):
        """Clears all stored results."""
        self.results = []


def aggregate_folds(fold_results: list[BenchmarkResult]) -> BenchmarkResult:
    """Collapses per-fold BenchmarkResults into a single mean ± std result.

    Used after the CV loop to produce one aggregated result per model per
    (dataset, qubit count) configuration. Peak RAM takes the max across folds
    since it reflects worst-case allocation.
    """
    accs   = np.array([r.accuracy          for r in fold_results])
    baccs  = np.array([r.balanced_accuracy  for r in fold_results])
    f1s    = np.array([r.f1                for r in fold_results])
    trains = np.array([r.train_time        for r in fold_results])
    evals  = np.array([r.eval_time         for r in fold_results])
    cpus   = np.array([r.cpu_time_s        for r in fold_results])
    rams   = np.array([r.peak_ram_mb       for r in fold_results])
    senss  = np.array([r.sensitivity       for r in fold_results])
    specs  = np.array([r.specificity       for r in fold_results])
    aucs   = np.array([r.auc               for r in fold_results], dtype=float)

    rep = fold_results[0]
    return BenchmarkResult(
        model_name             = rep.model_name,
        train_time             = float(np.mean(trains)),
        train_time_std         = float(np.std(trains)),
        eval_time              = float(np.mean(evals)),
        accuracy               = float(np.mean(accs)),
        accuracy_std           = float(np.std(accs)),
        balanced_accuracy      = float(np.mean(baccs)),
        balanced_accuracy_std  = float(np.std(baccs)),
        balanced_accuracy_max  = float(np.max(baccs)),
        f1                     = float(np.mean(f1s)),
        f1_std                 = float(np.std(f1s)),
        sensitivity            = float(np.mean(senss)),
        sensitivity_std        = float(np.std(senss)),
        sensitivity_max        = float(np.max(senss)),
        specificity            = float(np.mean(specs)),
        specificity_std        = float(np.std(specs)),
        specificity_max        = float(np.max(specs)),
        # nanmean/nanstd so a single Pegasos nan doesn't poison the aggregate
        auc                    = float(np.nanmean(aucs)),
        auc_std                = float(np.nanstd(aucs)),
        auc_max                = float(np.nanmax(aucs)),
        cpu_time_s             = float(np.mean(cpus)),
        peak_ram_mb            = float(np.max(rams)),
        extra                  = rep.extra,
        quantum_time_s         = rep.quantum_time_s,
        quantum_time_human     = rep.quantum_time_human,
    )


def timed(label: Optional[str] = None):
    """Decorator that prints execution time for any function."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            t0   = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"[timed] {name}: {elapsed:.4f}s")
            return result
        return wrapper
    return decorator
