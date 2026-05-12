"""
Statistical comparison: quantum models vs classical baselines.

Uses the Wilcoxon signed-rank test on per-fold balanced accuracy.
Paired because each observation pair comes from the same fold — same
data split, same preprocessing — so differences reflect model behaviour
not sampling luck.

Pooling note:
  5-fold Wilcoxon has a minimum two-sided p of 2/2^5 = 0.0625, so
  per-config significance is unreachable by construction. Pooling
  across 3 datasets × 7 qubit counts × 5 folds = 105 pairs gives the
  test enough power. The pooled p-value has a narrow interpretation:
  it tests whether one model is *consistently* higher across all
  configurations we ran, not whether it generalises beyond them. The
  per-dataset breakdowns (n=35) are the primary comparison; the
  pooled number is a headline summary.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.benchmark import BenchmarkResult


# Fixed pairs: quantum (a) vs classical (b). Full baselines excluded —
# they train on more data than the quantum models so a comparison there
# would be meaningless.
PAIRS = [
    ("QSVC",        "SVC (rbf, capped)"),
    ("QSVC",        "LogReg (capped)"),
    ("PegasosQSVC", "SVC (rbf, capped)"),
    ("PegasosQSVC", "LogReg (capped)"),
]

# Bonferroni-corrected thresholds for 4 simultaneous comparisons
ALPHA_STAR  = 0.05 / len(PAIRS)   # 0.0125 -> annotated *
ALPHA_DSTAR = 0.01 / len(PAIRS)   # 0.0025 -> annotated **


def wilcoxon_pairwise(
    all_fold_results: dict[str, dict[int, dict[str, list]]],
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Wilcoxon signed-rank test for each quantum-vs-classical pair.

    Parameters
    ----------
    all_fold_results : dataset -> n_qubits -> model_name -> [BenchmarkResult per fold]
    datasets         : restrict to these datasets; None means use all

    Returns a tidy DataFrame: model_a, model_b, W, p_two_sided, p_less
    (quantum worse), n_pairs, median_diff (a−b), winner.
    """
    if datasets is None:
        datasets = list(all_fold_results.keys())

    rows = []
    for model_a, model_b in PAIRS:
        a_scores, b_scores = [], []

        for ds in datasets:
            for n_q, fold_map in all_fold_results.get(ds, {}).items():
                if model_a not in fold_map or model_b not in fold_map:
                    continue
                fa = [r.balanced_accuracy for r in fold_map[model_a]]
                fb = [r.balanced_accuracy for r in fold_map[model_b]]
                # zip pairs same-fold observations; handles skipped folds
                # because both models are absent from the same fold equally
                for a, b in zip(fa, fb):
                    a_scores.append(a)
                    b_scores.append(b)

        n = len(a_scores)
        if n < 10:
            # not worth testing — shouldn't happen in normal runs
            continue

        diffs = np.array(a_scores) - np.array(b_scores)

        if np.all(diffs == 0):
            # identical performance on every fold; test undefined
            rows.append({
                "model_a": model_a, "model_b": model_b,
                "W": 0.0, "p_two_sided": 1.0, "p_less": 0.5,
                "n_pairs": n, "median_diff": 0.0, "winner": "tie",
            })
            continue

        stat_two, p_two = wilcoxon(diffs, zero_method="wilcox",
                                   alternative="two-sided")
        # p_less: P(quantum is worse than classical)
        _, p_less = wilcoxon(diffs, zero_method="wilcox",
                             alternative="less")

        rows.append({
            "model_a":     model_a,
            "model_b":     model_b,
            "W":           stat_two,
            "p_two_sided": p_two,
            "p_less":      p_less,
            "n_pairs":     n,
            "median_diff": float(np.median(diffs)),
            "winner":      model_a if np.median(diffs) > 0 else model_b,
        })

    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> str:
    """One-line-per-pair console table for end-of-run printing."""
    if df.empty:
        return "  No Wilcoxon results (insufficient data)."

    lines = [
        f"  {'Pair':<40} {'n':>5} {'W':>8} {'p (2-sided)':>12} "
        f"{'med Δ':>8} {'winner':<20}",
        "  " + "-" * 97,
    ]
    for _, r in df.iterrows():
        pair  = f"{r.model_a} vs {r.model_b}"
        stars = ""
        if r.p_two_sided < ALPHA_DSTAR:
            stars = "**"
        elif r.p_two_sided < ALPHA_STAR:
            stars = "*"
        lines.append(
            f"  {pair:<40} {r.n_pairs:>5} {r.W:>8.1f} "
            f"{r.p_two_sided:>12.4f}{stars:<2} "
            f"{r.median_diff * 100:>+7.2f}%  {r.winner:<20}"
        )
    lines += [
        "  " + "-" * 97,
        f"  * p < {ALPHA_STAR:.4f}   ** p < {ALPHA_DSTAR:.4f}   "
        "(Bonferroni-corrected for 4 simultaneous comparisons)",
    ]
    return "\n".join(lines)
