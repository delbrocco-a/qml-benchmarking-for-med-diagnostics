"""
Visualisation suite for the QML benchmark pipeline.

All public functions write to outputs/plots/<prefix>/ so results across
datasets and qubit counts never overwrite each other.

Per-run plots (called from visualise_all):
  plot_class_distribution      - class balance bar chart
  plot_feature_correlation     - Pearson correlation heatmap
  plot_feature_distributions   - per-feature KDE histograms by class
  plot_pca_scatter             - 2D PCA projection coloured by class
  plot_pca_variance            - scree plot with cumulative variance
  plot_model_accuracy          - horizontal accuracy bar chart
  plot_model_metrics           - 5-panel metric comparison (bal. acc, F1, sens, spec, AUC)
  plot_model_timing            - train vs eval time (log scale)
  plot_accuracy_vs_time        - accuracy vs train time scatter
  plot_accuracy_efficiency     - accuracy vs CPU cost, sized by RAM
  plot_model_radar             - spider chart (accuracy / speed / RAM)
  plot_dashboard               - 4-panel summary figure

Per-dataset qubit-sweep plots:
  plot_qubit_sweep_accuracy          - accuracy vs qubit count per model
  plot_qubit_sweep_timing            - training time vs qubit count (log)
  plot_qubit_sweep_quantum_time      - theoretical hardware time vs qubits
  plot_qubit_sweep_resource          - CPU time and peak RAM vs qubits
  plot_classical_vs_quantum          - sim time vs hardware estimate (crossover)

Cross-dataset summary plots:
  plot_cross_dataset_summary         - heatmap: dataset x qubits -> accuracy
  plot_cross_dataset_training_time   - heatmap: dataset x qubits -> train time
  plot_cross_dataset_all_models      - calls summary for every model found
  plot_wilcoxon_heatmap              - 2x2 significance matrix (quantum vs classical)
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from typing import Optional
import os

from src.benchmark import BenchmarkResult

matplotlib.rcParams.update({
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "figure.facecolor":  "#F7F6F2",
    "axes.facecolor":    "#F7F6F2",
})

PALETTE = {
    "blue":   "#185FA5",
    "teal":   "#0F6E56",
    "amber":  "#BA7517",
    "coral":  "#993C1D",
    "purple": "#534AB7",
    "pink":   "#993556",
    "gray":   "#5F5E5A",
    "green":  "#3B6D11",
    "red":    "#A32D2D",
}

ACCENT_COLORS = list(PALETTE.values())
BACKGROUND    = "#F7F6F2"
BORDER        = "#D3D1C7"
TEXT_DARK     = "#2C2C2A"
TEXT_MUTED    = "#888780"

BASE_OUTPUT_DIR = "outputs/plots"


# --- I/O helpers ---

def _output_dir(prefix: str) -> str:
    path = os.path.join(BASE_OUTPUT_DIR, prefix) if prefix else BASE_OUTPUT_DIR
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig: plt.Figure, name: str, prefix: str, show: bool) -> str:
    out  = _output_dir(prefix)
    path = os.path.join(out, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"  Saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _model_colors(results: list) -> list:
    """Consistent color assignment keyed by model name, not position."""
    seen = {}
    out  = []
    for r in results:
        if r.model_name not in seen:
            seen[r.model_name] = ACCENT_COLORS[len(seen) % len(ACCENT_COLORS)]
        out.append(seen[r.model_name])
    return out


# --- Dataset views ---

def plot_class_distribution(
    targets: np.ndarray,
    title:   str  = "Class distribution",
    prefix:  str  = "",
    show:    bool = True,
) -> str:
    """Bar chart of target class counts with percentage labels."""
    classes, counts = np.unique(targets, return_counts=True)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(max(4, len(classes) * 1.2), 4))
    bars = ax.bar(
        [str(c) for c in classes],
        counts,
        color=[ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(classes))],
        width=0.55,
        zorder=2,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8, color=TEXT_MUTED,
        )

    fig.tight_layout()
    return _save(fig, "class_distribution", prefix, show)


def plot_feature_correlation(
    features:      np.ndarray,
    feature_names: Optional[list] = None,
    title:         str  = "Feature correlation matrix",
    prefix:        str  = "",
    show:          bool = True,
) -> str:
    """Pearson correlation heatmap."""
    df   = pd.DataFrame(
        features,
        columns=feature_names if feature_names else [f"F{i}" for i in range(features.shape[1])],
    )
    corr = df.corr()
    n    = len(corr)

    fig, ax = plt.subplots(figsize=(max(5, n * 0.55), max(4, n * 0.5)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    if n <= 20:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.5 else TEXT_DARK)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_linewidth(0.5)
    ax.set_title(title, color=TEXT_DARK, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return _save(fig, "feature_correlation", prefix, show)


def plot_feature_distributions(
    features:      np.ndarray,
    targets:       np.ndarray,
    feature_names: Optional[list] = None,
    max_features:  int  = 12,
    title:         str  = "Feature distributions by class",
    prefix:        str  = "",
    show:          bool = True,
) -> str:
    """KDE histograms per feature, coloured by class."""
    classes = np.unique(targets)
    names   = feature_names if feature_names else [f"F{i}" for i in range(features.shape[1])]
    n_feat  = min(features.shape[1], max_features)
    ncols   = 4
    nrows   = int(np.ceil(n_feat / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.6))
    axes = np.array(axes).flatten()

    for i in range(n_feat):
        ax = axes[i]
        for j, cls in enumerate(classes):
            mask = targets == cls
            ax.hist(features[mask, i], bins=25, alpha=0.55,
                    color=ACCENT_COLORS[j % len(ACCENT_COLORS)],
                    edgecolor="none", density=True)
        ax.set_title(names[i], fontsize=9, color=TEXT_DARK)
        ax.set_xlabel("Value", fontsize=8)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)

    for i in range(n_feat, len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=ACCENT_COLORS[j % len(ACCENT_COLORS)],
                       label=f"Class {cls}", alpha=0.8)
        for j, cls in enumerate(classes)
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9,
               frameon=True, framealpha=0.9, edgecolor=BORDER)
    fig.suptitle(title, fontsize=13, color=TEXT_DARK, y=1.01)
    fig.tight_layout()
    return _save(fig, "feature_distributions", prefix, show)


def plot_pca_scatter(
    features: np.ndarray,
    targets:  np.ndarray,
    title:    str  = "PCA 2D projection",
    prefix:   str  = "",
    show:     bool = True,
) -> str:
    """2D PCA scatter coloured by class."""
    classes = np.unique(targets)
    pca     = PCA(n_components=2)
    proj    = pca.fit_transform(features)
    var     = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    for j, cls in enumerate(classes):
        mask = targets == cls
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=ACCENT_COLORS[j % len(ACCENT_COLORS)],
                   label=f"Class {cls}", alpha=0.65, s=22, linewidths=0)

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.axhline(0, color=BORDER, linewidth=0.5)
    ax.axvline(0, color=BORDER, linewidth=0.5)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor=BORDER)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    return _save(fig, "pca_scatter", prefix, show)


def plot_pca_variance(
    features: np.ndarray,
    title:    str  = "Explained variance (PCA)",
    prefix:   str  = "",
    show:     bool = True,
) -> str:
    """Scree plot: individual and cumulative explained variance."""
    n_comp = min(features.shape[1], 20)
    pca    = PCA(n_components=n_comp)
    pca.fit(features)
    ind = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(ind)

    fig, ax = plt.subplots(figsize=(max(5, n_comp * 0.55), 4))
    ax.bar(range(1, n_comp + 1), ind, color=PALETTE["blue"], alpha=0.75,
           width=0.6, zorder=2, label="Individual")

    ax2 = ax.twinx()
    ax2.plot(range(1, n_comp + 1), cum, color=PALETTE["coral"],
             linewidth=1.8, marker="o", markersize=4, label="Cumulative")
    ax2.axhline(90, color=PALETTE["amber"], linewidth=0.8, linestyle="--")
    ax2.text(n_comp, 90.5, "90%", fontsize=8, color=PALETTE["amber"], ha="right")
    ax2.set_ylabel("Cumulative variance (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.6)

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Individual variance (%)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              frameon=True, framealpha=0.9, edgecolor=BORDER)

    fig.tight_layout()
    return _save(fig, "pca_variance", prefix, show)


# --- Model comparison views ---

def plot_model_accuracy(
    results: list[BenchmarkResult],
    title:   str  = "Model accuracy",
    prefix:  str  = "",
    show:    bool = True,
) -> str:
    """Horizontal bar chart of balanced accuracy per model, sorted best-first.

    Shows balanced accuracy (primary) with CV error bars where available.
    Balanced accuracy is used throughout as it accounts for class imbalance
    in the medical datasets (see benchmark.py).
    """
    names  = [r.model_name              for r in results]
    scores = [r.balanced_accuracy * 100 for r in results]
    errs   = [r.balanced_accuracy_std * 100 for r in results]
    colors = _model_colors(results)

    order  = np.argsort(scores)[::-1]
    names  = [names[i]  for i in order]
    scores = [scores[i] for i in order]
    errs   = [errs[i]   for i in order]
    colors = [colors[i] for i in order]
    has_cv = any(e > 0 for e in errs)

    fig, ax = plt.subplots(figsize=(7, max(3, len(results) * 0.65)))
    bars = ax.barh(names, scores, xerr=errs if has_cv else None,
                   color=colors, height=0.5, zorder=2,
                   error_kw=dict(elinewidth=1.0, ecolor=TEXT_MUTED, capsize=3))
    ax.set_xlim(0, 115)
    ax.set_xlabel("Balanced accuracy (%)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
    ax.axvline(50, color=BORDER, linewidth=0.8, linestyle="--")
    ax.text(50.5, -0.5, "chance", fontsize=7, color=TEXT_MUTED)
    ax.set_axisbelow(True)

    for bar, score, err in zip(bars, scores, errs):
        label = f"{score:.1f}%" + (f" ±{err:.1f}" if has_cv else "")
        ax.text(score + (err if has_cv else 0) + 0.8,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8, color=TEXT_DARK)

    if has_cv:
        ax.text(0.98, 0.02, "Error bars = ±1 SD across CV folds",
                transform=ax.transAxes, fontsize=7, color=TEXT_MUTED, ha="right")

    fig.tight_layout()
    return _save(fig, "model_accuracy", prefix, show)


def plot_model_timing(
    results: list[BenchmarkResult],
    title:   str  = "Model training and evaluation time",
    prefix:  str  = "",
    show:    bool = True,
) -> str:
    """Grouped bar chart: train time vs eval time per model (log scale)."""
    names   = [r.model_name for r in results]
    train_t = [r.train_time for r in results]
    eval_t  = [r.eval_time  for r in results]
    x       = np.arange(len(names))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(max(5, len(results) * 1.8), 4.5))
    ax.bar(x - width / 2, train_t, width, color=PALETTE["blue"],  alpha=0.85,
           label="Train", zorder=2)
    ax.bar(x + width / 2, eval_t,  width, color=PALETTE["teal"], alpha=0.85,
           label="Eval",  zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Time (s, log scale)")
    ax.set_yscale("log")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor=BORDER)

    fig.tight_layout()
    return _save(fig, "model_timing", prefix, show)


def plot_accuracy_vs_time(
    results: list[BenchmarkResult],
    title:   str  = "Accuracy vs training time",
    prefix:  str  = "",
    show:    bool = True,
) -> str:
    """Scatter: x = train_time (log), y = accuracy, bubble size = eval time."""
    accs   = np.array([r.accuracy * 100 for r in results])
    trains = np.array([r.train_time     for r in results])
    evals  = np.array([r.eval_time      for r in results])
    names  = [r.model_name for r in results]
    colors = _model_colors(results)

    sizes = 60 + 400 * (evals - evals.min()) / (evals.max() - evals.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (x, y, s, name, col) in enumerate(zip(trains, accs, sizes, names, colors)):
        ax.scatter(x, y, s=s, c=col, alpha=0.8, linewidths=0.5,
                   edgecolors=TEXT_DARK, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=TEXT_DARK)

    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.text(0.98, 0.03, "Bubble size = eval time",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, ha="right")

    fig.tight_layout()
    return _save(fig, "accuracy_vs_time", prefix, show)


def plot_accuracy_efficiency(
    results: list[BenchmarkResult],
    title:   str  = "Accuracy vs CPU cost",
    prefix:  str  = "",
    show:    bool = True,
) -> str:
    """Scatter: x = CPU time (log), y = accuracy, bubble size = peak RAM.

    Shows which model gives the best accuracy per unit of CPU time.
    Relevant to the computational cost / environmental angle of the dissertation.
    """
    accs    = np.array([r.accuracy * 100  for r in results])
    cpu     = np.array([r.cpu_time_s      for r in results])
    ram     = np.array([r.peak_ram_mb     for r in results])
    names   = [r.model_name for r in results]
    colors  = _model_colors(results)

    # Bubble size proportional to RAM, with a minimum so tiny values are visible
    ram_range = ram.max() - ram.min() + 1e-6
    sizes = 60 + 500 * (ram - ram.min()) / ram_range

    fig, ax = plt.subplots(figsize=(7, 5))
    for x, y, s, name, col in zip(cpu, accs, sizes, names, colors):
        ax.scatter(x, y, s=s, c=col, alpha=0.8, linewidths=0.5,
                   edgecolors=TEXT_DARK, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=TEXT_DARK)

    ax.set_xscale("log")
    ax.set_xlabel("CPU time (s, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.text(0.98, 0.03, "Bubble size = peak RAM (MB)",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, ha="right")
    ax.text(0.02, 0.97, "Top-left = best efficiency",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, va="top")

    fig.tight_layout()
    return _save(fig, "accuracy_efficiency", prefix, show)


def plot_model_radar(
    results: list[BenchmarkResult],
    title:   str  = "Model comparison (radar)",
    prefix:  str  = "",
    show:    bool = True,
) -> str:
    """Spider chart: accuracy, train speed, eval speed, RAM efficiency."""
    def _norm(values):
        mn, mx = min(values), max(values)
        return [(v - mn) / (mx - mn + 1e-12) for v in values]

    accs      = [r.accuracy              for r in results]
    inv_train = [1 / (r.train_time + 1e-9) for r in results]
    inv_eval  = [1 / (r.eval_time  + 1e-9) for r in results]
    inv_ram   = [1 / (r.peak_ram_mb + 1e-6) for r in results]

    n_acc    = _norm(accs)
    n_train  = _norm(inv_train)
    n_eval   = _norm(inv_eval)
    n_ram    = _norm(inv_ram)

    categories = ["Accuracy", "Train speed", "Eval speed", "RAM eff."]
    N          = len(categories)
    angles     = [n / N * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.5)
    ax.xaxis.grid(True, color=BORDER, linewidth=0.5)
    ax.set_facecolor(BACKGROUND)

    colors = _model_colors(results)
    for i, r in enumerate(results):
        vals  = [n_acc[i], n_train[i], n_eval[i], n_ram[i], n_acc[i]]
        col   = colors[i]
        ax.plot(angles, vals, linewidth=1.5, color=col)
        ax.fill(angles, vals, color=col, alpha=0.12)

    ax.set_title(title, color=TEXT_DARK, pad=20)
    legend_handles = [
        mpatches.Patch(color=colors[i], label=r.model_name, alpha=0.8)
        for i, r in enumerate(results)
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.4, 1.1), fontsize=9,
              frameon=True, framealpha=0.9, edgecolor=BORDER)

    fig.tight_layout()
    return _save(fig, "model_radar", prefix, show)


def plot_model_metrics(
    results: list[BenchmarkResult],
    title:   str  = "Model performance metrics",
    prefix:  str  = "",
    show:    bool = True,
) -> str:
    """5-panel horizontal bar chart: one subplot per metric, all models shown.

    Metrics: balanced accuracy, F1, sensitivity, specificity, AUC.
    Models sorted by balanced accuracy descending. Error bars show ±1 SD
    across CV folds where available. AUC shown as n/a if model lacks a
    decision function (e.g. some Qiskit versions of PegasosQSVC).
    """
    METRICS = [
        ("balanced_accuracy", "balanced_accuracy_std", "Balanced\nAccuracy"),
        ("f1",                "f1_std",                "F1"),
        ("sensitivity",       "sensitivity_std",       "Sensitivity"),
        ("specificity",       "specificity_std",       "Specificity"),
        ("auc",               "auc_std",               "AUC"),
    ]

    names    = [r.model_name for r in results]
    colors   = _model_colors(results)
    order    = np.argsort([r.balanced_accuracy for r in results])[::-1]
    s_names   = [names[i]   for i in order]
    s_colors  = [colors[i]  for i in order]
    s_results = [results[i] for i in order]
    n         = len(results)

    fig, axes = plt.subplots(
        1, len(METRICS),
        figsize=(3.2 * len(METRICS), max(3, n * 0.65)),
        sharey=True,
    )

    for ax, (attr, std_attr, label) in zip(axes, METRICS):
        for yi, (r, col) in enumerate(zip(s_results, s_colors)):
            raw = getattr(r, attr, float("nan"))
            v   = float("nan") if (raw is None or np.isnan(raw)) else raw * 100
            std_raw = getattr(r, std_attr, 0.0)
            std = 0.0 if (std_raw is None or np.isnan(std_raw)) else std_raw * 100

            if np.isnan(v):
                ax.barh(yi, 0, height=0.5, color=PALETTE["gray"],
                        alpha=0.3, zorder=2)
                ax.text(1.5, yi, "n/a", va="center", fontsize=7, color=TEXT_MUTED)
            else:
                ax.barh(
                    yi, v, height=0.5, color=col, alpha=0.82, zorder=2,
                    xerr=std if std > 0 else None,
                    error_kw=dict(elinewidth=1.0, ecolor=TEXT_MUTED, capsize=2),
                )
                ax.text(v + (std if std > 0 else 0) + 0.8, yi,
                        f"{v:.1f}", va="center", fontsize=7, color=TEXT_DARK)

        ax.set_xlim(0, 118)
        ax.set_title(label, color=TEXT_DARK, fontsize=10, pad=6)
        ax.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
        ax.axvline(50, color=BORDER, linewidth=0.6, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlabel("Score (%)")

    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(s_names, fontsize=8)

    if any(any(getattr(r, s, 0) or 0 > 0 for _, s, _ in METRICS) for r in results):
        fig.text(0.98, 0.01, "Error bars = ±1 SD across CV folds",
                 ha="right", fontsize=7, color=TEXT_MUTED)

    fig.suptitle(title, fontsize=12, color=TEXT_DARK, y=1.02)
    fig.tight_layout()
    return _save(fig, "model_metrics", prefix, show)


# --- Qubit sweep views ---

def plot_qubit_sweep_accuracy(
    sweep_results: dict[int, list[BenchmarkResult]],
    title:         str  = "Balanced accuracy vs qubit count",
    prefix:        str  = "",
    show:          bool = False,
) -> str:
    """Line chart of balanced accuracy vs qubit count, one line per model.

    Shaded bands show ±1 SD across CV folds where available. Balanced
    accuracy is the primary metric; raw accuracy is in the summary table.
    """
    qubit_counts = sorted(sweep_results.keys())

    model_names: list[str] = []
    for res_list in sweep_results.values():
        for r in res_list:
            if r.model_name not in model_names:
                model_names.append(r.model_name)

    fig, ax = plt.subplots(figsize=(9, 5))
    has_cv = False

    for i, name in enumerate(model_names):
        accs, stds = [], []
        for q in qubit_counts:
            match = next((r for r in sweep_results[q] if r.model_name == name), None)
            accs.append(match.balanced_accuracy * 100 if match else np.nan)
            stds.append(match.balanced_accuracy_std * 100 if match else 0.0)

        accs_arr = np.array(accs)
        stds_arr = np.array(stds)
        col      = ACCENT_COLORS[i % len(ACCENT_COLORS)]

        ax.plot(qubit_counts, accs_arr, marker="o", linewidth=1.8, markersize=5,
                color=col, label=name)

        if np.any(stds_arr > 0):
            has_cv = True
            ax.fill_between(qubit_counts,
                            accs_arr - stds_arr, accs_arr + stds_arr,
                            color=col, alpha=0.12)

        last_valid = next(
            (accs_arr[j] for j in range(len(accs_arr) - 1, -1, -1)
             if not np.isnan(accs_arr[j])), None
        )
        if last_valid is not None:
            ax.annotate(f"{last_valid:.1f}%", (qubit_counts[-1], last_valid),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=7, color=col)

    ax.set_xlabel("Qubits (= PCA features)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.set_xticks(qubit_counts)
    ax.set_ylim(0, 108)
    ax.axhline(50, color=BORDER, linewidth=0.6, linestyle="--")
    ax.text(qubit_counts[0], 51, "chance level", fontsize=7, color=TEXT_MUTED)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor=BORDER)

    if has_cv:
        ax.text(0.98, 0.02, "Shaded bands = ±1 SD across CV folds",
                transform=ax.transAxes, fontsize=7, color=TEXT_MUTED, ha="right")

    fig.tight_layout()
    return _save(fig, "qubit_sweep_accuracy", prefix, show)


def plot_qubit_sweep_timing(
    sweep_results: dict[int, list[BenchmarkResult]],
    title:         str  = "Training time vs qubit count",
    prefix:        str  = "",
    show:          bool = False,
) -> str:
    """Log-scale line chart of training time vs qubit count per model."""
    qubit_counts = sorted(sweep_results.keys())

    model_names: list[str] = []
    for res_list in sweep_results.values():
        for r in res_list:
            if r.model_name not in model_names:
                model_names.append(r.model_name)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, name in enumerate(model_names):
        times = []
        for q in qubit_counts:
            match = next((r for r in sweep_results[q] if r.model_name == name), None)
            times.append(match.train_time if match else np.nan)

        col = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        ls  = "--" if "QSVC" in name else "-"
        ax.plot(qubit_counts, times, marker="o", linewidth=1.8, markersize=5,
                color=col, label=name, linestyle=ls)

    ax.set_xlabel("Qubits (= PCA features)")
    ax.set_ylabel("Training time (s, log scale)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.set_xticks(qubit_counts)
    ax.set_yscale("log")
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor=BORDER)
    ax.text(0.98, 0.03, "Quantum models shown dashed",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, ha="right")

    fig.tight_layout()
    return _save(fig, "qubit_sweep_timing", prefix, show)


def plot_qubit_sweep_quantum_time(
    sweep_results: dict[int, list[BenchmarkResult]],
    title:         str  = "Estimated quantum hardware time vs qubits",
    prefix:        str  = "",
    show:          bool = False,
) -> str:
    """Theoretical hardware runtime (IBM Eagle) vs qubit count for QSVC."""
    qubit_counts = sorted(sweep_results.keys())

    times, valid_qs = [], []
    for q in qubit_counts:
        match = next(
            (r for r in sweep_results[q]
             if r.model_name == "QSVC" and getattr(r, "quantum_time_s", None) is not None),
            None,
        )
        if match:
            times.append(match.quantum_time_s)
            valid_qs.append(q)

    if not valid_qs:
        return ""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(valid_qs, times, marker="o", linewidth=2.0, markersize=6,
            color=PALETTE["purple"], label="QSVC (theoretical)")

    for q, t in zip(valid_qs, times):
        label = (f"{t:.1f}s" if t < 60
                 else f"{t/60:.1f}m" if t < 3600
                 else f"{t/3600:.1f}h")
        ax.annotate(label, (q, t), textcoords="offset points",
                    xytext=(0, 8), fontsize=7, color=PALETTE["purple"], ha="center")

    ax.set_xlabel("Qubits")
    ax.set_ylabel("Estimated hardware time (s, log scale)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.set_xticks(valid_qs)
    ax.set_yscale("log")
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor=BORDER)
    ax.text(0.98, 0.03, "IBM Eagle r3 — theoretical lower bound, excludes queue time",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, ha="right")

    fig.tight_layout()
    return _save(fig, "qubit_sweep_quantum_time", prefix, show)


def plot_qubit_sweep_resource(
    sweep_results: dict[int, list[BenchmarkResult]],
    title:         str  = "CPU time and peak RAM vs qubit count",
    prefix:        str  = "",
    show:          bool = False,
) -> str:
    """Two-panel plot: CPU time (log) and peak RAM vs qubit count per model.

    Supports the environmental and resource cost angle of the dissertation.
    CPU time reflects actual processor work (user + system), not wall clock.
    Peak RAM is Python heap allocation during training only.
    """
    qubit_counts = sorted(sweep_results.keys())

    model_names: list[str] = []
    for res_list in sweep_results.values():
        for r in res_list:
            if r.model_name not in model_names:
                model_names.append(r.model_name)

    fig, (ax_cpu, ax_ram) = plt.subplots(1, 2, figsize=(13, 5))

    for i, name in enumerate(model_names):
        cpu_vals, ram_vals = [], []
        for q in qubit_counts:
            match = next((r for r in sweep_results[q] if r.model_name == name), None)
            cpu_vals.append(match.cpu_time_s  if match else np.nan)
            ram_vals.append(match.peak_ram_mb if match else np.nan)

        col = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        ls  = "--" if "QSVC" in name else "-"
        ax_cpu.plot(qubit_counts, cpu_vals, marker="o", linewidth=1.8,
                    markersize=5, color=col, label=name, linestyle=ls)
        ax_ram.plot(qubit_counts, ram_vals, marker="s", linewidth=1.8,
                    markersize=5, color=col, label=name, linestyle=ls)

    for ax, ylabel, fname_part in [
        (ax_cpu, "CPU time (s, log scale)", "cpu"),
        (ax_ram, "Peak RAM (MB)",           "ram"),
    ]:
        ax.set_xlabel("Qubits (= PCA features)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(qubit_counts)
        ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor=BORDER)

    ax_cpu.set_yscale("log")
    ax_cpu.set_title("CPU time vs qubits", color=TEXT_DARK, pad=10)
    ax_cpu.text(0.98, 0.03, "Quantum models shown dashed",
                transform=ax_cpu.transAxes, fontsize=8, color=TEXT_MUTED, ha="right")
    ax_ram.set_title("Peak RAM vs qubits", color=TEXT_DARK, pad=10)
    ax_ram.text(0.98, 0.97, "Python heap only (excludes C extensions)",
                transform=ax_ram.transAxes, fontsize=7, color=TEXT_MUTED,
                ha="right", va="top")

    fig.suptitle(title, fontsize=12, color=TEXT_DARK)
    fig.tight_layout()
    return _save(fig, "qubit_sweep_resource", prefix, show)


def plot_classical_vs_quantum(
    sweep_results: dict[int, list[BenchmarkResult]],
    title:         str  = "Classical simulation vs quantum hardware time",
    prefix:        str  = "",
    show:          bool = False,
) -> str:
    """Log-scale plot comparing classical simulation time against the
    theoretical quantum hardware runtime across qubit counts.

    This is the key crossover figure for the dissertation: it shows at what
    qubit count (if any) quantum hardware would become faster than classical
    simulation, and how the gap evolves with circuit complexity.

    Classical line: actual wall-clock training time of QSVC on this machine.
    Quantum line: theoretical lower-bound runtime on IBM Eagle r3.
    """
    qubit_counts = sorted(sweep_results.keys())

    classical, quantum, valid_qs = [], [], []
    for q in qubit_counts:
        match = next(
            (r for r in sweep_results[q]
             if r.model_name == "QSVC" and getattr(r, "quantum_time_s", None) is not None),
            None,
        )
        if match:
            classical.append(match.train_time)
            quantum.append(match.quantum_time_s)
            valid_qs.append(q)

    if not valid_qs:
        return ""

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(valid_qs, classical, marker="o", linewidth=2.0, markersize=6,
            color=PALETTE["blue"], label="Classical simulation (this machine)")
    ax.plot(valid_qs, quantum, marker="^", linewidth=2.0, markersize=6,
            color=PALETTE["purple"], linestyle="--",
            label="Quantum hardware estimate (IBM Eagle r3)")

    # Shade the region where quantum is faster (if it occurs)
    classical_arr = np.array(classical)
    quantum_arr   = np.array(quantum)
    q_arr         = np.array(valid_qs)
    if np.any(quantum_arr < classical_arr):
        ax.fill_between(q_arr, classical_arr, quantum_arr,
                        where=(quantum_arr < classical_arr),
                        alpha=0.12, color=PALETTE["teal"],
                        label="Quantum advantage region")

    ax.set_xlabel("Qubits (= PCA features)")
    ax.set_ylabel("Time (s, log scale)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.set_xticks(valid_qs)
    ax.set_yscale("log")
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor=BORDER)
    ax.text(0.02, 0.03,
            "Quantum estimate: lower bound only, excludes queue time and error mitigation",
            transform=ax.transAxes, fontsize=7, color=TEXT_MUTED)

    fig.tight_layout()
    return _save(fig, "classical_vs_quantum", prefix, show)


# --- Cross-dataset summary ---

def plot_cross_dataset_summary(
    all_results:  dict[str, dict[int, list[BenchmarkResult]]],
    model_filter: str  = "QSVC",
    title:        str  = "QSVC accuracy: dataset x qubits",
    prefix:       str  = "summary",
    show:         bool = False,
) -> str:
    """Heatmap: rows = datasets, columns = qubit counts, cells = accuracy."""
    datasets     = list(all_results.keys())
    qubit_counts = sorted({q for d in all_results.values() for q in d})

    grid = np.full((len(datasets), len(qubit_counts)), np.nan)
    for di, ds in enumerate(datasets):
        for qi, q in enumerate(qubit_counts):
            match = next(
                (r for r in all_results[ds].get(q, []) if model_filter in r.model_name),
                None,
            )
            if match:
                grid[di, qi] = match.accuracy * 100

    fig, ax = plt.subplots(
        figsize=(max(6, len(qubit_counts) * 0.9), max(3, len(datasets) * 0.7))
    )
    im = ax.imshow(grid, cmap="YlGn", vmin=40, vmax=100, aspect="auto")

    ax.set_xticks(range(len(qubit_counts)))
    ax.set_xticklabels([str(q) for q in qubit_counts], fontsize=9)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlabel("Qubits")
    ax.set_title(title, color=TEXT_DARK, pad=10)

    for di in range(len(datasets)):
        for qi in range(len(qubit_counts)):
            v = grid[di, qi]
            if not np.isnan(v):
                ax.text(qi, di, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 75 else TEXT_DARK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=9)
    cbar.outline.set_linewidth(0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    slug = model_filter.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return _save(fig, f"cross_dataset_{slug}", prefix, show)


def plot_cross_dataset_training_time(
    all_results:  dict[str, dict[int, list[BenchmarkResult]]],
    model_filter: str  = "QSVC",
    title:        str  = "QSVC training time: dataset x qubits",
    prefix:       str  = "summary",
    show:         bool = False,
) -> str:
    """Heatmap of training time (log seconds): rows = datasets, cols = qubits.

    Complements the accuracy heatmap — shows the computational cost side
    of the crossover question.
    """
    datasets     = list(all_results.keys())
    qubit_counts = sorted({q for d in all_results.values() for q in d})

    grid = np.full((len(datasets), len(qubit_counts)), np.nan)
    for di, ds in enumerate(datasets):
        for qi, q in enumerate(qubit_counts):
            match = next(
                (r for r in all_results[ds].get(q, []) if model_filter in r.model_name),
                None,
            )
            if match:
                grid[di, qi] = match.train_time

    # Log-transform for display so small and large values are both readable
    log_grid = np.where(np.isnan(grid), np.nan, np.log10(np.clip(grid, 1e-6, None)))

    fig, ax = plt.subplots(
        figsize=(max(6, len(qubit_counts) * 0.9), max(3, len(datasets) * 0.7))
    )
    im = ax.imshow(log_grid, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(qubit_counts)))
    ax.set_xticklabels([str(q) for q in qubit_counts], fontsize=9)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlabel("Qubits")
    ax.set_title(title, color=TEXT_DARK, pad=10)

    for di in range(len(datasets)):
        for qi in range(len(qubit_counts)):
            v = grid[di, qi]
            if not np.isnan(v):
                label = f"{v:.1f}s" if v < 60 else f"{v/60:.1f}m"
                lv    = log_grid[di, qi]
                ax.text(qi, di, label, ha="center", va="center",
                        fontsize=7, color="white" if lv > 1.5 else TEXT_DARK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10(train time / s)", fontsize=9)
    cbar.outline.set_linewidth(0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    slug = model_filter.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return _save(fig, f"cross_dataset_time_{slug}", prefix, show)


def plot_cross_dataset_all_models(
    all_results: dict[str, dict[int, list[BenchmarkResult]]],
    title:       str  = "All models: accuracy by dataset and qubit count",
    prefix:      str  = "summary",
    show:        bool = False,
) -> list[str]:
    """Accuracy and training time heatmaps for every model found in results."""
    model_names: list[str] = []
    for ds_data in all_results.values():
        for res_list in ds_data.values():
            for r in res_list:
                if r.model_name not in model_names:
                    model_names.append(r.model_name)

    paths = []
    for name in model_names:
        paths.append(plot_cross_dataset_summary(
            all_results,
            model_filter=name,
            title=f"{name} accuracy: dataset x qubits",
            prefix=prefix,
            show=show,
        ))
        paths.append(plot_cross_dataset_training_time(
            all_results,
            model_filter=name,
            title=f"{name} training time: dataset x qubits",
            prefix=prefix,
            show=show,
        ))
    return paths


# --- Dashboard ---

def plot_dashboard(
    results:  list[BenchmarkResult],
    features: np.ndarray,
    targets:  np.ndarray,
    title:    str  = "ML Benchmark dashboard",
    prefix:   str  = "",
    show:     bool = True,
) -> str:
    """4-panel summary: class distribution, PCA scatter, accuracy, timing."""
    classes, counts = np.unique(targets, return_counts=True)
    pca  = PCA(n_components=2)
    proj = pca.fit_transform(features)
    var  = pca.explained_variance_ratio_ * 100

    names  = [r.model_name for r in results]
    accs   = [r.accuracy * 100 for r in results]
    trains = [r.train_time     for r in results]
    evals  = [r.eval_time      for r in results]
    colors = _model_colors(results)

    fig = plt.figure(figsize=(14, 9), facecolor=BACKGROUND)
    gs  = GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.38)

    ax_cls  = fig.add_subplot(gs[0, 0])
    ax_pca  = fig.add_subplot(gs[0, 1:3])
    ax_acc  = fig.add_subplot(gs[1, 0:2])
    ax_time = fig.add_subplot(gs[1, 2:4])

    # Class distribution
    ax_cls.bar([str(c) for c in classes], counts,
               color=[ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(classes))],
               width=0.5, zorder=2)
    ax_cls.set_title("Class distribution", fontsize=10, color=TEXT_DARK)
    ax_cls.set_xlabel("Class"); ax_cls.set_ylabel("Count")
    ax_cls.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
    ax_cls.set_axisbelow(True)

    # PCA scatter
    for j, cls in enumerate(classes):
        mask = targets == cls
        ax_pca.scatter(proj[mask, 0], proj[mask, 1],
                       c=ACCENT_COLORS[j % len(ACCENT_COLORS)],
                       label=f"Class {cls}", alpha=0.6, s=18, linewidths=0)
    ax_pca.set_title("PCA 2D projection", fontsize=10, color=TEXT_DARK)
    ax_pca.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax_pca.axhline(0, color=BORDER, lw=0.4); ax_pca.axvline(0, color=BORDER, lw=0.4)
    ax_pca.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor=BORDER)
    ax_pca.grid(True, color=BORDER, linewidth=0.3, alpha=0.5)

    # Model accuracy
    order    = np.argsort(accs)[::-1]
    s_names  = [names[i]   for i in order]
    s_accs   = [accs[i]    for i in order]
    s_colors = [colors[i]  for i in order]
    ax_acc.barh(s_names, s_accs, color=s_colors, height=0.45, zorder=2)
    ax_acc.set_xlim(0, 108)
    ax_acc.set_xlabel("Accuracy (%)")
    ax_acc.set_title("Model accuracy", fontsize=10, color=TEXT_DARK)
    ax_acc.xaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
    ax_acc.axvline(50, color=BORDER, linewidth=0.8, linestyle="--")
    ax_acc.set_axisbelow(True)
    for bar, score in zip(ax_acc.patches, s_accs):
        ax_acc.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va="center", fontsize=8, color=TEXT_DARK)

    # Timing (log scale so classical and quantum are both visible)
    x = np.arange(len(names))
    w = 0.35
    ax_time.bar(x - w / 2, trains, w, color=PALETTE["blue"],  alpha=0.85,
                label="Train", zorder=2)
    ax_time.bar(x + w / 2, evals,  w, color=PALETTE["teal"], alpha=0.85,
                label="Eval",  zorder=2)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax_time.set_ylabel("Time (s, log scale)")
    ax_time.set_yscale("log")
    ax_time.set_title("Train vs eval time", fontsize=10, color=TEXT_DARK)
    ax_time.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
    ax_time.set_axisbelow(True)
    ax_time.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor=BORDER)

    fig.suptitle(title, fontsize=14, color=TEXT_DARK, y=1.01)
    return _save(fig, "dashboard", prefix, show)


# --- Statistical summary ---

def plot_wilcoxon_heatmap(
    df:     "pd.DataFrame",
    title:  str  = "Quantum vs classical: median balanced accuracy difference",
    prefix: str  = "summary",
    show:   bool = False,
) -> str:
    """2×2 heatmap: rows = quantum models, cols = classical baselines.

    Cell value is the median balanced accuracy difference (quantum minus
    classical) across all pooled fold observations. Significance stars are
    Bonferroni-corrected for the 4 simultaneous comparisons:
      *  p < 0.0125  (alpha 0.05 / 4)
      ** p < 0.0025  (alpha 0.01 / 4)
    """
    from src.stats import ALPHA_STAR, ALPHA_DSTAR

    quantum_models   = ["QSVC", "PegasosQSVC"]
    classical_models = ["SVC (rbf, capped)", "LogReg (capped)"]
    n_q, n_c         = len(quantum_models), len(classical_models)

    grid_diff = np.full((n_q, n_c), np.nan)
    grid_p    = np.full((n_q, n_c), np.nan)

    for row in df.itertuples():
        if row.model_a in quantum_models and row.model_b in classical_models:
            qi = quantum_models.index(row.model_a)
            ci = classical_models.index(row.model_b)
            grid_diff[qi, ci] = row.median_diff
            grid_p[qi, ci]    = row.p_two_sided

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    # Centre colormap on zero so green = quantum better, red = worse
    vmax = max(np.nanmax(np.abs(grid_diff)), 0.01)
    im   = ax.imshow(grid_diff, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                     aspect="auto")

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(classical_models, fontsize=9)
    ax.set_yticks(range(n_q))
    ax.set_yticklabels(quantum_models, fontsize=9)
    ax.set_xlabel("Classical baseline", fontsize=9)
    ax.set_ylabel("Quantum model", fontsize=9)
    ax.set_title(title, color=TEXT_DARK, pad=10)

    for qi in range(n_q):
        for ci in range(n_c):
            d = grid_diff[qi, ci]
            p = grid_p[qi, ci]
            if np.isnan(d):
                continue
            stars = ""
            if p < ALPHA_DSTAR:
                stars = "**"
            elif p < ALPHA_STAR:
                stars = "*"
            label      = f"{d * 100:+.1f}%"
            text_color = "white" if abs(d) > vmax * 0.55 else TEXT_DARK
            ax.text(ci, qi, f"{label}\n{stars}", ha="center", va="center",
                    fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Median Δ balanced accuracy", fontsize=8)
    cbar.outline.set_linewidth(0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return _save(fig, "wilcoxon_heatmap", prefix, show)


# --- Convenience wrapper ---

def visualise_all(
    results:       list[BenchmarkResult],
    features:      np.ndarray,
    targets:       np.ndarray,
    feature_names: Optional[list] = None,
    dataset_name:  str            = "",
    prefix:        str            = "",
    show:          bool           = False,
) -> list[str]:
    """Generate and save all per-run plots. Returns list of saved paths."""
    ds    = f" [{dataset_name}]" if dataset_name else ""
    paths = []

    print(f"  Rendering dataset views{ds}...")
    paths.append(plot_class_distribution(
        targets, title=f"Class distribution{ds}", prefix=prefix, show=show))
    paths.append(plot_pca_scatter(
        features, targets, title=f"PCA 2D projection{ds}", prefix=prefix, show=show))
    paths.append(plot_pca_variance(
        features, title=f"Explained variance{ds}", prefix=prefix, show=show))

    if features.shape[1] <= 50:
        paths.append(plot_feature_distributions(
            features, targets, feature_names=feature_names,
            title=f"Feature distributions by class{ds}", prefix=prefix, show=show))
    if features.shape[1] <= 30:
        paths.append(plot_feature_correlation(
            features, feature_names=feature_names,
            title=f"Feature correlation{ds}", prefix=prefix, show=show))

    print(f"  Rendering model comparison views{ds}...")
    paths.append(plot_model_accuracy(
        results, title=f"Model accuracy{ds}", prefix=prefix, show=show))
    paths.append(plot_model_metrics(
        results, title=f"Model performance metrics{ds}", prefix=prefix, show=show))
    paths.append(plot_model_timing(
        results, title=f"Training and evaluation time{ds}", prefix=prefix, show=show))
    paths.append(plot_accuracy_vs_time(
        results, title=f"Accuracy vs training time{ds}", prefix=prefix, show=show))
    paths.append(plot_accuracy_efficiency(
        results, title=f"Accuracy vs CPU cost{ds}", prefix=prefix, show=show))

    if len(results) >= 2:
        paths.append(plot_model_radar(
            results, title=f"Model comparison radar{ds}", prefix=prefix, show=show))

    print(f"  Rendering dashboard{ds}...")
    paths.append(plot_dashboard(
        results, features, targets,
        title=f"ML Benchmark dashboard{ds}", prefix=prefix, show=show))

    return paths
