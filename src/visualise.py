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
  "font.family": "monospace",
  "axes.spines.top": False,
  "axes.spines.right": False,
  "axes.linewidth": 0.6,
  "xtick.major.width": 0.6,
  "ytick.major.width": 0.6,
  "xtick.labelsize": 9,
  "ytick.labelsize": 9,
  "axes.labelsize": 10,
  "axes.titlesize": 11,
  "figure.facecolor": "#F7F6F2",
  "axes.facecolor": "#F7F6F2",
})

# ── Palette ──────────────────────────────────────────────────────────────────

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

BACKGROUND  = "#F7F6F2"
BORDER      = "#D3D1C7"
TEXT_DARK   = "#2C2C2A"
TEXT_MUTED  = "#888780"

OUTPUT_DIR = "outputs/plots"


def _ensure_output_dir() -> str:
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  return OUTPUT_DIR


def _save(fig: plt.Figure, name: str, show: bool) -> str:
  path = os.path.join(_ensure_output_dir(), f"{name}.png")
  fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
  print(f"  Saved → {path}")
  if show:
      plt.show()
  plt.close(fig)
  return path


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET VIEWS
# ═══════════════════════════════════════════════════════════════════════════


def plot_class_distribution(
  targets: np.ndarray,
  title: str = "Class distribution",
  show: bool = True,
) -> str:
  """Bar chart of target class counts."""
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
      ha="center", va="bottom",
      fontsize=8, color=TEXT_MUTED,
    )

  fig.tight_layout()
  return _save(fig, "class_distribution", show)


def plot_feature_correlation(
  features: np.ndarray,
  feature_names: Optional[list] = None,
  title: str = "Feature correlation matrix",
  show: bool = True,
) -> str:
  """Heatmap of Pearson correlations between features."""
  df = pd.DataFrame(
    features,
    columns=feature_names if feature_names 
    else [f"F{i}" for i in range(features.shape[1])],
  )
  corr = df.corr()
  n = len(corr)

  fig, ax = plt.subplots(figsize=(max(5, n * 0.55), max(4, n * 0.5)))
  im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

  ticks = list(range(n))
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
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
  return _save(fig, "feature_correlation", show)


def plot_feature_distributions(
  features: np.ndarray,
  targets: np.ndarray,
  feature_names: Optional[list] = None,
  max_features: int = 12,
  title: str = "Feature distributions by class",
  show: bool = True,
) -> str:
  """KDE histograms per feature, coloured by class."""
  classes = np.unique(targets)
  names   = feature_names if feature_names else [f"F{i}" for i in range(
                                                            features.shape[1])]
  n_feat  = min(features.shape[1], max_features)
  ncols   = 4
  nrows   = int(np.ceil(n_feat / ncols))

  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.6))
  axes = np.array(axes).flatten()

  for i in range(n_feat):
    ax = axes[i]
    for j, cls in enumerate(classes):
      mask = targets == cls
      col  = ACCENT_COLORS[j % len(ACCENT_COLORS)]
      ax.hist(
        features[mask, i], bins=25, alpha=0.55,
        color=col, edgecolor="none", density=True,
      )
    ax.set_title(names[i], fontsize=9, color=TEXT_DARK)
    ax.set_xlabel("Value", fontsize=8)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)

  for i in range(n_feat, len(axes)):
    axes[i].set_visible(False)

  legend_handles = [
    mpatches.Patch(
      color=ACCENT_COLORS[j % len(ACCENT_COLORS)],
      label=f"Class {cls}",
      alpha=0.8
    )
    for j, cls in enumerate(classes)
  ]
  fig.legend(handles=legend_handles, loc="upper right", fontsize=9,
              frameon=True, framealpha=0.9, edgecolor=BORDER)
  fig.suptitle(title, fontsize=13, color=TEXT_DARK, y=1.01)
  fig.tight_layout()
  return _save(fig, "feature_distributions", show)


def plot_pca_scatter(
  features: np.ndarray,
  targets: np.ndarray,
  title: str = "PCA — 2D projection",
  show: bool = True,
) -> str:
  """Reduce to 2 principal components and scatter-plot with class colouring."""
  classes = np.unique(targets)
  pca = PCA(n_components=2)
  proj = pca.fit_transform(features)
  var  = pca.explained_variance_ratio_ * 100

  fig, ax = plt.subplots(figsize=(6, 5))
  for j, cls in enumerate(classes):
    mask = targets == cls
    ax.scatter(
      proj[mask, 0], proj[mask, 1],
      c=ACCENT_COLORS[j % len(ACCENT_COLORS)],
      label=f"Class {cls}", alpha=0.65, s=22, linewidths=0,
    )

  ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
  ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
  ax.set_title(title, color=TEXT_DARK, pad=10)
  ax.axhline(0, color=BORDER, linewidth=0.5)
  ax.axvline(0, color=BORDER, linewidth=0.5)
  ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor=BORDER)
  ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)

  fig.tight_layout()
  return _save(fig, "pca_scatter", show)


def plot_pca_variance(
  features: np.ndarray,
  title: str = "Explained variance — PCA",
  show: bool = True,
) -> str:
  """Scree plot showing individual and cumulative explained variance."""
  n_comp = min(features.shape[1], 20)
  pca    = PCA(n_components=n_comp)
  pca.fit(features)
  ind  = pca.explained_variance_ratio_ * 100
  cum  = np.cumsum(ind)

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
  return _save(fig, "pca_variance", show)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON VIEWS
# ═══════════════════════════════════════════════════════════════════════════


def plot_model_accuracy(
  results: list[BenchmarkResult],
  title: str = "Model accuracy comparison",
  show: bool = True,
) -> str:
  """Horizontal bar chart of accuracy per model."""

  names  = [r.model_name for r in results]
  scores = [r.accuracy * 100 for r in results]
  colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(results))]

  # Sort best → worst
  order  = np.argsort(scores)[::-1]
  names  = [names[i]  for i in order]
  scores = [scores[i] for i in order]
  colors = [colors[i] for i in order]

  fig, ax = plt.subplots(figsize=(7, max(3, len(results) * 0.65)))
  bars = ax.barh(names, scores, color=colors, height=0.5, zorder=2)
  ax.set_xlim(0, 105)
  ax.set_xlabel("Accuracy (%)")
  ax.set_title(title, color=TEXT_DARK, pad=10)
  ax.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
  ax.set_axisbelow(True)

  for bar, score in zip(bars, scores):
    ax.text(
      score + 0.5, bar.get_y() + bar.get_height() / 2,
      f"{score:.2f}%", va="center", fontsize=9, color=TEXT_DARK,
    )

  fig.tight_layout()
  return _save(fig, "model_accuracy", show)


def plot_model_timing(
  results: list[BenchmarkResult],
  title: str = "Model training & evaluation time",
  show: bool = True,
) -> str:
  """Grouped bar chart: train time vs eval time per model."""

  names      = [r.model_name for r in results]
  train_t    = [r.train_time for r in results]
  eval_t     = [r.eval_time  for r in results]
  x          = np.arange(len(names))
  width      = 0.35

  fig, ax = plt.subplots(figsize=(max(5, len(results) * 1.6), 4.5))
  ax.bar(x - width / 2, train_t, width, color=PALETTE["blue"],  alpha=0.85,
          label="Train", zorder=2)
  ax.bar(x + width / 2, eval_t,  width, color=PALETTE["teal"], alpha=0.85,
          label="Eval",  zorder=2)

  ax.set_xticks(x)
  ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
  ax.set_ylabel("Time (seconds)")
  ax.set_title(title, color=TEXT_DARK, pad=10)
  ax.yaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
  ax.set_axisbelow(True)
  ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor=BORDER)

  fig.tight_layout()
  return _save(fig, "model_timing", show)


def plot_model_radar(
  results: list[BenchmarkResult],
  title: str = "Model comparison — radar",
  show: bool = True,
) -> str:
  """
  Radar / spider chart comparing models on three normalised axes:
  accuracy, training speed (1 / train_time normalised), eval speed.
  """

  def _norm(values: list[float]) -> list[float]:
      mn, mx = min(values), max(values)
      return [(v - mn) / (mx - mn + 1e-12) for v in values]

  accs        = [r.accuracy  for r in results]
  inv_train   = [1 / (r.train_time + 1e-9) for r in results]
  inv_eval    = [1 / (r.eval_time  + 1e-9) for r in results]

  n_acc  = _norm(accs)
  n_train= _norm(inv_train)
  n_eval = _norm(inv_eval)

  categories  = ["Accuracy", "Train speed", "Eval speed"]
  N           = len(categories)
  angles      = [n / N * 2 * np.pi for n in range(N)]
  angles     += angles[:1]

  fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
  ax.set_theta_offset(np.pi / 2)
  ax.set_theta_direction(-1)
  ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
  ax.set_ylim(0, 1.1)
  ax.yaxis.grid(True, color=BORDER, linewidth=0.5)
  ax.xaxis.grid(True, color=BORDER, linewidth=0.5)
  ax.set_facecolor(BACKGROUND)

  for i, r in enumerate(results):
    vals  = [n_acc[i], n_train[i], n_eval[i]]
    vals += vals[:1]
    col   = ACCENT_COLORS[i % len(ACCENT_COLORS)]
    ax.plot(angles, vals, linewidth=1.5, linestyle="solid", color=col)
    ax.fill(angles, vals, color=col, alpha=0.12)

  ax.set_title(title, color=TEXT_DARK, pad=20)
  legend_handles = [
    mpatches.Patch(
      color=ACCENT_COLORS[i % len(ACCENT_COLORS)],
      label=r.model_name,
      alpha=0.8
    )
    for i, r in enumerate(results)
  ]
  ax.legend(handles=legend_handles, loc="upper right",
            bbox_to_anchor=(1.35, 1.1), fontsize=9,
            frameon=True, framealpha=0.9, edgecolor=BORDER)

  fig.tight_layout()
  return _save(fig, "model_radar", show)


def plot_accuracy_vs_time(
    results: list[BenchmarkResult],
    title: str = "Accuracy vs training time",
    show: bool = True,
) -> str:
    """
    Scatter: x = train_time (log scale), 
    y = accuracy, bubbles sized by eval time.
    """

    accs    = np.array([r.accuracy   * 100 for r in results])
    trains  = np.array([r.train_time       for r in results])
    evals   = np.array([r.eval_time        for r in results])
    names   = [r.model_name for r in results]

    sizes   = 60 + 400 * (
      evals - evals.min()) / (evals.max() - evals.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (x, y, s, name) in enumerate(zip(trains, accs, sizes, names)):
      col = ACCENT_COLORS[i % len(ACCENT_COLORS)]
      ax.scatter(x, y, s=s, c=col, alpha=0.8, linewidths=0.5,
                  edgecolors=TEXT_DARK, zorder=3)
      ax.annotate(name, (x, y), textcoords="offset points",
                  xytext=(8, 4), fontsize=8, color=TEXT_DARK)

    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, color=TEXT_DARK, pad=10)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.text(0.98, 0.03, "Bubble size ∝ eval time",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, ha="right")

    fig.tight_layout()
    return _save(fig, "accuracy_vs_time", show)


# ═══════════════════════════════════════════════════════════════════════════
#  DASHBOARD — ALL-IN-ONE
# ═══════════════════════════════════════════════════════════════════════════


def plot_dashboard(
  results: list[BenchmarkResult],
  features: np.ndarray,
  targets: np.ndarray,
  show: bool = True,
) -> str:
  """
  One-page dashboard combining:
    - Class distribution
    - PCA 2D scatter
    - Model accuracy bars
    - Train vs eval timing
  """

  classes, counts = np.unique(targets, return_counts=True)
  pca = PCA(n_components=2)
  proj = pca.fit_transform(features)
  var  = pca.explained_variance_ratio_ * 100

  names  = [r.model_name for r in results]
  accs   = [r.accuracy * 100 for r in results]
  trains = [r.train_time     for r in results]
  evals  = [r.eval_time      for r in results]

  fig = plt.figure(figsize=(14, 9), facecolor=BACKGROUND)
  gs  = GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.38)

  ax_cls  = fig.add_subplot(gs[0, 0])
  ax_pca  = fig.add_subplot(gs[0, 1:3])
  ax_acc  = fig.add_subplot(gs[1, 0:2])
  ax_time = fig.add_subplot(gs[1, 2:4])

  # ── Class distribution ─────────────────────────────────────────────
  bar_cols = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(
                                                                 len(classes))]
  ax_cls.bar([str(c) for c in classes], 
              counts, color=bar_cols, width=0.5, zorder=2)
  ax_cls.set_title("Class distribution", fontsize=10, color=TEXT_DARK)
  ax_cls.set_xlabel("Class"); ax_cls.set_ylabel("Count")
  ax_cls.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
  ax_cls.set_axisbelow(True)

  # ── PCA scatter ────────────────────────────────────────────────────
  for j, cls in enumerate(classes):
      mask = targets == cls
      ax_pca.scatter(proj[mask, 0], proj[mask, 1],
                      c=ACCENT_COLORS[j % len(ACCENT_COLORS)],
                      label=f"Class {cls}", alpha=0.6, s=18, linewidths=0)
  ax_pca.set_title("PCA — 2D projection", fontsize=10, color=TEXT_DARK)
  ax_pca.set_xlabel(f"PC1 ({var[0]:.1f}%)"); ax_pca.set_ylabel(
                                                        f"PC2 ({var[1]:.1f}%)")
  ax_pca.axhline(0, color=BORDER, lw=0.4); ax_pca.axvline(
                                                       0, color=BORDER, lw=0.4)
  ax_pca.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor=BORDER)
  ax_pca.grid(True, color=BORDER, linewidth=0.3, alpha=0.5)

  # ── Model accuracy ─────────────────────────────────────────────────
  order = np.argsort(accs)[::-1]
  s_names  = [names[i]  for i in order]
  s_accs   = [accs[i]   for i in order]
  s_colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in order]
  ax_acc.barh(s_names, s_accs, color=s_colors, height=0.45, zorder=2)
  ax_acc.set_xlim(0, 105)
  ax_acc.set_xlabel("Accuracy (%)")
  ax_acc.set_title("Model accuracy", fontsize=10, color=TEXT_DARK)
  ax_acc.xaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
  ax_acc.set_axisbelow(True)
  for bar, score in zip(ax_acc.patches, s_accs):
    ax_acc.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%", va="center", fontsize=8, color=TEXT_DARK)

  # ── Timing ─────────────────────────────────────────────────────────
  x     = np.arange(len(names))
  w     = 0.35
  ax_time.bar(x - w / 2, trains, w, color=PALETTE["blue"],  alpha=0.85,
              label="Train", zorder=2)
  ax_time.bar(x + w / 2, evals,  w, color=PALETTE["teal"], alpha=0.85,
              label="Eval",  zorder=2)
  ax_time.set_xticks(x)
  ax_time.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
  ax_time.set_ylabel("Time (s)")
  ax_time.set_title("Train vs eval time", fontsize=10, color=TEXT_DARK)
  ax_time.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
  ax_time.set_axisbelow(True)
  ax_time.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor=BORDER)

  fig.suptitle("ML Benchmark dashboard", fontsize=14, color=TEXT_DARK, y=1.01)
  return _save(fig, "dashboard", show)


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE — run everything at once
# ═══════════════════════════════════════════════════════════════════════════


def visualise_all(
  results: list[BenchmarkResult],
  features: np.ndarray,
  targets: np.ndarray,
  feature_names: Optional[list] = None,
  show: bool = False,
) -> list[str]:
  """
  Generates and saves all plots.  Returns a list of output paths.

  Parameters
  ----------
  results      : list of BenchmarkResult from benchmark.Benchmark
  features     : raw feature array (pre-reduction, for dataset views)
  targets      : target labels array
  feature_names: optional column labels for correlation / distribution plots
  show         : whether to call plt.show() for each figure
  """
  paths = []

  print("Rendering dataset views …")
  paths.append(plot_class_distribution(targets, show=show))
  paths.append(plot_pca_scatter(features, targets, show=show))
  paths.append(plot_pca_variance(features, show=show))

  if features.shape[1] <= 50:
      paths.append(plot_feature_distributions(features, targets,
                                       feature_names=feature_names, show=show))
  if features.shape[1] <= 30:
      paths.append(plot_feature_correlation(
                             features, feature_names=feature_names, show=show))

  print("Rendering model comparison views …")
  paths.append(plot_model_accuracy(results, show=show))
  paths.append(plot_model_timing(results, show=show))
  paths.append(plot_accuracy_vs_time(results, show=show))

  if len(results) >= 2:
      paths.append(plot_model_radar(results, show=show))

  print("Rendering dashboard …")
  paths.append(plot_dashboard(results, features, targets, show=show))

  print(f"\nAll plots saved to {OUTPUT_DIR}/")
  return paths