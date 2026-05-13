"""
Quantum ML benchmark pipeline.

Runs 4 datasets x 7 qubit counts (2-8) x 5 CV folds = 140 configurations.
Each configuration: SelectKBest -> PCA -> ScaleForQuantum -> train -> evaluate.

Models per configuration:
  SVC (rbf, capped)     -- classical, trained on same n≤200 cap as quantum (fair comparison)
  LogReg (capped)       -- classical, same cap
  QSVC (ZZFeatureMap)   -- quantum kernel SVM
  PegasosQSVC           -- stochastic quantum kernel SVM, O(tau) kernel evals
  SVC (rbf, full)       -- classical, full training set (upper bound reference)
  LogReg (full)         -- classical, full training set (upper bound reference)

The capped classical models are the primary comparison: quantum vs classical
under equal training data. Full classical models are included as an upper bound
to contextualise how much the quantum cap costs.

Datasets:
  Pima Indians Diabetes -- Smith et al. (1988), UCI #34
  Hungarian Heart Disease -- Janosi et al. (1988), UCI #45
  Wisconsin Breast Cancer -- Street et al. (1993), UCI #17
"""


### Import Python Libraries
import json
import os
import sys
import warnings
from collections import defaultdict
import numpy as np

## Tee stdout to outputs/run.log so the full terminal output survives power-off.
## Line-buffered (buffering=1) so each printed line is flushed immediately.
os.makedirs("outputs", exist_ok=True)
_log_file = open("outputs/run.log", "w", buffering=1)

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams: s.write(data)
    def flush(self):
        for s in self._streams: s.flush()

sys.stdout = _Tee(sys.__stdout__, _log_file)

### Import Project Library
from src.CONST import ON, OFF, TRAINING, TESTING, FEATURES, TARGETS
from src.load_data import load_csv, split_data, cv_splits, N_FOLDS
from src.feat_selector import PCASelect, ScaleForQuantum
from src.svc import trainSVC, evalSVC, RBF, tuneSVC
from src.qsvc import trainQSVC, evalQSVC, trainPQSVC, evalPQSVC
from src.log_reg import trainLogReg, evalLogReg, tuneLogReg
from src.qkernel import FQKernel, ZZ
from src.benchmark import Benchmark, aggregate_folds
from src.stats import wilcoxon_pairwise, summary_table
from src.visualise import (
    visualise_all,
    plot_qubit_sweep_accuracy,
    plot_qubit_sweep_timing,
    plot_qubit_sweep_quantum_time,
    plot_qubit_sweep_resource,
    plot_classical_vs_quantum,
    plot_cross_dataset_all_models,
    plot_wilcoxon_heatmap,
)
from src.q_estimate import estimate_kernel_runtime, estimate_pegasos_runtime

## Suppress noisy - but harmless - warnings:
### RuntimeWarning: *numpy divide-by-zero from f_classif on constant features*
### UserWarning: *sklearn "Features X are constant" from SelectKBest*
### ConvergenceWarning: *lbfgs max_iter (no longer raised; kept as safety net)*
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ! SET-UP - CONFIGURE PIPELINE HERE !


## 1.) Input target datasets, their names, and the feature reduction.

# Datasets:
#   Smith, J.W. et al. (1988) 'Using the ADAP learning algorithm to forecast
#   the onset of diabetes mellitus', Proc. SCAMC, pp. 261-265.
#   Available at: https://archive.ics.uci.edu/dataset/34 (Accessed: 28 April 2026).
#
#   Janosi, A. et al. (1988) Heart Disease [Dataset]. UCI ML Repository.
#   Available at: https://archive.ics.uci.edu/dataset/45 (Accessed: 28 April 2026).
#
#   Detrano, R. et al. (1989) 'International application of a new probability
#   algorithm for the diagnosis of coronary artery disease', American Journal of
#   Cardiology, 64(5), pp. 304-310.
#   Available at: https://archive.ics.uci.edu/dataset/45 (Accessed: 29 April 2026).
#
#   Street, W.N., Wolberg, W.H. and Mangasarian, O.L. (1993) 'Nuclear feature
#   extraction for breast tumor diagnosis', IS&T/SPIE Symposium on Electronic
#   Imaging, 1905, pp. 861-870.
#   Available at: https://archive.ics.uci.edu/dataset/17 (Accessed: 28 April 2026).

DATASETS = [
### (file_path, short_label, k_select_features)
    ("data/diabetes.csv",     "diabetes",     8),   ### 8 features total; use all
    ("data/hungarian.csv",    "hungarian",    10),  ### 13 features; select 10
    ("data/cleveland.csv",    "cleveland",    10),  ### 13 features; select 10
    ("data/breast_cancer.csv","breast_cancer",15),  ### 30 features; select 15
]


## 2.) Input cross-validation configuration.

# 5-fold stratified CV is the standard for academic ML benchmarking:
# each fold preserves class balance, and 5 repeats give stable mean ± SD
# estimates without inflating runtime beyond overnight feasibility on this
# hardware (i7-1165G7, 16 GB RAM).
# N_FOLDS is imported from load_data to keep CV config in one place.

print(f"Cross-validation: {N_FOLDS}-fold stratified")


## 3.) Input quantum configuration; qubits used, training cap.

QUBIT_RANGE = range(2, 9)  ### Test machines from 2 to 8 qubits

QSVC_MAX_TRAIN = 200
# QSVC kernel matrix is O(n_train^2) evaluations. Cap at 200 to keep each
# CV fold tractable on consumer hardware. Capped classical models train on
# the same subset for a fair accuracy comparison.


## 4.) Input Pegasos custom parameters.

PEGASOS_C   = 1000.0
# Qiskit default. Quantum kernels in high-dimensional spaces produce values
# close to zero (exponential concentration), so a large C is needed to give
# the SVM margin enough room to act on the kernel signal.
# C=1.0 causes degenerate majority-class prediction at all qubit counts.
# Ref: Kübler et al. (2021), 'The Inductive Bias of Quantum Kernels', NeurIPS.
PEGASOS_TAU = 200
# tau bounds training to O(tau) kernel evals — much cheaper than QSVC's
# O(n^2). Kept at 200 (not the Qiskit default of 1000) to stay within the
# overnight compute budget on this hardware; tau=200 is sufficient for
# convergence once C is correctly set.

## 5.) ZZFeatureMap repetitions: fixed at 2 throughout all qubit counts.
# Using a constant reps value ensures the architecture does not change
# mid-sweep, so qubit count is the only variable on the quantum side.
REPS = 2


def enough_classes(data: dict) -> bool:
    """Check the split hasn't collapsed to a single class (edge case on small folds)."""
    return (
        len(np.unique(data[TRAINING][TARGETS])) >= 2
        and len(np.unique(data[TESTING][TARGETS]))  >= 2
    )


# Exit codes
EXIT_SUCCESS     = 0
EXIT_ALL_SKIPPED = 1
EXIT_EXCEPTION   = 2





# ============================================================================
# Ex1: Quantum vs Classical - 5-fold stratified CV, 4 datasets x 7 qubit counts
# ============================================================================

### Cross-dataset summary: { dataset_label: { n_qubits: [BenchmarkResult] } }
all_results: dict[str, dict[int, list]] = {}

# Raw per-fold results — needed for the Wilcoxon test; aggregated results
# lose the fold-level variance that makes the test meaningful.
all_fold_results: dict[str, dict[int, dict[str, list]]] = {}

# Nested-CV hyperparameter log: { dataset: { n_feats: { model: [per-fold entries] } } }
tuned_params_log: dict = {}

try:
    for file_path, ds_label, n_select in DATASETS:

        print(f"\n\n -|- Dataset: {ds_label}  ({file_path})\n")

        csv, feats = load_csv(file_path)
        print(f"  Loaded {len(feats)} columns: {feats}")

        all_results[ds_label]      = {}
        all_fold_results[ds_label] = {}
        sweep_results: dict[int, list] = {}

        for n_feats in QUBIT_RANGE:

            select = max(n_feats, min(n_select, len(feats) - 1))

            print(f"\n -|- {ds_label} | {n_feats} qubit(s) "
                  f"(SelectKBest={select} -> PCA={n_feats}, {N_FOLDS}-fold CV)")

            ### Accumulate per-fold results for each model name
            fold_results: dict[str, list] = defaultdict(list)
            fold_skipped = 0

            for fold_idx, fold_data in enumerate(cv_splits(csv, feats[-1], N_FOLDS)):

                ### Feature selection: SelectKBest -> PCA (fit on this fold's train only)
                train_pca, test_pca = PCASelect(select=select, feats=n_feats, data=fold_data)

                pca_data = {
                    TRAINING: {FEATURES: train_pca, TARGETS: fold_data[TRAINING][TARGETS]},
                    TESTING:  {FEATURES: test_pca,  TARGETS: fold_data[TESTING][TARGETS]},
                }

                if not enough_classes(pca_data):
                    print(f"  Fold {fold_idx + 1}: skipped (single class after PCA).")
                    fold_skipped += 1
                    continue

                ### Rescale to [0, π] for quantum angle encoding
                ### Scaler is fit on this fold's training data only
                train_q, test_q = ScaleForQuantum(train_pca, test_pca)

                q_data = {
                    TRAINING: {FEATURES: train_q, TARGETS: fold_data[TRAINING][TARGETS]},
                    TESTING:  {FEATURES: test_q,  TARGETS: fold_data[TESTING][TARGETS]},
                }

                ### Cap training set for quantum models and capped classical models
                n_train_full = pca_data[TRAINING][FEATURES].shape[0]
                if n_train_full > QSVC_MAX_TRAIN:
                    idx = np.random.default_rng(42 + fold_idx).choice(
                        n_train_full, size=QSVC_MAX_TRAIN, replace=False
                    )
                    q_train_capped = {
                        FEATURES: q_data[TRAINING][FEATURES][idx],
                        TARGETS:  q_data[TRAINING][TARGETS][idx],
                    }
                    pca_train_capped = {
                        FEATURES: pca_data[TRAINING][FEATURES][idx],
                        TARGETS:  pca_data[TRAINING][TARGETS][idx],
                    }
                else:
                    q_train_capped   = q_data[TRAINING]
                    pca_train_capped = pca_data[TRAINING]

                # Nested CV: tune SVC and LogReg on this fold's capped training
                # portion using 3-fold inner CV. The test fold is never seen
                # during tuning — this is correct nested-CV protocol.
                # We bias the comparison in favour of quantum: classical models
                # are tuned while quantum models use Qiskit defaults throughout.
                svc_params, svc_score = tuneSVC(pca_train_capped)
                lr_params,  lr_score  = tuneLogReg(pca_train_capped)

                # Record which params won for this fold
                tuned_params_log.setdefault(ds_label, {}).setdefault(
                    str(n_feats), {"SVC": [], "LogReg": []}
                )
                tuned_params_log[ds_label][str(n_feats)]["SVC"].append(
                    {"fold": fold_idx, **svc_params, "inner_cv": round(svc_score, 4)}
                )
                tuned_params_log[ds_label][str(n_feats)]["LogReg"].append(
                    {"fold": fold_idx, **lr_params, "inner_cv": round(lr_score, 4)}
                )

                qkernel = FQKernel(qubits=n_feats, reps=REPS, map=ZZ)

                fold_bench = Benchmark()

                # --- Primary comparison: equal training data ---
                # These three models all train on the same ≤200 sample cap,
                # making the accuracy comparison directly fair.
                # Classical models use the tuned params from above.

                fold_bench.run(
                    model_name = "SVC (rbf, capped)",
                    train_fn   = lambda p=svc_params: trainSVC(
                        pca_train_capped, kernel=RBF, **p
                    ),
                    eval_fn    = lambda m: evalSVC(m, pca_data[TESTING]),
                    train_data = pca_train_capped, test_data=pca_data[TESTING],
                    extra      = {"kernel": RBF, "n_feats": n_feats,
                                  "n_train": pca_train_capped[FEATURES].shape[0],
                                  **svc_params},
                )

                fold_bench.run(
                    model_name = "LogReg (capped)",
                    train_fn   = lambda p=lr_params: trainLogReg(
                        pca_train_capped, **p
                    ),
                    eval_fn    = lambda m: evalLogReg(m, pca_data[TESTING]),
                    train_data = pca_train_capped, test_data=pca_data[TESTING],
                    extra      = {"n_feats": n_feats,
                                  "n_train": pca_train_capped[FEATURES].shape[0],
                                  **lr_params},
                )

                fold_bench.run(
                    model_name = "QSVC",
                    train_fn   = lambda: trainQSVC(qkernel, q_train_capped),
                    eval_fn    = lambda m: evalQSVC(m, q_data[TESTING]),
                    train_data = q_train_capped, test_data=q_data[TESTING],
                    extra      = {"map": ZZ, "qubits": n_feats, "reps": REPS,
                                  "n_train": q_train_capped[FEATURES].shape[0]},
                )

                fold_bench.run(
                    model_name = "PegasosQSVC",
                    train_fn   = lambda: trainPQSVC(
                        qkernel, q_train_capped, C=PEGASOS_C, tau=PEGASOS_TAU
                    ),
                    eval_fn    = lambda m: evalPQSVC(m, q_data[TESTING]),
                    train_data = q_train_capped, test_data=q_data[TESTING],
                    extra      = {"map": ZZ, "qubits": n_feats, "reps": REPS,
                                  "C": PEGASOS_C, "tau": PEGASOS_TAU,
                                  "n_train": q_train_capped[FEATURES].shape[0]},
                )

                # --- Upper bound reference: full classical training set ---
                # Shows what classical models can achieve with more data;
                # not a fair comparison against quantum but useful context.

                fold_bench.run(
                    model_name = "SVC (rbf, full)",
                    train_fn   = lambda: trainSVC(pca_data[TRAINING], kernel=RBF),
                    eval_fn    = lambda m: evalSVC(m, pca_data[TESTING]),
                    train_data = pca_data[TRAINING], test_data=pca_data[TESTING],
                    extra      = {"kernel": RBF, "n_feats": n_feats,
                                  "n_train": pca_data[TRAINING][FEATURES].shape[0]},
                )

                fold_bench.run(
                    model_name = "LogReg (full)",
                    train_fn   = lambda: trainLogReg(pca_data[TRAINING]),
                    eval_fn    = lambda m: evalLogReg(m, pca_data[TESTING]),
                    train_data = pca_data[TRAINING], test_data=pca_data[TESTING],
                    extra      = {"n_feats": n_feats,
                                  "n_train": pca_data[TRAINING][FEATURES].shape[0]},
                )

                for r in fold_bench.results:
                    fold_results[r.model_name].append(r)

                print(f"  Fold {fold_idx + 1}/{N_FOLDS} done.")

            if fold_skipped == N_FOLDS:
                print(f"  All folds skipped for {ds_label} @ {n_feats}q — no results.")
                continue

            # Save raw fold results — used later for the Wilcoxon test
            all_fold_results[ds_label][n_feats] = dict(fold_results)

            ### Aggregate folds -> mean ± SD for each model
            bench = Benchmark()
            for model_name, folds in fold_results.items():
                if folds:
                    bench.results.append(aggregate_folds(folds))

            ### Attach quantum hardware runtime estimates (fold-independent)
            n_test = len(csv) // N_FOLDS  # approximate test set size

            for r in bench.results:
                if r.model_name == "QSVC":
                    qe = estimate_kernel_runtime(
                        n_train  = QSVC_MAX_TRAIN,
                        qubits   = n_feats,
                        reps     = REPS,
                        map_name = ZZ,
                    )
                    r.quantum_time_s     = qe.total_time_s
                    r.quantum_time_human = qe.total_time_human

                elif r.model_name == "PegasosQSVC":
                    pqe = estimate_pegasos_runtime(
                        n_train  = QSVC_MAX_TRAIN,
                        n_test   = n_test,
                        tau      = PEGASOS_TAU,
                        qubits   = n_feats,
                        reps     = REPS,
                        map_name = ZZ,
                    )
                    r.quantum_time_s     = pqe.total_time_s
                    r.quantum_time_human = pqe.total_time_human

            print(bench.summary())

            # Per-run plots -> outputs/plots/<ds_label>_<n_feats>q/
            prefix = f"{ds_label}_{n_feats}q"
            visualise_all(
                results      = bench.results,
                features     = csv.drop(csv.columns[-1], axis=1).values,
                targets      = csv[csv.columns[-1]].values,
                feature_names= list(csv.columns[:-1]),
                dataset_name = f"{ds_label} | {n_feats}q",
                prefix       = prefix,
                show         = False,
            )

            sweep_results[n_feats]         = bench.results
            all_results[ds_label][n_feats] = bench.results

        # Per-dataset qubit sweep plots -> outputs/plots/<ds_label>_sweep/
        sweep_prefix = f"{ds_label}_sweep"
        print(f"\n  Rendering qubit-sweep plots for {ds_label}...")

        plot_qubit_sweep_accuracy(
            sweep_results,
            title  = f"Balanced accuracy vs qubits — {ds_label}",
            prefix = sweep_prefix, show=False,
        )
        plot_qubit_sweep_timing(
            sweep_results,
            title  = f"Training time vs qubits — {ds_label}",
            prefix = sweep_prefix, show=False,
        )
        plot_qubit_sweep_quantum_time(
            sweep_results,
            title  = f"Estimated quantum hardware time vs qubits — {ds_label}",
            prefix = sweep_prefix, show=False,
        )
        plot_qubit_sweep_resource(
            sweep_results,
            title  = f"CPU time and peak RAM vs qubits — {ds_label}",
            prefix = sweep_prefix, show=False,
        )
        plot_classical_vs_quantum(
            sweep_results,
            title  = f"Classical simulation vs quantum hardware time — {ds_label}",
            prefix = sweep_prefix, show=False,
        )

except Exception:
    ex2_code = EXIT_EXCEPTION
else:
    ex2_code = (
        EXIT_ALL_SKIPPED
        if not any(all_results[ds] for ds in all_results)
        else EXIT_SUCCESS
    )


msg1 = "OK" if ex1_code == EXIT_SUCCESS else "ERROR"
msg2 = "OK" if ex2_code == EXIT_SUCCESS else "ERROR"

print(
    f"\nExperiment 1 (QPCA vs PCA):          {ex1_code} {msg1}"
    f"\nExperiment 2 (QSVC vs classical, CV): {ex2_code} {msg2}"
)

## Post process
print(
    f"\n{'=' * 80}"
    " -|- Rendering cross-dataset summary plots..."
    f"\n{'=' * 80}"
)

plot_cross_dataset_all_models(
    all_results = all_results,
    title       = "All models — balanced accuracy by dataset and qubit count",
    prefix      = "summary",
    show        = False,
)

# ============================================================================
# Statistical analysis and hyperparameter summary
# ============================================================================

os.makedirs("outputs/stats", exist_ok=True)

### Persist tuned hyperparameters from nested CV
with open("outputs/stats/tuned_params.json", "w") as f:
    json.dump(tuned_params_log, f, indent=2, default=str)
print("  Saved -> outputs/stats/tuned_params.json")

### Wilcoxon signed-rank test: quantum vs classical baselines
# Run only if Ex2 produced fold-level data
if all_fold_results and any(all_fold_results[ds] for ds in all_fold_results):
    print(
        f"\n{'=' * 80}"
        "\n -|- Wilcoxon signed-rank test: quantum vs classical"
        f"\n{'=' * 80}"
    )

    # Per-dataset breakdown — primary reported comparison (n≈35 per pair)
    for ds in all_fold_results:
        df_ds = wilcoxon_pairwise(all_fold_results, datasets=[ds])
        path  = f"outputs/stats/wilcoxon_{ds}.csv"
        df_ds.to_csv(path, index=False)
        print(f"\n  {ds} (n≈{df_ds['n_pairs'].max() if not df_ds.empty else 0} pairs):")
        print(summary_table(df_ds))
        print(f"  Saved -> {path}")

    # Pooled across all datasets and qubit counts — secondary / supplementary.
    df_pooled = wilcoxon_pairwise(all_fold_results)
    df_pooled.to_csv("outputs/stats/wilcoxon_pooled.csv", index=False)
    n_pooled = df_pooled['n_pairs'].max() if not df_pooled.empty else 0
    print(f"\n  Pooled (n={n_pooled}) — interpret per §2.5; primary analysis is per-dataset above")
    print(summary_table(df_pooled))
    print("  Saved -> outputs/stats/wilcoxon_pooled.csv")

    plot_wilcoxon_heatmap(
        df_pooled,
        title  = "Quantum vs classical — median Δ balanced accuracy (pooled)",
        prefix = "summary",
        show   = False,
    )

print(
    "\nDone. Output structure:"
    "\n     outputs/plots/"
    "\n     |- <dataset>_<n>q/   (per-run plots, 21 folders)"
    "\n     |- <dataset>_sweep/  (qubit-sweep charts, 3 folders)"
    "\n     `- summary/          (cross-dataset heatmaps)"
    "\n     outputs/stats/"
    "\n     |- tuned_params.json         (nested-CV hyperparameter selections)"
    "\n     |- wilcoxon_pooled.csv       (pooled test, n≈140 per pair)"
    "\n     `- wilcoxon_<dataset>.csv    (per-dataset test, n≈35 per pair)"
)
