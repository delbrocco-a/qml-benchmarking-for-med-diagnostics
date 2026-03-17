from src.CONST import FILE
from src.load_data import load_csv, split_data, TRAINING, TESTING
from src.feat_selector import PCASelect
from src.svc import trainSVC, evalSVC, RBF
from src.qsvc import trainQSVC, evalQSVC
from src.log_reg import trainLogReg, evalLogReg
from src.qkernel import FQKernel, ZZ
from src.benchmark import Benchmark
from src.visualise import visualise_all

# ── Load & split ──────────────────────────────────────────────────────────
csv, feats = load_csv(FILE)
data = split_data(csv, target=feats[-1])

# ── Feature reduction ─────────────────────────────────────────────────────
N_SELECT, N_FEATS = 10, 4
train_pca, test_pca = PCASelect(select=N_SELECT, feats=N_FEATS, data=data)

pca_data = {
  TRAINING: {"features": train_pca, "targets": data[TRAINING]["targets"]},
  TESTING:  {"features": test_pca,  "targets": data[TESTING]["targets"]},
}

# ── Benchmark ─────────────────────────────────────────────────────────────
bench = Benchmark()

bench.run(
  model_name="SVC (rbf)",
  train_fn=lambda: trainSVC(pca_data[TRAINING], kernel=RBF),
  eval_fn=lambda m: evalSVC(m, pca_data[TESTING]),
  train_data=pca_data[TRAINING],
  test_data=pca_data[TESTING],
  extra={"kernel": RBF, "n_feats": N_FEATS},
)

bench.run(
  model_name="Logistic Regression",
  train_fn=lambda: trainLogReg(pca_data[TRAINING]),
  eval_fn=lambda m: evalLogReg(m, pca_data[TESTING]),
  train_data=pca_data[TRAINING],
  test_data=pca_data[TESTING],
  extra={"n_feats": N_FEATS},
)

actual_feats = pca_data[TRAINING]["features"].shape[1]
qkernel = FQKernel(qubits=actual_feats, encodes=2, map=ZZ)
bench.run(
  model_name="QSVC (ZZFeatureMap)",
  train_fn=lambda: trainQSVC(qkernel, pca_data[TRAINING]),
  eval_fn=lambda m: evalQSVC(m, pca_data[TESTING]),
  train_data=pca_data[TRAINING],
  test_data=pca_data[TESTING],
  extra={"map": ZZ, "qubits": N_FEATS, "reps": 2},
)

print(bench.summary())

# ── Visualise ─────────────────────────────────────────────────────────────
# Pass the raw (pre-PCA) training features for dataset views so you see
# the full original feature space, not just the reduced one.
visualise_all(
  results=bench.results,
  features=data[TRAINING]["features"],
  targets=data[TRAINING]["targets"],
  feature_names=feats[:-1],   # all columns except the target
  show=False,                 # set True to open interactive windows
)