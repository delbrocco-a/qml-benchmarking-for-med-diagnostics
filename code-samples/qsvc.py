"""
QSVC Kernel training (taken from documentation)
"""

### Random seed for reproducibility on documentation results (NOT NEEDED)
from qiskit_machine_learning.utils import algorithm_globals
algorithm_globals.random_seed = 12345


# DATA GENERATION & VISUALISATION (AD HOC FOR TESTING)

from qiskit_machine_learning.datasets import ad_hoc_data
import matplotlib.pyplot as plt
import numpy as np

## Load in some data so we have something to train with
DIMS = 2
train_feats, train_labels, test_feats, test_labels, total = ad_hoc_data(
  training_size=20,
  test_size=5,
  n=DIMS,
  gap=0.3,
  plot_data=False,
  one_hot=False,
  include_sample_total=True
)

## Plot features
def plot_feats(ax, feats, labels, class_labl, marker, face, edge, labl):
  ax.scatter(
    feats[np.where(labels[:] == class_labl), 0],
    feats[np.where(labels[:] == class_labl), 1],
    marker=marker,
    facecolors=face,
    edgecolors=edge,
    label=labl
  )

## Plot data
def plot_dataset(train_feats, train_labels, test_feats, test_labels, total):
  plt.figure(figsize=(5, 5))
  plt.ylim(0, 2 * np.pi)
  plt.xlim(0, 2 * np.pi)
  plt.imshow(
    np.asmatrix(total).T,
    interpolation="nearest",
    origin="lower",
    cmap="RdBu",
    extent=[0, 2 * np.pi, 0, 2 * np.pi]
  )

  ### Train and test to spot both class labels (A & B)
  plot_feats(plt, train_feats, train_labels, 0, "s", "w", "b", "A train")
  plot_feats(plt, train_feats, train_labels, 1, "o", "w", "r", "B train")
  plot_feats(plt, test_feats, test_labels, 0, "s", "b", "w", "A test")
  plot_feats(plt, test_feats, test_labels, 1, "o", "r", "w", "B test")

  plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
  plt.title("Ad hoc dataset")
  plt.show()

## Run the numbers...
plot_dataset(train_feats, train_labels, test_feats, test_labels, total)


# CREATE A QUANTUM KERNEL TO TRAIN WITH DATA

from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

## Create some way of sampling the state vector (sanme in p_qsvc.py)
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)

## Generate a feature map ... (?)
adhoc_feature_map = zz_feature_map(
  feature_dimension=DIMS, reps=2, entanglement="linear"
)

## Create the kernel
adhoc_kernel = FidelityQuantumKernel(
  fidelity=fidelity, feature_map=adhoc_feature_map
)


