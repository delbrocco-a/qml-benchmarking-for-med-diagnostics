import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tt_split, StratifiedKFold
from typing import Optional, Generator
from src.CONST import TRAINING, TESTING, FEATURES, TARGETS

N_FOLDS = 5  # stratified k-fold cross-validation across all experiments


def load_csv(file_path: str) -> tuple[pd.DataFrame, list[str]]:
  """Loads CSV data from file, into CSV data frame with header list"""

  csv = pd.read_csv(file_path)
  feats = csv.columns.to_list()
  
  # Encode categorical columns.
  # Explicitly construct a new int64 Series to avoid pandas StringDtype
  # silently rejecting integer assignment (observed in pandas >=2.0 with
  # the qml-env; a plain assignment leaves the column as object/string,
  # then pd.to_numeric coerces the text to NaN and fillna collapses all
  # targets to 0, destroying class information).
  for col in csv.columns:
    if not pd.api.types.is_numeric_dtype(csv[col]):
      codes = pd.factorize(csv[col])[0]
      csv[col] = pd.Series(codes, index=csv.index, dtype='int64')

  # Ensure all columns are numeric (catches any residual non-numeric after
  # encoding; legitimate NaNs from genuine missing values are zero-filled)
  csv = csv.apply(pd.to_numeric, errors='coerce')
  csv = csv.fillna(0)
  return csv, feats


def cv_splits(
  csv: pd.DataFrame, target: str, n_splits: int = N_FOLDS, seed: int = 42
) -> Generator[dict, None, None]:
  """Yields n_splits stratified train/test dicts for cross-validation.

  Stratification preserves class balance across folds — important for the
  imbalanced medical datasets used in this project (diabetes ~65/35,
  Hungarian ~64/36).
  """
  X   = csv.drop(target, axis=1).values
  y   = csv[target].values
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

  for train_idx, test_idx in skf.split(X, y):
    yield {
      TRAINING : { FEATURES: X[train_idx], TARGETS: y[train_idx] },
      TESTING  : { FEATURES: X[test_idx],  TARGETS: y[test_idx]  },
    }


def split_data(data: pd.DataFrame, target: str, split: Optional[float] = 0.2
  ) -> dict:
  """Splits data into training & testing, features & targets, dictionaries"""

  ### Separate target column from data column for training & testing
  targs = data[target].values
  feats = data.drop(target, axis=1).values

  ### Test train split ...
  feats_train, feats_test, targs_train, targs_test = tt_split(
    feats, targs, test_size=split, random_state=42
  )

  return {
    TRAINING : { FEATURES: feats_train, TARGETS: targs_train },
    TESTING  : { FEATURES: feats_test,  TARGETS: targs_test  }
  }



