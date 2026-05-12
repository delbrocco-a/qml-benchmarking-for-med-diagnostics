import pandas as pd
import numpy as np
from src.load_data import load_csv, split_data


LOAD_CSV_TEST_CONTENT = "a,b,c\n1,2,3\n4,5,6"

# Regression: heart3-style CSV where target is text ('negative'/'positive').
# pandas >=2.0 StringDtype silently rejected integer assignment from
# pd.factorize, leaving the column as strings; pd.to_numeric then coerced
# everything to NaN, fillna(0) made all targets 0, destroying class info.
LOAD_CSV_TEXT_TARGET = (
    "feature1,feature2,label\n"
    "1.0,2.0,negative\n"
    "3.0,4.0,positive\n"
    "5.0,6.0,negative\n"
    "7.0,8.0,positive\n"
)
LOAD_CSV_TEST_ANSWERS = [
  ["a", "b", "c"], (2, 3)
]

SPLIT_DATA_TEST_CONTENT = [
  pd.DataFrame({
    "a": [1, 2, 3, 4, 5],
    "b": [10, 20, 30, 40, 50],
    "target": [0, 1, 0, 1, 0]
  }),
  "target", 
  0.4 ### Split ratio
]


def test_load_csv_returns_dataframe_and_columns(tmp_path):
  ### Create fake file for us to test
  test_file = tmp_path / "test.csv"
  test_file.write_text(LOAD_CSV_TEST_CONTENT)

  ### Store function results
  csv, feats = load_csv(str(test_file))

  ### Test ---
  assert isinstance(csv, pd.DataFrame)
  assert feats == LOAD_CSV_TEST_ANSWERS[0]
  assert csv.shape == LOAD_CSV_TEST_ANSWERS[1]


def test_split_data_basic():

  result = split_data(
    SPLIT_DATA_TEST_CONTENT[0],
    SPLIT_DATA_TEST_CONTENT[1],
    SPLIT_DATA_TEST_CONTENT[2]
  )

  # Check top-level keys
  assert set(result.keys()) == {"training", "testing"}

  for subset in ["training", "testing"]:
    # Check nested keys
    assert set(result[subset].keys()) == {"features", "targets"}
    feats = result[subset]["features"]
    targs = result[subset]["targets"]
    # Check that features and targets have the same length
    assert len(feats) == len(targs)

  # Check that split roughly matches requested ratio
  total_rows = len(SPLIT_DATA_TEST_CONTENT[0])
  test_len = len(result["testing"]["features"])
  expected_test_len = int(total_rows * SPLIT_DATA_TEST_CONTENT[2])
  # Allow ±1 row for rounding
  assert abs(test_len - expected_test_len) <= 1

  # Check that training + testing rows equals original data
  train_len = len(result["training"]["features"])
  assert train_len + test_len == total_rows


def test_load_csv_text_target_both_classes_preserved(tmp_path):
  """Regression: text target columns must encode to both classes, not all-zero.

  Covers the pandas >=2.0 StringDtype bug where factorize codes were
  silently dropped on reassignment, collapsing all targets to 0 via NaN.
  """
  test_file = tmp_path / "text_target.csv"
  test_file.write_text(LOAD_CSV_TEXT_TARGET)

  csv, feats = load_csv(str(test_file))

  target_col = feats[-1]   # 'label'
  unique_vals = csv[target_col].unique()

  # Must have exactly two distinct integer codes (0 and 1), not all zeros
  assert len(unique_vals) == 2, (
      f"Expected 2 classes after encoding, got {len(unique_vals)}: {unique_vals}"
  )
  assert csv[target_col].dtype != object, (
      "Target column must be numeric after load_csv, not object/string"
  )