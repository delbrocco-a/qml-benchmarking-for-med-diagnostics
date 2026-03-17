import pandas as pd
from sklearn.model_selection import train_test_split as tt_split
from typing import Optional
from src.CONST import TRAINING, TESTING, FEATURES, TARGETS


def load_csv(file_path: str) -> tuple[pd.DataFrame, list[str]]:
  """Loads CSV data from file, into CSV data frame with header list"""

  csv = pd.read_csv(file_path)
  feats = csv.columns.to_list()
  return csv, feats

def split_data(data: pd.DataFrame, target: str, split: Optional[float] = 0.2
  ) -> dict:
  """Splits data into training & testing, features & targets, dictionaries"""

  ### Separate target column from data column for training & testing
  targs = data[target].values
  feats = data.drop(target, axis=1).values

  ### Test train split ...
  feats_train, feats_test, targs_train, targs_test = tt_split(
    feats, targs, test_size=split
  )

  return {
    TRAINING : { FEATURES: feats_train, TARGETS: targs_train },
    TESTING  : { FEATURES: feats_test,  TARGETS: targs_test  }
  }



