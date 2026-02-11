from sklearn.linear_model import LogisticRegression

from src.load_data import FEATURES, TARGETS

def trainLogReg(data: dict) -> LogisticRegression:
  """Trains a classical Logistic Regression model using training data"""
  
  log_reg = LogisticRegression()
  log_reg.fit(data[FEATURES], data[TARGETS])

  return log_reg


def evalLogReg(log_reg: LogisticRegression, data: dict) -> float:
  """Tests a classical Logistic Regression model using testing data,
  returns accuracy score"""

  return log_reg.score(data[FEATURES], data[TARGETS]) 