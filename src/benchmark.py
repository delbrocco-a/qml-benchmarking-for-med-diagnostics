import time
import functools
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from src.load_data import TESTING, FEATURES, TARGETS


@dataclass
class BenchmarkResult:
  """Stores the results of a single model benchmark run"""

  model_name: str
  train_time: float           # seconds
  eval_time: float            # seconds
  accuracy: float             # 0.0 - 1.0
  extra: dict = field(default_factory=dict)  # optional metadata (e.g. kernel, n_components)

  def __str__(self) -> str:
    lines = [
      f"[{self.model_name}]",
      f"  Train time : {self.train_time:.4f}s",
      f"  Eval time  : {self.eval_time:.4f}s",
      f"  Accuracy   : {self.accuracy * 100:.2f}%",
    ]
    if self.extra:
      for k, v in self.extra.items():
        lines.append(f"  {k:<11}: {v}")
    return "\n".join(lines)


class Benchmark:
    """Collects and reports BenchmarkResults across multiple model runs"""

    def __init__(self):
      self.results: list[BenchmarkResult] = []

    def run(
      self,
      model_name: str,
      train_fn: Callable,
      eval_fn: Callable,
      train_data: dict,
      test_data: dict,
      extra: Optional[dict] = None,
    ) -> BenchmarkResult:
      """
      Trains and evaluates a model, recording timing and accuracy.

      Parameters
      ----------
      model_name : str
          Human-readable label for the model (e.g. "QSVC (ZZFeatureMap)")
      train_fn : Callable
          A zero-argument callable that trains the model and returns it.
          e.g. lambda: trainSVC(train_data, kernel="rbf")
      eval_fn : Callable
          A one-argument callable that takes the trained model and returns
          a float accuracy score.
          e.g. lambda model: evalSVC(model, test_data)
      train_data : dict
          Training split dict (used only for reference/extra metadata).
      test_data : dict
          Testing split dict (used only for reference/extra metadata).
      extra : dict, optional
          Any additional metadata to attach to the result (e.g. kernel name,
          n_components, qubit count).

      Returns
      -------
      BenchmarkResult
      """
      # --- Train ---
      t0 = time.perf_counter()
      model = train_fn()
      train_time = time.perf_counter() - t0

      # --- Evaluate ---
      t1 = time.perf_counter()
      accuracy = eval_fn(model)
      eval_time = time.perf_counter() - t1

      result = BenchmarkResult(
        model_name=model_name,
        train_time=train_time,
        eval_time=eval_time,
        accuracy=accuracy,
        extra=extra or {},
      )
      self.results.append(result)
      return result

    def summary(self) -> str:
      """Returns a formatted summary table of all recorded results"""
      if not self.results:
        return "No benchmark results recorded."

      header = f"{'Model':<30} {'Train (s)':>10} {'Eval (s)':>10} {'Accuracy':>10}"
      sep = "-" * len(header)
      rows = [header, sep]

      for r in self.results:
        rows.append(
          f"{r.model_name:<30} {r.train_time:>10.4f} {r.eval_time:>10.4f} {r.accuracy * 100:>9.2f}%"
        )

      best = max(self.results, key=lambda r: r.accuracy)
      fastest = min(self.results, key=lambda r: r.train_time)
      rows += [
        sep,
        f"Best accuracy : {best.model_name} ({best.accuracy * 100:.2f}%)",
        f"Fastest train : {fastest.model_name} ({fastest.train_time:.4f}s)",
      ]
      return "\n".join(rows)

    def clear(self):
      """Clears all stored results"""
      self.results = []


def timed(label: Optional[str] = None):
    """
    Decorator that prints execution time for any function.
    Useful for quick ad-hoc timing outside of the Benchmark class.

    Usage:
        @timed("My function")
        def my_fn(): ...
    """
    
    def decorator(fn: Callable) -> Callable:
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        name = label or fn.__name__
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[timed] {name}: {elapsed:.4f}s")
        return result
      return wrapper
    return decorator