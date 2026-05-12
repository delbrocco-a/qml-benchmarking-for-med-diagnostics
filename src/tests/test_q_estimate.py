"""
Unit tests for q_estimate.py — validates the arithmetic formulas used to
estimate quantum hardware runtime from first principles.

Each formula is verified against a hand-calculated value so the test doubles
as documentation of the derivation.
"""

import math
import pytest

from src.q_estimate import (
    _human_time,
    analyse_circuit,
    estimate_kernel_runtime,
    estimate_pegasos_runtime,
    HARDWARE_PROFILES,
)


# ---------------------------------------------------------------------------
# _human_time
# ---------------------------------------------------------------------------

def test_human_time_seconds():
    assert _human_time(0)    == "0.0 seconds"
    assert _human_time(59)   == "59.0 seconds"
    assert _human_time(59.9) == "59.9 seconds"


def test_human_time_minutes():
    assert _human_time(60)   == "1.0 minutes"
    assert _human_time(90)   == "1.5 minutes"
    assert _human_time(3599) == "60.0 minutes"


def test_human_time_hours():
    assert _human_time(3600)  == "1.0 hours"
    assert _human_time(5400)  == "1.5 hours"
    assert _human_time(86399) == "24.0 hours"


def test_human_time_days():
    assert _human_time(86400) == "1.0 days"
    assert _human_time(172800) == "2.0 days"


# ---------------------------------------------------------------------------
# analyse_circuit — structural sanity
# ---------------------------------------------------------------------------

def test_analyse_circuit_returns_positive_counts_for_zzfeaturemap():
    """ZZFeatureMap with 2 qubits and reps=2 requires at least one CX gate."""
    c = analyse_circuit(qubits=2, reps=2, map_name="ZZFeatureMap",
                        hardware="ibm_eagle")
    assert c.depth             > 0
    assert c.single_qubit_gates >= 0
    assert c.two_qubit_gates   > 0,  "ZZFeatureMap must produce CX gates"
    assert c.circuit_time_ns   > 0
    assert c.readout_time_ns   > 0
    # Shot time = circuit + readout
    assert abs(c.total_shot_time_ns -
               (c.circuit_time_ns + c.readout_time_ns)) < 1


def test_analyse_circuit_zfeaturemap_has_no_cx_gates():
    """ZFeatureMap uses only single-qubit rotations — no CX expected."""
    c = analyse_circuit(qubits=2, reps=1, map_name="ZFeatureMap",
                        hardware="ibm_eagle")
    assert c.two_qubit_gates == 0, "ZFeatureMap must not produce CX gates"


def test_analyse_circuit_readout_scales_with_qubits():
    """Readout time must scale linearly with qubit count."""
    profile = HARDWARE_PROFILES["ibm_eagle"]
    r_ns    = profile["readout_ns"]
    for n in (2, 4, 6):
        c = analyse_circuit(qubits=n, reps=1, map_name="ZZFeatureMap",
                            hardware="ibm_eagle")
        assert c.readout_time_ns == n * r_ns, (
            f"Expected readout={n * r_ns} ns for {n} qubits, got {c.readout_time_ns}"
        )


# ---------------------------------------------------------------------------
# estimate_kernel_runtime — kernel evaluation count formula
# ---------------------------------------------------------------------------

def test_kernel_eval_count_upper_triangle_formula():
    """QSVC kernel matrix is symmetric; only n*(n+1)/2 entries need computing.

    Hand-calculated values:
      n=10  -> 10*11//2  =  55
      n=100 -> 100*101//2 = 5050
      n=200 -> 200*201//2 = 20100   (the cap used in this project)
    """
    cases = [(10, 55), (100, 5050), (200, 20100)]
    for n_train, expected_evals in cases:
        est = estimate_kernel_runtime(
            n_train=n_train, qubits=2, reps=1,
            map_name="ZZFeatureMap", hardware="ibm_eagle",
        )
        assert est.n_kernel_evals == expected_evals, (
            f"n_train={n_train}: expected {expected_evals} evals, "
            f"got {est.n_kernel_evals}"
        )


def test_kernel_runtime_total_time_positive_and_grows_with_n():
    """Runtime must be positive and increase with training-set size."""
    est_small = estimate_kernel_runtime(
        n_train=10, qubits=2, reps=1, hardware="ibm_eagle")
    est_large = estimate_kernel_runtime(
        n_train=100, qubits=2, reps=1, hardware="ibm_eagle")
    assert est_small.total_time_s > 0
    assert est_large.total_time_s > est_small.total_time_s


def test_kernel_runtime_manual_calculation():
    """Verify total_time_s against a fully hand-rolled calculation.

    For n_train=4, qubits=2, reps=1, ZZFeatureMap, ibm_eagle, 4096 shots:
      n_evals          = 4*5//2 = 10
      circuit analysis = analyse_circuit(2, 1, ZZFeatureMap, ibm_eagle)
      time_per_eval_ns = 2 * shot_ns * 4096
      total_ns         = 10 * time_per_eval_ns
      total_s          = total_ns / 1e9
    """
    from src.q_estimate import DEFAULT_SHOTS
    c        = analyse_circuit(qubits=2, reps=1, map_name="ZZFeatureMap",
                               hardware="ibm_eagle")
    n_evals  = 4 * 5 // 2          # 10
    time_per = 2 * c.total_shot_time_ns * DEFAULT_SHOTS
    expected = n_evals * time_per / 1e9

    est = estimate_kernel_runtime(
        n_train=4, qubits=2, reps=1,
        map_name="ZZFeatureMap", hardware="ibm_eagle",
    )
    assert est.n_kernel_evals == 10
    assert abs(est.total_time_s - expected) < 1e-9


# ---------------------------------------------------------------------------
# estimate_pegasos_runtime — different eval-count formula
# ---------------------------------------------------------------------------

def test_pegasos_eval_count_formula():
    """Pegasos evals = tau * (1 + n_test).

    Hand-calculated:
      tau=200, n_test=50  -> 200 * 51 = 10200
      tau=100, n_test=100 -> 100 * 101 = 10100
    """
    cases = [(200, 50, 10200), (100, 100, 10100)]
    for tau, n_test, expected in cases:
        est = estimate_pegasos_runtime(
            n_train=200, n_test=n_test, tau=tau,
            qubits=2, reps=1, hardware="ibm_eagle",
        )
        assert est.n_kernel_evals == expected, (
            f"tau={tau}, n_test={n_test}: expected {expected}, "
            f"got {est.n_kernel_evals}"
        )


def test_pegasos_cheaper_than_qsvc_for_equal_n():
    """Pegasos with tau<<n_train must require fewer kernel evals than QSVC.

    This is the key algorithmic advantage claimed in the report: O(tau*n_test)
    vs O(n_train^2).
    """
    n = 200
    qsvc   = estimate_kernel_runtime(n_train=n, qubits=2, reps=1,
                                     hardware="ibm_eagle")
    pegasos = estimate_pegasos_runtime(n_train=n, n_test=50, tau=200,
                                       qubits=2, reps=1, hardware="ibm_eagle")
    assert pegasos.n_kernel_evals < qsvc.n_kernel_evals, (
        "Pegasos should require fewer kernel evals than QSVC at equal n_train"
    )
