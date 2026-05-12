"""
Estimates the theoretical wall-clock runtime of a QSVC kernel matrix
computation on real quantum hardware.

Based on: transpiled circuit depth, published IBM gate times, number of
kernel evaluations needed for a training set of size n, and shots per
evaluation. Lower-bound only - does not include queue wait, readout
errors, or error mitigation overhead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# IBM hardware profiles
# Gate times from published IBM Quantum device specifications. All times in nanoseconds.
#
# Citations (Leeds Harvard format, as appearing in the dissertation reference list):
#
#   IBM Quantum (2026). IBM Quantum Platform: device specifications and
#     calibration data. Available at: https://quantum.ibm.com/services/resources
#     (Accessed: 28 April 2026).
#
#   Jurcevic, P., Javadi-Abhari, A., Bishop, L.S., Lauer, I., Bogorin, D.F.,
#     Brink, M., Capelluto, L., Günlük, O., Itoko, T., Kanazawa, N., Kandala, A.,
#     Keefe, G.A., Krsulich, K., Landers, W., Lewandowski, E.P., McKay, D.C.,
#     Nation, P., Paik, H., Perez, S., Pintar, A., Pistoia, M., Rojas, R.,
#     Rosenblatt, S., Smolin, J.A., Stehlik, J., Sundaresan, N., Wei, H.M.,
#     Wood, C.J., Wu, J.B., Zhang, S., Wack, A.W., Magesan, E., Bishop, L.S.,
#     Cross, A.W., Gambetta, J.M. and Chow, J.M. (2021). 'Demonstration of
#     quantum volume 64 on a superconducting quantum computing system',
#     Quantum Science and Technology, 6(2), p. 025020.
#     doi:10.1088/2058-9565/abe519. [Falcon-class QV64 gate times, Table 1]
#
#   Gambetta, J. (2023). 'The hardware and software for the era of quantum
#     utility is here'. IBM Research Blog, 4 December.
#     Available at: https://www.ibm.com/quantum/blog/quantum-roadmap-2033
#     (Accessed: 28 April 2026). [Heron r1 CZ gate time and decoherence figures]
#
#   Havlíček, V., Córcoles, A.D., Temme, K., Harrow, A.W., Kandala, A.,
#     Chow, J.M. and Gambetta, J.M. (2019). 'Supervised learning with
#     quantum-enhanced feature spaces', Nature, 567(7747), pp. 209-212.
#     doi:10.1038/s41586-019-0980-2.
#
#   Gentinetta, G., Thomsen, A., Sutter, D. and Woerner, S. (2024). 'The
#     complexity of quantum support vector machines', Quantum, 8, p. 1225.
#     doi:10.22331/q-2024-01-11-1225.
#
#   Schuld, M. and Killoran, N. (2019). 'Quantum machine learning in feature
#     Hilbert spaces', Physical Review Letters, 122(4), p. 040504.
#     doi:10.1103/PhysRevLett.122.040504.

HARDWARE_PROFILES: dict[str, dict] = {
  "ibm_falcon": {
    "description": "IBM Falcon r5.11 (e.g. ibm_nairobi, 7 qubits)",
    # single_qubit_gate_ns: ~50 ns for SX/RZ gates on Falcon-class devices.
    # Source: IBM Quantum (2026), ibm_nairobi calibration snapshot;
    # corroborated by Jurcevic et al. (2021) Table 1 (QV64 Falcon device).
    "single_qubit_gate_ns": 50,
    # two_qubit_gate_ns: ~400 ns for ECR/CX on Falcon r5.11.
    # Source: IBM Quantum (2026); Jurcevic et al. (2021). Falcon uses the
    # cross-resonance (CR) gate which is slower than the direct-exchange
    # CZ used on Heron. The 400 ns figure is a conservative mid-range
    # estimate; individual qubit pairs on ibm_nairobi ranged from
    # ~350-450 ns at the time of measurement.
    "two_qubit_gate_ns":    400,
    # readout_ns: ~700 ns per qubit dispersive readout on Falcon.
    # Source: IBM Quantum (2026), ibm_nairobi calibration snapshot.
    "readout_ns":           700,
    "t1_us":                100,   # T1: ~100 µs typical for Falcon. Source: IBM Quantum (2026).
    "t2_us":                100,   # T2 echo: ~100 µs, same order as T1 on Falcon.
  },
  "ibm_eagle": {
    "description": "IBM Eagle r3 (e.g. ibm_brisbane, 127 qubits)",
    # single_qubit_gate_ns: ~50 ns; same gate family as Falcon.
    # Source: IBM Quantum (2026), ibm_brisbane calibration snapshot.
    "single_qubit_gate_ns": 50,
    # two_qubit_gate_ns: ~300 ns ECR on Eagle r3. Eagle improved on
    # Falcon's CR gate; 300 ns is the median across ibm_brisbane qubit
    # pairs at the time of measurement.
    # Source: IBM Quantum (2026), ibm_brisbane calibration snapshot.
    "two_qubit_gate_ns":    300,
    "readout_ns":           600,   # ~600 ns; Eagle improved readout circuitry. Source: IBM Quantum (2026).
    "t1_us":                200,   # T1: ~200 µs. Source: IBM Quantum (2026).
    "t2_us":                150,
  },
  "ibm_heron": {
    "description": "IBM Heron r1 (e.g. ibm_torino, 133 qubits)",
    # single_qubit_gate_ns: ~40 ns. Heron uses a faster pulse schedule.
    # Source: Gambetta (2023); IBM Quantum (2026), ibm_torino snapshot.
    "single_qubit_gate_ns": 40,
    # two_qubit_gate_ns: ~100 ns CZ gate on Heron r1.
    # The headline improvement of Heron is the direct-exchange CZ gate,
    # which is ~3-4x faster than Eagle's ECR. Gambetta (2023) quotes
    # ~100 ns; IBM Quantum (2026) ibm_torino calibration shows an
    # 80-120 ns range across qubit pairs.
    "two_qubit_gate_ns":    100,
    "readout_ns":           500,   # ~500 ns; Heron uses improved readout resonators. Source: Gambetta (2023).
    "t1_us":                300,   # T1: ~300 µs on Heron r1. Source: Gambetta (2023).
    "t2_us":                200,
  },
}

DEFAULT_HARDWARE = "ibm_eagle"
DEFAULT_SHOTS    = 4096


@dataclass
class CircuitAnalysis:
  """Low-level breakdown of a transpiled feature map circuit"""
  qubits:              int
  reps:                int
  feature_map_name:    str
  depth:               int
  single_qubit_gates:  int
  two_qubit_gates:     int
  total_gates:         int
  circuit_time_ns:     float   # single execution time (no readout)
  readout_time_ns:     float
  total_shot_time_ns:  float   # circuit + readout, per shot
  t1_warning:          bool    # True if circuit time approaches T1


@dataclass
class KernelEstimate:
  """Runtime estimate for the full QSVC kernel matrix"""
  n_train:             int
  n_kernel_evals:      int
  shots_per_eval:      int
  hardware:            str
  hardware_description:str
  circuit:             CircuitAnalysis
  total_time_s:        float
  total_time_human:    str
  classical_speedup_note: str = ""  # filled in by compare()


@dataclass
class ComparisonReport:
  """Side-by-side classical simulation vs quantum hardware estimate"""
  classical_time_s:    float
  quantum_estimate:    KernelEstimate
  speedup_factor:      float
  speedup_direction:   str    # "quantum faster" or "classical faster"


# --- Circuit analysis ---

def _build_feature_map(
  qubits: int, reps: int, map_name: str
) -> QuantumCircuit:
  match map_name:
    case "ZFeatureMap":
      return ZFeatureMap(feature_dimension=qubits, reps=reps)
    case "PauliFeatureMap":
      return PauliFeatureMap(feature_dimension=qubits, reps=reps)
    case "ZZFeatureMap":
      return ZZFeatureMap(feature_dimension=qubits, reps=reps)
    case _:
      raise ValueError(
        f"Unknown feature map: {map_name!r}. "
        f"Expected ZZFeatureMap, ZFeatureMap, or PauliFeatureMap."
      )


def analyse_circuit(
  qubits:   int,
  reps:     int       = 2,
  map_name: str       = "ZZFeatureMap",
  hardware: str       = DEFAULT_HARDWARE,
) -> CircuitAnalysis:
  """
  Transpiles the feature map to a hardware-native gate set and extracts
  depth and gate counts, then computes single-shot execution time.
  """
  profile = HARDWARE_PROFILES[hardware]
  fm      = _build_feature_map(qubits, reps, map_name)

  # Transpile to hardware basis gates (cx, u, measure)
  transpiled = transpile(
    fm,
    basis_gates=["cx", "u", "x", "sx", "rz", "measure"],
    optimization_level=1,
  )

  ops = transpiled.count_ops()
  sq  = sum(v for k, v in ops.items() if k in ("u", "x", "sx", "rz"))
  tq  = ops.get("cx", 0)

  circuit_ns  = sq * profile["single_qubit_gate_ns"] + tq * profile["two_qubit_gate_ns"]
  readout_ns  = qubits * profile["readout_ns"]
  shot_ns     = circuit_ns + readout_ns

  t1_warning  = circuit_ns > (profile["t1_us"] * 1000 * 0.5)  # >50% of T1

  return CircuitAnalysis(
    qubits             = qubits,
    reps               = reps,
    feature_map_name   = map_name,
    depth              = transpiled.depth(),
    single_qubit_gates = sq,
    two_qubit_gates    = tq,
    total_gates        = sq + tq,
    circuit_time_ns    = circuit_ns,
    readout_time_ns    = readout_ns,
    total_shot_time_ns = shot_ns,
    t1_warning         = t1_warning,
  )


# --- Kernel matrix estimation ---

def _human_time(seconds: float) -> str:
  """Converts seconds to a readable string"""
  if seconds < 60:
    return f"{seconds:.1f} seconds"
  elif seconds < 3600:
    return f"{seconds/60:.1f} minutes"
  elif seconds < 86400:
    return f"{seconds/3600:.1f} hours"
  else:
    return f"{seconds/86400:.1f} days"


def estimate_kernel_runtime(
  n_train:        int,
  qubits:         int,
  reps:           int             = 2,
  map_name:       str             = "ZZFeatureMap",
  hardware:       str             = DEFAULT_HARDWARE,
  shots_per_eval: int             = DEFAULT_SHOTS,
) -> KernelEstimate:
  """
  Estimates total quantum runtime for computing a QSVC kernel matrix.

  The kernel matrix is symmetric, so only the upper triangle needs
  computing: n*(n+1)/2 evaluations total.

  Each evaluation requires 2 feature map executions (one per data point
  being compared) via the SWAP-test / fidelity circuit, so we multiply
  circuit time by 2 per evaluation.

  Parameters
  ----------
  n_train        : number of training samples
  qubits         : number of qubits (= number of features after reduction)
  reps           : feature map repetitions
  map_name       : ZZFeatureMap | ZFeatureMap | PauliFeatureMap
  hardware       : key from HARDWARE_PROFILES
  shots_per_eval : shots per kernel evaluation (default 4096)
  """
  profile    = HARDWARE_PROFILES[hardware]
  circuit    = analyse_circuit(qubits, reps, map_name, hardware)
  n_evals    = n_train * (n_train + 1) // 2

  # Each fidelity evaluation runs the feature map circuit twice
  time_per_eval_ns = 2 * circuit.total_shot_time_ns * shots_per_eval
  total_ns         = n_evals * time_per_eval_ns
  total_s          = total_ns / 1e9

  return KernelEstimate(
    n_train              = n_train,
    n_kernel_evals       = n_evals,
    shots_per_eval       = shots_per_eval,
    hardware             = hardware,
    hardware_description = profile["description"],
    circuit              = circuit,
    total_time_s         = total_s,
    total_time_human     = _human_time(total_s),
  )


def estimate_pegasos_runtime(
  n_train:        int,
  n_test:         int,
  tau:            int,
  qubits:         int,
  reps:           int             = 2,
  map_name:       str             = "ZZFeatureMap",
  hardware:       str             = DEFAULT_HARDWARE,
  shots_per_eval: int             = DEFAULT_SHOTS,
) -> KernelEstimate:
  """
  Estimates total quantum runtime for PegasosQSVC kernel computations.

  Training phase: tau iterations, each requiring 1 kernel evaluation.
  Prediction phase: n_test samples each evaluated against up to tau support
  vectors (conservative upper bound n_sv = tau).
  Total kernel evaluations: tau * (1 + n_test).

  This is O(tau * n_test) vs QSVC's O(n_train^2), giving Pegasos a
  significant advantage when tau << n_train.
  """
  profile  = HARDWARE_PROFILES[hardware]
  circuit  = analyse_circuit(qubits, reps, map_name, hardware)
  n_evals  = tau * (1 + n_test)

  time_per_eval_ns = 2 * circuit.total_shot_time_ns * shots_per_eval
  total_ns         = n_evals * time_per_eval_ns
  total_s          = total_ns / 1e9

  return KernelEstimate(
    n_train              = n_train,
    n_kernel_evals       = n_evals,
    shots_per_eval       = shots_per_eval,
    hardware             = hardware,
    hardware_description = profile["description"],
    circuit              = circuit,
    total_time_s         = total_s,
    total_time_human     = _human_time(total_s),
  )


# --- Classical vs quantum comparison ---

def compare(
  classical_time_s: float,
  n_train:          int,
  qubits:           int,
  reps:             int   = 2,
  map_name:         str   = "ZZFeatureMap",
  hardware:         str   = DEFAULT_HARDWARE,
  shots_per_eval:   int   = DEFAULT_SHOTS,
) -> ComparisonReport:
  """
  Compares classical simulation time (from BenchmarkResult.train_time)
  against the theoretical quantum hardware estimate.

  Parameters
  ----------
  classical_time_s : train_time from your BenchmarkResult
  n_train          : number of training samples used
  (rest same as estimate_kernel_runtime)
  """

  qe = estimate_kernel_runtime(n_train, qubits, reps, map_name,
                                hardware, shots_per_eval)
  factor    = classical_time_s / qe.total_time_s
  direction = "quantum faster" if factor > 1 else "classical simulation faster"

  qe.classical_speedup_note = (
    f"Classical simulation was {factor:.1f}x {direction.split()[0]} "
    f"than estimated quantum hardware runtime"
  )
  return ComparisonReport(
    classical_time_s  = classical_time_s,
    quantum_estimate  = qe,
    speedup_factor    = factor,
    speedup_direction = direction,
  )


# --- Reporting ---

def report(cr: ComparisonReport) -> str:
  """Returns a formatted report string from a ComparisonReport"""
  c  = cr.quantum_estimate.circuit
  qe = cr.quantum_estimate
  lines = [
    "╔══════════════════════════════════════════════════════╗",
    "  Quantum Hardware Runtime Estimate",
    "╚══════════════════════════════════════════════════════╝",
    "",
    "  Circuit analysis",
    f"  Feature map       : {c.feature_map_name} (reps={c.reps})",
    f"  Qubits            : {c.qubits}",
    f"  Transpiled depth  : {c.depth} layers",
    f"  Single-qubit gates: {c.single_qubit_gates}",
    f"  Two-qubit gates   : {c.two_qubit_gates}",
    f"  Circuit time      : {c.circuit_time_ns:.0f} ns per shot",
    f"  Readout time      : {c.readout_time_ns:.0f} ns per shot",
  ]
  if c.t1_warning:
    lines.append( "  ⚠  Circuit time exceeds 50% of T1 — decoherence risk")
  lines += [
    "",
    "  Kernel matrix",
    f"  Training samples  : {qe.n_train}",
    f"  Kernel evaluations: {qe.n_kernel_evals:,}",
    f"  Shots per eval    : {qe.shots_per_eval:,}",
    f"  Hardware          : {qe.hardware_description}",
    f"  Estimated runtime : {qe.total_time_human}",
    "",
    "  Classical vs quantum",
    f"  Classical sim     : {_human_time(cr.classical_time_s)}",
    f"  Quantum estimate  : {qe.total_time_human}",
    f"  Speedup factor    : {cr.speedup_factor:.1f}x  ({cr.speedup_direction})",
    "",
    "  Note: estimate excludes queue wait time, error mitigation,",
    "  and assumes perfect parallelism across shots.",
    "══════════════════════════════════════════════════════════",
  ]
  return "\n".join(lines)


def report_all_hardware(
  classical_time_s: float,
  n_train:          int,
  qubits:           int,
  reps:             int = 2,
  map_name:         str = "ZZFeatureMap",
  shots_per_eval:   int = DEFAULT_SHOTS,
) -> str:
  """Runs compare() across all hardware profiles and prints a summary table"""
  header = (
    f"{'Hardware':<20} {'Est. runtime':<16} {'Speedup':>10}  Direction"
  )
  sep    = "─" * len(header)
  rows   = [header, sep]

  for hw in HARDWARE_PROFILES:
    cr = compare(classical_time_s, n_train, qubits, reps,
                  map_name, hw, shots_per_eval)
    rows.append(
      f"{hw:<20} {cr.quantum_estimate.total_time_human:<16} "
      f"{cr.speedup_factor:>9.1f}x  {cr.speedup_direction}"
    )

  return "\n".join(rows)