"""
quantum_estimate.py
───────────────────
Estimates the theoretical wall-clock runtime of a QSVC kernel matrix
computation on a real quantum computer, based on:

  1. Transpiled circuit depth & gate counts for the feature map
  2. Published IBM Falcon/Eagle/Heron processor gate times
  3. Number of kernel evaluations required for a training set of size n
  4. Shots per evaluation (for fidelity estimation)

This is a theoretical lower-bound estimate — it does not account for
queue wait times, readout errors, or error mitigation overhead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ── IBM Hardware Profiles ─────────────────────────────────────────────────
# Gate times sourced from IBM Quantum published device specifications.
# All times in nanoseconds.
# Refs:
#   IBM Falcon r5.11 (e.g. ibm_nairobi)  — IBM Quantum documentation 2023
#   IBM Eagle r3     (e.g. ibm_brisbane) — IBM Quantum documentation 2023
#   IBM Heron r1     (e.g. ibm_torino)   — IBM Quantum documentation 2024

HARDWARE_PROFILES: dict[str, dict] = {
  "ibm_falcon": {
    "description": "IBM Falcon r5.11 (e.g. ibm_nairobi, 7 qubits)",
    "single_qubit_gate_ns": 50,
    "two_qubit_gate_ns":    400,
    "readout_ns":           700,
    "t1_us":                100,   # decoherence time T1 (microseconds)
    "t2_us":                100,
  },
  "ibm_eagle": {
    "description": "IBM Eagle r3 (e.g. ibm_brisbane, 127 qubits)",
    "single_qubit_gate_ns": 50,
    "two_qubit_gate_ns":    300,
    "readout_ns":           600,
    "t1_us":                200,
    "t2_us":                150,
  },
  "ibm_heron": {
    "description": "IBM Heron r1 (e.g. ibm_torino, 133 qubits)",
    "single_qubit_gate_ns": 40,
    "two_qubit_gate_ns":    100,   # CZ gate — significantly faster
    "readout_ns":           500,
    "t1_us":                300,
    "t2_us":                200,
  },
}

"""
Gate times / hardware specs:

IBM Quantum (2023). IBM Quantum system two and Heron processor. https://www.ibm.com/quantum/blog/ibm-quantum-roadmap-2025
Jurcevic, P. et al. (2021). Demonstration of quantum volume 64 on a superconducting quantum computing system. Quantum Science and Technology, 6(2), 025020. — this is the most citable peer-reviewed source for Falcon processor characteristics.

Quantum kernel methods / fidelity circuit structure:

Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced feature spaces. Nature, 567, 209–212. — the foundational paper, almost certainly already in your bibliography.
Schuld, M. & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. Physical Review Letters, 122, 040504.

ZZFeatureMap specifically:

Havlíček et al. (2019) above covers this directly, as ZZFeatureMap is the circuit they proposed.
"""

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


# ── Circuit Analysis ──────────────────────────────────────────────────────

def _build_feature_map(
  qubits: int, reps: int, map_name: str
) -> QuantumCircuit:
  match map_name:
    case "ZFeatureMap":
      return ZFeatureMap(feature_dimension=qubits, reps=reps)
    case "PauliFeatureMap":
      return PauliFeatureMap(feature_dimension=qubits, reps=reps)
    case _:
      return ZZFeatureMap(feature_dimension=qubits, reps=reps)


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


# ── Kernel Matrix Estimation ──────────────────────────────────────────────

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


# ── Classical vs Quantum Comparison ──────────────────────────────────────

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


# ── Reporting ─────────────────────────────────────────────────────────────

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