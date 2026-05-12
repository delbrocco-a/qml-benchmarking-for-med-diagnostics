"""
Pytest wrapper for the Havlíček sanity check.

Runs on every push to catch regressions in the quantum kernel pipeline —
if QSVC can't separate make_circles with ZZFeatureMap something is broken
in the circuit construction, the kernel evaluation, or the scaling.
"""

from src.sanity_havlicek import run_havlicek_check, ACCURACY_THRESHOLD


def test_zz_feature_map_separates_circles():
    """QSVC with ZZFeatureMap must reach >{threshold:.0%} on make_circles.

    This reproduces the result claimed in Havlíček et al. (2019) — that
    ZZ-based quantum kernels can separate datasets classical kernels cannot.
    If this fails, the quantum pipeline is broken regardless of benchmark
    results.
    """.format(threshold=ACCURACY_THRESHOLD)
    bacc = run_havlicek_check(verbose=False)
    assert bacc >= ACCURACY_THRESHOLD, (
        f"PIPELINE BROKEN — ZZFeatureMap scored {bacc:.3f} on make_circles "
        f"(expected ≥{ACCURACY_THRESHOLD}). "
        "Check FQKernel construction, QSVC wrapper, and [0,π] scaling."
    )
