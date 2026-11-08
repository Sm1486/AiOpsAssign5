"""
test_gate.py -- pytest suite for decide() in eta_mlflow.py (Part B2).

Run from partB/:
    pytest test_gate.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from eta_mlflow import decide

# Default thresholds for reference
DEFAULTS = dict(min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05)

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 8.0}
CAND_BETTER = {"mae": 9.5, "mae_rush": 11.5, "mae_rain": 10.5, "p95_ms": 7.0}  # should promote


# ---------------------------------------------------------------------------
# 1. Clear promote: candidate better on all criteria
# ---------------------------------------------------------------------------
def test_all_criteria_pass():
    res = decide(PROD, CAND_BETTER)
    assert res["promote"] is True
    assert all(res["criteria"].values())


# ---------------------------------------------------------------------------
# 2. No promote when MAE does not improve enough (boundary: exactly at threshold)
# ---------------------------------------------------------------------------
def test_mae_improvement_exact_boundary_fails():
    # gain = (10.0 - 9.7) / 10.0 = 0.03 exactly → passes (>= semantics)
    cand = {**CAND_BETTER, "mae": 9.7}
    res = decide(PROD, cand, min_rel_gain=0.03)
    assert res["criteria"]["mae_improvement"] is True


# ---------------------------------------------------------------------------
# 3. Fails just below threshold
# ---------------------------------------------------------------------------
def test_mae_improvement_just_below_fails():
    # gain = (10.0 - 9.71) / 10.0 = 0.029 < 0.03
    cand = {**CAND_BETTER, "mae": 9.71}
    res = decide(PROD, cand, min_rel_gain=0.03)
    assert res["criteria"]["mae_improvement"] is False
    assert res["promote"] is False


# ---------------------------------------------------------------------------
# 4. No promote when latency too high
# ---------------------------------------------------------------------------
def test_latency_fails():
    cand = {**CAND_BETTER, "p95_ms": 10.1}
    res = decide(PROD, cand, max_p95_ms=10.0)
    assert res["criteria"]["latency_ok"] is False
    assert res["promote"] is False


# ---------------------------------------------------------------------------
# 5. Latency at exact threshold passes
# ---------------------------------------------------------------------------
def test_latency_exact_boundary_passes():
    cand = {**CAND_BETTER, "p95_ms": 10.0}
    res = decide(PROD, cand, max_p95_ms=10.0)
    assert res["criteria"]["latency_ok"] is True


# ---------------------------------------------------------------------------
# 6. Rush-hour slice regression blocks promotion
# ---------------------------------------------------------------------------
def test_rush_regression_fails():
    # mae_rush goes from 12.0 (prod) to 17.5 (cand): regression = 5.5 > 0.05
    cand = {**CAND_BETTER, "mae_rush": 17.5}
    res = decide(PROD, cand)
    assert res["criteria"]["no_rush_regression"] is False
    assert res["promote"] is False


# ---------------------------------------------------------------------------
# 7. Rain slice regression blocks promotion
# ---------------------------------------------------------------------------
def test_rain_regression_fails():
    cand = {**CAND_BETTER, "mae_rain": 16.5}
    res = decide(PROD, cand)
    assert res["criteria"]["no_rain_regression"] is False
    assert res["promote"] is False


# ---------------------------------------------------------------------------
# 8. Slice regression exactly at max_slice_regression passes
# ---------------------------------------------------------------------------
def test_slice_regression_exact_boundary():
    # rain regression = 0.05 exactly → passes
    cand = {**CAND_BETTER, "mae_rain": PROD["mae_rain"] + 0.05}
    res = decide(PROD, cand, max_slice_regression=0.05)
    assert res["criteria"]["no_rain_regression"] is True


# ---------------------------------------------------------------------------
# 9. Slice regression one epsilon above threshold fails
# ---------------------------------------------------------------------------
def test_slice_regression_just_above_fails():
    cand = {**CAND_BETTER, "mae_rain": PROD["mae_rain"] + 0.051}
    res = decide(PROD, cand, max_slice_regression=0.05)
    assert res["criteria"]["no_rain_regression"] is False
    assert res["promote"] is False


# ---------------------------------------------------------------------------
# 10. Candidate with worse overall MAE is not promoted
# ---------------------------------------------------------------------------
def test_candidate_worse_mae_no_promotion():
    cand = {**CAND_BETTER, "mae": 11.0}  # worse than prod's 10.0
    res = decide(PROD, cand)
    assert res["criteria"]["mae_improvement"] is False
    assert res["promote"] is False


# ---------------------------------------------------------------------------
# 11. Very aggressive gain threshold blocks even a good candidate
# ---------------------------------------------------------------------------
def test_aggressive_threshold_no_promotion():
    res = decide(PROD, CAND_BETTER, min_rel_gain=0.60)
    assert res["criteria"]["mae_improvement"] is False
    assert res["promote"] is False


# ---------------------------------------------------------------------------
# 12. decide() returns required keys
# ---------------------------------------------------------------------------
def test_result_structure():
    res = decide(PROD, CAND_BETTER)
    assert "criteria" in res
    assert "promote" in res
    assert isinstance(res["criteria"], dict)
    assert isinstance(res["promote"], bool)
    expected_keys = {"mae_improvement", "latency_ok", "no_rush_regression", "no_rain_regression"}
    assert expected_keys.issubset(set(res["criteria"].keys()))

