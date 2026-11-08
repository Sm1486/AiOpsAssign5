"""
test_eta_app.py -- pytest suite for eta_app.py (Part A2).

Run from the partA/ directory (after generating the model files):
    ETA_MODEL=../data/eta_v1.joblib pytest test_eta_app.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient

# Point at a real model file; skip gracefully if data not generated yet.
os.environ.setdefault("ETA_MODEL", "../data/eta_v1.joblib")

from eta_app import app, model, _model_version  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

VALID_INSTANCE = {
    "distance_km": 25.0,
    "n_stops": 5,
    "hour": 10,
    "weekday": 1,
    "traffic_index": 0.3,
    "rain": 0,
    "weight_kg": 10.0,
}

PREDICT_URL = "/v1/models/eta:predict"


# ---------------------------------------------------------------------------
# 1. Valid single-row request
# ---------------------------------------------------------------------------
def test_valid_single_request():
    r = client.post(PREDICT_URL, json={"instances": [VALID_INSTANCE]})
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert isinstance(body["predictions"][0], float)
    assert body["model_version"] == _model_version
    assert "request_id" in body


# ---------------------------------------------------------------------------
# 2. Batch order preserved
# ---------------------------------------------------------------------------
def test_batch_order_preserved():
    inst_a = {**VALID_INSTANCE, "distance_km": 5.0}   # short → smaller ETA
    inst_b = {**VALID_INSTANCE, "distance_km": 100.0}  # long  → larger ETA
    r = client.post(PREDICT_URL, json={"instances": [inst_a, inst_b]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0] < preds[1], "shorter distance should give lower ETA"


# ---------------------------------------------------------------------------
# 3. Invalid input – out-of-range distance_km
# ---------------------------------------------------------------------------
def test_invalid_distance_km():
    bad = {**VALID_INSTANCE, "distance_km": -1.0}
    r = client.post(PREDICT_URL, json={"instances": [bad]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"


# ---------------------------------------------------------------------------
# 4. Invalid input – bad rain value
# ---------------------------------------------------------------------------
def test_invalid_rain_value():
    bad = {**VALID_INSTANCE, "rain": 5}
    r = client.post(PREDICT_URL, json={"instances": [bad]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 5. Invalid input – hour out of range
# ---------------------------------------------------------------------------
def test_invalid_hour():
    bad = {**VALID_INSTANCE, "hour": 25}
    r = client.post(PREDICT_URL, json={"instances": [bad]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 6. Cross-field infeasibility rule
# ---------------------------------------------------------------------------
def test_cross_field_infeasible():
    bad = {**VALID_INSTANCE, "distance_km": 200.0, "n_stops": 25}
    r = client.post(PREDICT_URL, json={"instances": [bad]})
    assert r.status_code == 422
    # Must mention the word "infeasible"
    assert "infeasible" in r.text.lower()


# ---------------------------------------------------------------------------
# 7. Batch size too small (empty list)
# ---------------------------------------------------------------------------
def test_batch_size_too_small():
    r = client.post(PREDICT_URL, json={"instances": []})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 8. Batch size exactly at upper limit (500)
# ---------------------------------------------------------------------------
def test_batch_size_max_limit():
    batch = [VALID_INSTANCE] * 500
    r = client.post(PREDICT_URL, json={"instances": batch})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 500


# ---------------------------------------------------------------------------
# 9. Request-ID echo – client header present
# ---------------------------------------------------------------------------
def test_request_id_echo():
    my_id = "test-id-12345"
    r = client.post(
        PREDICT_URL,
        json={"instances": [VALID_INSTANCE]},
        headers={"X-Request-ID": my_id},
    )
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == my_id
    assert r.json()["request_id"] == my_id


# ---------------------------------------------------------------------------
# 10. Request-ID generated when absent
# ---------------------------------------------------------------------------
def test_request_id_generated():
    r = client.post(PREDICT_URL, json={"instances": [VALID_INSTANCE]})
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    # Must be a UUID-shaped string (at least 30 chars)
    assert len(r.headers["X-Request-ID"]) >= 30


# ---------------------------------------------------------------------------
# 11. Request-ID present on 422 error response
# ---------------------------------------------------------------------------
def test_request_id_in_error():
    bad = {**VALID_INSTANCE, "distance_km": -5.0}
    my_id = "err-req-id-999"
    r = client.post(
        PREDICT_URL,
        json={"instances": [bad]},
        headers={"X-Request-ID": my_id},
    )
    assert r.status_code == 422
    body = r.json()
    assert body.get("request_id") == my_id
    assert r.headers.get("X-Request-ID") == my_id


# ---------------------------------------------------------------------------
# 12. /metrics endpoint – Prometheus counters and histogram present
# ---------------------------------------------------------------------------
def test_metrics_endpoint():
    # Make one successful request first so counters are non-zero
    client.post(PREDICT_URL, json={"instances": [VALID_INSTANCE]})
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "eta_requests_total" in text
    assert "eta_request_latency_seconds_bucket" in text
    assert "eta_request_latency_seconds_sum" in text
    assert "eta_request_latency_seconds_count" in text
    # Infinity bucket must be present
    assert '+Inf' in text


# ---------------------------------------------------------------------------
# 13. /healthz returns ok
# ---------------------------------------------------------------------------
def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body


# ---------------------------------------------------------------------------
# 14. Generic 500 when model raises (monkeypatched)
# ---------------------------------------------------------------------------
def test_model_raise_returns_500(monkeypatch):
    import eta_app as ea

    def boom(X):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(ea.model, "predict", boom)
    r = client.post(PREDICT_URL, json={"instances": [VALID_INSTANCE]})
    assert r.status_code == 500
    body = r.json()
    # Must NOT leak internals
    assert "simulated model failure" not in r.text
    assert "request_id" in body

