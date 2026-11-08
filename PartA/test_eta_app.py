import os
from fastapi.testclient import TestClient

os.environ.setdefault("ETA_MODEL", "../data/eta_v1.joblib")

from eta_app import app, model, model_version
import eta_app as ea

client = TestClient(app, raise_server_exceptions=False)

EXAMPLE_VALID_INSTANCE = {"distance_km": 25.0, "n_stops": 5, "hour": 10, "weekday": 1, "traffic_index": 0.3, "rain": 0, "weight_kg": 10.0}
PREDICT_URL = "/v1/models/eta:predict"

def test_valid_single_request():
    r = client.post(PREDICT_URL, json={"instances": [EXAMPLE_VALID_INSTANCE]})
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert isinstance(body["predictions"][0], float)
    assert body["model_version"] == model_version
    assert "request_id" in body

def test_batch_order_preserved():
    inst_a = {**EXAMPLE_VALID_INSTANCE, "distance_km": 5.0}
    inst_b = {**EXAMPLE_VALID_INSTANCE, "distance_km": 100.0}
    r = client.post(PREDICT_URL, json={"instances": [inst_a, inst_b]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0] < preds[1], "shorter distance should give lower ETA"

def test_invalid_distance_km():
    invalid_distance_case = {**EXAMPLE_VALID_INSTANCE, "distance_km": -1.0}
    r = client.post(PREDICT_URL, json={"instances": [invalid_distance_case]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"

def test_invalid_rain_value():
    invalid_rain_case = {**EXAMPLE_VALID_INSTANCE, "rain": 5}
    r = client.post(PREDICT_URL, json={"instances": [invalid_rain_case]})
    assert r.status_code == 422

def test_invalid_hour():
    invalid_hour_case = {**EXAMPLE_VALID_INSTANCE, "hour": 25}
    r = client.post(PREDICT_URL, json={"instances": [invalid_hour_case]})
    assert r.status_code == 422

def test_cross_field_infeasible():
    invalid_distance_n_stop_case = {**EXAMPLE_VALID_INSTANCE, "distance_km": 200.0, "n_stops": 25}
    r = client.post(PREDICT_URL, json={"instances": [invalid_distance_n_stop_case]})
    assert r.status_code == 422
    assert "infeasible" in r.text.lower()

def test_batch_size_too_small():
    r = client.post(PREDICT_URL, json={"instances": []})
    assert r.status_code == 422

def test_batch_size_max_limit():
    batch = [EXAMPLE_VALID_INSTANCE] * 500
    r = client.post(PREDICT_URL, json={"instances": batch})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 500

def test_request_id_echo():
    my_id = "test-id-12345"
    r = client.post(PREDICT_URL, json={"instances": [EXAMPLE_VALID_INSTANCE]}, headers={"X-Request-ID": my_id})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == my_id
    assert r.json()["request_id"] == my_id

def test_request_id_generated():
    r = client.post(PREDICT_URL, json={"instances": [EXAMPLE_VALID_INSTANCE]})
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert len(r.headers["X-Request-ID"]) >= 30

def test_request_id_in_error():
    invalid_distance_case = {**EXAMPLE_VALID_INSTANCE, "distance_km": -5.0}
    my_id = "err-req-id-999"
    r = client.post(PREDICT_URL, json={"instances": [invalid_distance_case]}, headers={"X-Request-ID": my_id})
    assert r.status_code == 422
    body = r.json()
    assert body.get("request_id") == my_id
    assert r.headers.get("X-Request-ID") == my_id

def test_metrics_endpoint():
    client.post(PREDICT_URL, json={"instances": [EXAMPLE_VALID_INSTANCE]})
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "eta_requests_total" in text
    assert "eta_request_latency_seconds_bucket" in text
    assert "eta_request_latency_seconds_sum" in text
    assert "eta_request_latency_seconds_count" in text
    assert '+Inf' in text

def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body

def test_model_raise_returns_500(monkeypatch):
    def boom(X):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(ea.model, "predict", boom)
    r = client.post(PREDICT_URL, json={"instances": [EXAMPLE_VALID_INSTANCE]})
    assert r.status_code == 500
    body = r.json()
    assert "simulated model failure" not in r.text
    assert "request_id" in body
