"""
eta_app.py -- FastAPI service for SwiftParcel ETA model (Part A).

Usage:
    ETA_MODEL=data/eta_v1.joblib uvicorn partA.eta_app:app --port 8000
    # or from within partA/:
    ETA_MODEL=../data/eta_v1.joblib uvicorn eta_app:app --port 8000
"""

import os
import time
import uuid
import logging
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("eta_app")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Model — loaded once at module level
# ---------------------------------------------------------------------------
_model_path = os.environ.get("ETA_MODEL", "data/eta_v1.joblib")
_bundle = joblib.load(_model_path)
model = _bundle["model"]
_feature_columns = _bundle["feature_columns"]
_model_version = _bundle["version"]

# ---------------------------------------------------------------------------
# Prometheus metrics (pure Python, no prometheus_client required at import)
# ---------------------------------------------------------------------------
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

eta_requests_total = Counter(
    "eta_requests_total",
    "Total number of requests",
    ["path", "status"],
)
eta_request_latency_seconds = Histogram(
    "eta_request_latency_seconds",
    "Request latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
VALID_RAIN = {0, 1}


class ETAInstance(BaseModel):
    distance_km: float
    n_stops: int
    hour: int
    weekday: int
    traffic_index: float
    rain: int
    weight_kg: float

    @field_validator("distance_km")
    @classmethod
    def check_distance(cls, v):
        if not (0.1 <= v <= 500):
            raise ValueError("distance_km must be between 0.1 and 500")
        return v

    @field_validator("n_stops")
    @classmethod
    def check_n_stops(cls, v):
        if not (1 <= v <= 60):
            raise ValueError("n_stops must be between 1 and 60")
        return v

    @field_validator("hour")
    @classmethod
    def check_hour(cls, v):
        if not (0 <= v <= 23):
            raise ValueError("hour must be between 0 and 23")
        return v

    @field_validator("weekday")
    @classmethod
    def check_weekday(cls, v):
        if not (0 <= v <= 6):
            raise ValueError("weekday must be between 0 and 6")
        return v

    @field_validator("traffic_index")
    @classmethod
    def check_traffic(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("traffic_index must be between 0 and 1")
        return v

    @field_validator("rain")
    @classmethod
    def check_rain(cls, v):
        if v not in VALID_RAIN:
            raise ValueError("rain must be 0 or 1")
        return v

    @field_validator("weight_kg")
    @classmethod
    def check_weight(cls, v):
        if not (0 < v <= 1000):
            raise ValueError("weight_kg must be > 0 and <= 1000")
        return v


class PredictRequest(BaseModel):
    instances: list[ETAInstance]

    @model_validator(mode="after")
    def check_batch_size(self):
        if not (1 <= len(self.instances) <= 500):
            raise ValueError("batch size must be between 1 and 500")
        return self


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="SwiftParcel ETA Service")


def _get_request_id(request: Request) -> str:
    """Return client's X-Request-ID or generate a fresh UUID."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


def _add_request_id_header(response: Response, request_id: str) -> None:
    response.headers["X-Request-ID"] = request_id


# ---------------------------------------------------------------------------
# /v1/models/eta:predict
# ---------------------------------------------------------------------------
@app.post("/v1/models/eta:predict")
async def predict(payload: Request):
    request_id = _get_request_id(payload)
    t_start = time.perf_counter()

    # --- parse body manually so we can return structured 422 errors ---
    try:
        body = await payload.json()
    except Exception:
        eta_requests_total.labels(path="/v1/models/eta:predict", status="422").inc()
        resp = JSONResponse(
            status_code=422,
            content={
                "error": "validation_failed",
                "details": [{"field": "body", "msg": "invalid JSON"}],
                "request_id": request_id,
            },
        )
        resp.headers["X-Request-ID"] = request_id
        return resp

    # --- validate with Pydantic ---
    try:
        req = PredictRequest.model_validate(body)
    except Exception as exc:
        details = []
        try:
            import json
            errs = exc.errors() if hasattr(exc, "errors") else []
            for e in errs:
                loc = ".".join(str(x) for x in e.get("loc", []))
                details.append({"field": loc, "msg": e.get("msg", str(e))})
        except Exception:
            details = [{"field": "unknown", "msg": str(exc)}]

        eta_requests_total.labels(path="/v1/models/eta:predict", status="422").inc()
        resp = JSONResponse(
            status_code=422,
            content={
                "error": "validation_failed",
                "details": details,
                "request_id": request_id,
            },
        )
        resp.headers["X-Request-ID"] = request_id
        return resp

    # --- cross-field: infeasible ---
    infeasible = []
    for i, inst in enumerate(req.instances):
        if inst.distance_km > 150 and inst.n_stops > 20:
            infeasible.append(i)
    if infeasible:
        eta_requests_total.labels(path="/v1/models/eta:predict", status="422").inc()
        resp = JSONResponse(
            status_code=422,
            content={
                "error": "validation_failed",
                "details": [
                    {
                        "field": f"instances[{i}]",
                        "msg": f"infeasible delivery: distance_km > 150 and n_stops > 20",
                    }
                    for i in infeasible
                ],
                "request_id": request_id,
            },
        )
        resp.headers["X-Request-ID"] = request_id
        return resp

    # --- inference ---
    try:
        import pandas as pd

        rows = [inst.model_dump() for inst in req.instances]
        X = pd.DataFrame(rows)[_feature_columns]
        preds = model.predict(X)
        predictions = [round(float(p), 1) for p in preds]
    except Exception as exc:
        log.error("Inference error: %s", exc)
        eta_requests_total.labels(path="/v1/models/eta:predict", status="500").inc()
        resp = JSONResponse(
            status_code=500,
            content={"error": "internal_error", "request_id": request_id},
        )
        resp.headers["X-Request-ID"] = request_id
        return resp

    elapsed = time.perf_counter() - t_start
    eta_request_latency_seconds.observe(elapsed)
    eta_requests_total.labels(path="/v1/models/eta:predict", status="200").inc()

    resp = JSONResponse(
        content={
            "predictions": predictions,
            "model_version": _model_version,
            "request_id": request_id,
        }
    )
    resp.headers["X-Request-ID"] = request_id
    return resp


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------
@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    try:
        import pandas as pd

        sample = pd.DataFrame(
            [
                {
                    "distance_km": 10.0,
                    "n_stops": 2,
                    "hour": 12,
                    "weekday": 1,
                    "traffic_index": 0.5,
                    "rain": 0,
                    "weight_kg": 5.0,
                }
            ]
        )[_feature_columns]
        model.predict(sample)
        return JSONResponse({"status": "ok", "model_version": _model_version})
    except Exception as exc:
        log.error("Health-check inference failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "internal_error"})

