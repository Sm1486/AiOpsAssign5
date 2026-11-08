"""
eta_ray.py -- Ray Serve sticky canary for the ETA service (Part C).

Interface contract:
    build(weight=0.1, extra_work_ms=0.0) -> ServeApplication

The returned application routes POST /v1/models/eta:predict.
Request body: {"customer_id": "...", "instances": [{7 features}, ...]}
Response:     {predictions, model_version, arm, request_id}

Sticky canary logic:
    - mmh3 (MurmurHash3) used for a stable, process-independent hash.
    - customer_id whose hash % 1000 < weight*1000 → candidate (v2).
    - X-Beta: 1 header forces candidate regardless of hash.

Fractional CPU allocation keeps the whole app within 2 CPUs:
    prod     2 replicas × 0.45 CPU = 0.90 CPU
    candidate autoscale 1-3 × 0.30 CPU (max 0.90 CPU)
    router   1 replica  × 0.20 CPU = 0.20 CPU
    total max                        2.00 CPU

Usage:
    python eta_ray.py          # deploy and serve on http://localhost:8000
"""

import os
import time
import uuid
import asyncio
from typing import Optional

import numpy as np
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    import mmh3  # stable, process-independent hash
    def _stable_hash(s: str) -> int:
        return mmh3.hash(s, signed=False)
except ImportError:
    # Fallback: use SHA-256 (also stable across processes/restarts)
    import hashlib
    def _stable_hash(s: str) -> int:
        return int(hashlib.sha256(s.encode()).hexdigest(), 16)

DATA_DIR = os.environ.get("ETA_DATA_DIR", "../data")

# ---------------------------------------------------------------------------
# EtaModel deployment
# ---------------------------------------------------------------------------
@serve.deployment(
    name="EtaModel",
    # CPU will be overridden per-deployment in build()
    ray_actor_options={"num_cpus": 0.45},
)
class EtaModel:
    def __init__(self, bundle_path: str, extra_work_ms: float = 0.0):
        import joblib
        bundle = joblib.load(bundle_path)
        self._model = bundle["model"]
        self._features = bundle["feature_columns"]
        self._version = bundle["version"]
        self._extra_work_ms = extra_work_ms

    async def predict(self, instances: list[dict]) -> dict:
        import pandas as pd
        X = pd.DataFrame(instances)[self._features]
        if self._extra_work_ms > 0:
            await asyncio.sleep(self._extra_work_ms / 1000.0)
        preds = self._model.predict(X)
        return {
            "predictions": [round(float(p), 1) for p in preds],
            "model_version": self._version,
        }


# ---------------------------------------------------------------------------
# Router / ingress
# ---------------------------------------------------------------------------
ingress_app = FastAPI(title="ETA Canary Router")


@serve.deployment(
    name="ETARouter",
    ray_actor_options={"num_cpus": 0.20},
)
@serve.ingress(ingress_app)
class ETARouter:
    def __init__(
        self,
        prod_handle: DeploymentHandle,
        cand_handle: DeploymentHandle,
        weight: float = 0.1,
    ):
        self._prod = prod_handle
        self._cand = cand_handle
        self._weight = weight
        # bucket boundary: 0 .. 999, candidate if hash % 1000 < threshold
        self._threshold = int(weight * 1000)

    def _route(self, customer_id: str, force_beta: bool) -> tuple[DeploymentHandle, str]:
        if force_beta:
            return self._cand, "v2"
        bucket = _stable_hash(customer_id) % 1000
        if bucket < self._threshold:
            return self._cand, "v2"
        return self._prod, "v1"

    @ingress_app.post("/v1/models/eta:predict")
    async def predict(self, payload: Request):
        request_id = payload.headers.get("X-Request-ID", str(uuid.uuid4()))
        force_beta = payload.headers.get("X-Beta", "0") == "1"

        try:
            body = await payload.json()
        except Exception:
            resp = JSONResponse(
                status_code=400,
                content={"error": "invalid JSON", "request_id": request_id},
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

        customer_id = body.get("customer_id", "anonymous")
        instances = body.get("instances", [])

        handle, arm = self._route(customer_id, force_beta)
        result = await handle.predict.remote(instances)

        resp = JSONResponse(
            content={
                **result,
                "arm": arm,
                "request_id": request_id,
            }
        )
        resp.headers["X-Request-ID"] = request_id
        return resp


# ---------------------------------------------------------------------------
# build() -- public API required by the interface contract
# ---------------------------------------------------------------------------
def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """
    Return a bound Ray Serve application.

    prod:      2 replicas, 0.45 CPU each
    candidate: autoscale 1-3, 0.30 CPU each
    router:    1 replica, 0.20 CPU
    """
    prod_deployment = EtaModel.options(
        name="EtaProd",
        num_replicas=2,
        ray_actor_options={"num_cpus": 0.45},
    ).bind(
        bundle_path=f"{DATA_DIR}/eta_v1.joblib",
        extra_work_ms=extra_work_ms,
    )

    cand_deployment = EtaModel.options(
        name="EtaCand",
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 5,
            "upscale_delay_s": 10,
            "downscale_delay_s": 30,
        },
        ray_actor_options={"num_cpus": 0.30},
    ).bind(
        bundle_path=f"{DATA_DIR}/eta_v2.joblib",
        extra_work_ms=extra_work_ms,
    )

    router = ETARouter.options(
        name="ETARouter",
        num_replicas=1,
        ray_actor_options={"num_cpus": 0.20},
    ).bind(
        prod_handle=prod_deployment,
        cand_handle=cand_deployment,
        weight=weight,
    )

    return router


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ray.init(num_cpus=2, ignore_reinit_error=True)
    app = build(weight=0.1)
    serve.run(app, host="0.0.0.0", port=8000)
    print("ETA canary service running on http://localhost:8000")
    input("Press Enter to stop...\n")
    serve.shutdown()
    ray.shutdown()

