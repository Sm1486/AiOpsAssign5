"""
eta_mlflow.py -- MLflow registry, promotion gate and shadow comparison (Part B).

Interface contract:
    register()                           - log both models, register, set aliases
    decide(prod, cand, **thresholds)     - pure function, returns {criteria, promote}
    promotion_gate(**thresholds)         - loads aliases, evaluates, calls decide

Environment:
    MLFLOW_TRACKING_URI  (default: sqlite:///mlruns.db)

Usage:
    python eta_mlflow.py register
    python eta_mlflow.py gate
    python eta_mlflow.py shadow
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import joblib

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

log = logging.getLogger("eta_mlflow")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Tracking URI
# ---------------------------------------------------------------------------
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(TRACKING_URI)

MODEL_NAME = "eta-model"
DATA_DIR = os.environ.get("ETA_DATA_DIR", "../data")

FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]


# ---------------------------------------------------------------------------
# Custom pyfunc wrapper
# ---------------------------------------------------------------------------
class ETAPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a joblib bundle {model, feature_columns, version}."""

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["bundle_path"])
        self._model = bundle["model"]
        self._features = bundle["feature_columns"]
        self._version = bundle["version"]

    def predict(self, context, model_input, params=None):
        X = model_input[self._features]
        preds = self._model.predict(X)
        return pd.DataFrame({
            "eta_minutes": [round(float(p), 1) for p in preds],
            "model_version": self._version,
        })


# ---------------------------------------------------------------------------
# B1 – register()
# ---------------------------------------------------------------------------
def register():
    """Log both model bundles as pyfunc, register in Model Registry, set aliases."""
    client = MlflowClient()

    # Ensure model exists in registry
    try:
        client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(MODEL_NAME)

    train_df = pd.read_csv(f"{DATA_DIR}/eta_train.csv")
    fresh_df = pd.read_csv(f"{DATA_DIR}/eta_fresh.csv")

    input_example = fresh_df[FEATURES].head(3)

    registered_versions = {}

    for tag in ["v1", "v2"]:
        bundle_path = f"{DATA_DIR}/eta_{tag}.joblib"
        bundle = joblib.load(bundle_path)

        X_train = train_df[FEATURES]
        pyfunc_model = ETAPyfunc()
        # Build a dummy output for signature inference
        dummy_out = pd.DataFrame({
            "eta_minutes": [0.0],
            "model_version": [tag],
        })

        signature = infer_signature(input_example, dummy_out)

        with mlflow.start_run(run_name=f"eta-{tag}") as run:
            mlflow.log_param("model_tag", tag)
            mlflow.log_param("bundle_path", bundle_path)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_model,
                artifacts={"bundle_path": bundle_path},
                signature=signature,
                input_example=input_example,
                pip_requirements=[
                    "scikit-learn>=1.4",
                    "pandas>=2.2",
                    "joblib>=1.4",
                    "numpy",
                ],
                registered_model_name=MODEL_NAME,
            )

            run_id = run.info.run_id

        # Find the newly created version (highest version number)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        new_ver = max(versions, key=lambda v: int(v.version))
        registered_versions[tag] = new_ver.version
        log.info(f"Registered {tag} as version {new_ver.version}")

    # Set aliases: prod -> v1 version, candidate -> v2 version
    client.set_registered_model_alias(MODEL_NAME, "prod", registered_versions["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", registered_versions["v2"])

    # Print registry table
    print("\n=== Model Registry Table ===")
    print(f"{'Version':<10} {'Alias':<15} {'Run'}")
    print("-" * 50)
    for tag, ver in registered_versions.items():
        alias = "prod" if tag == "v1" else "candidate"
        print(f"{ver:<10} {alias:<15} eta-{tag}")

    return registered_versions


# ---------------------------------------------------------------------------
# B2 – decide() – pure function
# ---------------------------------------------------------------------------
def decide(
    prod: dict,
    cand: dict,
    min_rel_gain: float = 0.03,
    max_p95_ms: float = 10.0,
    max_slice_regression: float = 0.05,
) -> dict:
    """
    Pure promotion gate.

    prod / cand dicts must have keys:
        mae, mae_rush, mae_rain, p95_ms

    Returns:
        {"criteria": {<name>: bool, ...}, "promote": bool}
    """
    criteria = {}

    # 1. Overall MAE improvement (relative to prod)
    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    criteria["mae_improvement"] = bool(rel_gain >= min_rel_gain)

    # 2. Latency
    criteria["latency_ok"] = bool(cand["p95_ms"] <= max_p95_ms)

    # 3. No slice regression – rush hours
    rush_regression = cand["mae_rush"] - prod["mae_rush"]
    criteria["no_rush_regression"] = bool(rush_regression <= max_slice_regression)

    # 4. No slice regression – rain
    rain_regression = cand["mae_rain"] - prod["mae_rain"]
    criteria["no_rain_regression"] = bool(rain_regression <= max_slice_regression)

    promote = all(criteria.values())

    return {"criteria": criteria, "promote": promote}


# ---------------------------------------------------------------------------
# B2 – promotion_gate()
# ---------------------------------------------------------------------------
def promotion_gate(**thresholds) -> dict:
    """
    Load prod and candidate aliases, evaluate on fresh data, call decide().
    """
    from eta_data import make_data, FEATURES as ETA_FEATURES

    client = MlflowClient()

    def _load_alias(alias):
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
        uri = f"models:/{MODEL_NAME}@{alias}"
        return mlflow.pyfunc.load_model(uri), mv.version

    prod_model, prod_ver = _load_alias("prod")
    cand_model, cand_ver = _load_alias("candidate")

    fresh = make_data(2000, seed=99)
    X = fresh[ETA_FEATURES]
    y = fresh["eta_minutes"].values

    rush_mask = fresh["hour"].isin([7, 8, 9, 16, 17, 18, 19])
    rain_mask = fresh["rain"] == 1

    def _metrics(m):
        preds_df = m.predict(X)
        preds = preds_df["eta_minutes"].values
        mae = float(np.mean(np.abs(preds - y)))
        mae_rush = float(np.mean(np.abs(preds[rush_mask] - y[rush_mask]))) if rush_mask.any() else 0.0
        mae_rain = float(np.mean(np.abs(preds[rain_mask] - y[rain_mask]))) if rain_mask.any() else 0.0

        # measure p95 over 200 single-row predictions
        single_latencies = []
        for i in range(200):
            row = X.iloc[[i % len(X)]]
            t0 = time.perf_counter()
            m.predict(row)
            single_latencies.append((time.perf_counter() - t0) * 1000)
        p95_ms = float(np.percentile(single_latencies, 95))

        return {"mae": mae, "mae_rush": mae_rush, "mae_rain": mae_rain, "p95_ms": p95_ms}

    prod_metrics = _metrics(prod_model)
    cand_metrics = _metrics(cand_model)

    result = decide(prod_metrics, cand_metrics, **thresholds)
    result["prod_version"] = prod_ver
    result["cand_version"] = cand_ver
    result["prod_metrics"] = prod_metrics
    result["cand_metrics"] = cand_metrics
    return result


# ---------------------------------------------------------------------------
# B3 – shadow comparison helper
# ---------------------------------------------------------------------------
def shadow_compare(
    prod_url: str = "http://localhost:5101/invocations",
    cand_url: str = "http://localhost:5102/invocations",
    n: int = 200,
):
    """
    Send the same n rows to both servers and report:
        (i)   mean absolute difference in predictions
        (ii)  % of requests differing by > 10 minutes
        (iii) each server's p95 latency
    """
    import requests as req_lib

    fresh = pd.read_csv(f"{DATA_DIR}/eta_fresh.csv")
    rows = fresh.head(n)

    prod_preds, cand_preds = [], []
    prod_lats, cand_lats = [], []

    for _, row in rows.iterrows():
        payload = {"dataframe_records": [row[FEATURES].to_dict()]}

        t0 = time.perf_counter()
        rp = req_lib.post(prod_url, json=payload, timeout=30)
        prod_lats.append((time.perf_counter() - t0) * 1000)
        prod_preds.append(rp.json()["predictions"][0]["eta_minutes"])

        t0 = time.perf_counter()
        rc = req_lib.post(cand_url, json=payload, timeout=30)
        cand_lats.append((time.perf_counter() - t0) * 1000)
        cand_preds.append(rc.json()["predictions"][0]["eta_minutes"])

    prod_arr = np.array(prod_preds)
    cand_arr = np.array(cand_preds)
    diff = np.abs(prod_arr - cand_arr)

    result = {
        "mean_abs_diff_minutes": round(float(diff.mean()), 3),
        "pct_diff_gt_10min": round(float((diff > 10).mean() * 100), 2),
        "prod_p95_ms": round(float(np.percentile(prod_lats, 95)), 2),
        "cand_p95_ms": round(float(np.percentile(cand_lats, 95)), 2),
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    cmd = sys.argv[1] if len(sys.argv) > 1 else "register"

    if cmd == "register":
        register()

    elif cmd == "gate":
        # Default thresholds
        result = promotion_gate()
        print("\n=== Gate output (default thresholds) ===")
        print(json.dumps(result, indent=2))

        # Aggressive threshold (should fail)
        result2 = promotion_gate(min_rel_gain=0.60)
        print("\n=== Gate output (min_rel_gain=0.60) ===")
        print(json.dumps(result2, indent=2))

    elif cmd == "shadow":
        prod_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5101/invocations"
        cand_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:5102/invocations"
        result = shadow_compare(prod_url, cand_url)
        print("\n=== Shadow comparison ===")
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {cmd}. Use: register | gate | shadow")
        sys.exit(1)

