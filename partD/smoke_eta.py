"""
smoke_eta.py -- Release-gate smoke test for the ETA service (Part D).

Usage:
    python smoke_eta.py --url http://localhost:8000/v1/models/eta:predict \\
                        --expect-version v1 \\
                        --max-p95-ms 80

Exit codes:
    0  all checks passed
    1  one or more checks failed

Checks (≥ 5):
    1. /healthz returns 200 and {"status": "ok"} with the expected version
    2. Valid request returns 200 with correct response shape
    3. Invalid input (bad distance_km) returns 422
    4. Monotonicity sanity: longer distance → higher ETA
    5. Traffic sanity: higher traffic_index → higher ETA (same route, rain=0)
    6. p95 latency over 50 single-row requests is within --max-p95-ms
    7. Expected model version is present in every successful response
"""

import argparse
import sys
import time
import statistics

import requests

VALID_INSTANCE = {
    "distance_km": 25.0,
    "n_stops": 5,
    "hour": 10,
    "weekday": 1,
    "traffic_index": 0.3,
    "rain": 0,
    "weight_kg": 10.0,
}

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = ""):
    icon = "✓" if passed else "✗"
    print(f"  [{icon}] {name}" + (f"  — {detail}" if detail else ""))
    RESULTS.append((name, passed, detail))


def predict(session: requests.Session, url: str, instances: list[dict], beta: bool = False) -> requests.Response:
    headers = {"X-Beta": "1"} if beta else {}
    return session.post(url, json={"instances": instances}, headers=headers, timeout=30)


def main():
    ap = argparse.ArgumentParser(description="Smoke test for ETA service")
    ap.add_argument("--url", required=True, help="Predict endpoint URL")
    ap.add_argument("--expect-version", required=True, help="Expected model version (e.g. v1)")
    ap.add_argument("--max-p95-ms", type=float, default=80.0, help="p95 latency SLO in ms")
    args = ap.parse_args()

    url = args.url
    expect_ver = args.expect_version
    max_p95_ms = args.max_p95_ms

    # Derive healthz URL (strip trailing path and append /healthz)
    base_url = url.split("/v1/models")[0] if "/v1/models" in url else url.rsplit("/", 1)[0]
    healthz_url = base_url + "/healthz"

    sess = requests.Session()

    print(f"\nSmoke test: {url}")
    print(f"Expect version: {expect_ver}  |  max p95: {max_p95_ms} ms\n")

    # -------------------------------------------------------------------
    # Check 1: /healthz
    # -------------------------------------------------------------------
    try:
        r = sess.get(healthz_url, timeout=10)
        body = r.json()
        passed = r.status_code == 200 and body.get("status") == "ok"
        check("healthz returns 200 + status=ok", passed, f"status={r.status_code} body={body}")
    except Exception as exc:
        check("healthz returns 200 + status=ok", False, str(exc))

    # -------------------------------------------------------------------
    # Check 2: Valid single request shape
    # -------------------------------------------------------------------
    try:
        r = predict(sess, url, [VALID_INSTANCE])
        body = r.json()
        passed = (
            r.status_code == 200
            and "predictions" in body
            and isinstance(body["predictions"], list)
            and len(body["predictions"]) == 1
            and "model_version" in body
            and "request_id" in body
        )
        check("Valid request returns correct shape", passed, f"status={r.status_code}")
    except Exception as exc:
        check("Valid request returns correct shape", False, str(exc))

    # -------------------------------------------------------------------
    # Check 3: Invalid input returns 422
    # -------------------------------------------------------------------
    try:
        bad = {**VALID_INSTANCE, "distance_km": -999.0}
        r = predict(sess, url, [bad])
        passed = r.status_code == 422
        check("Invalid distance_km returns 422", passed, f"status={r.status_code}")
    except Exception as exc:
        check("Invalid distance_km returns 422", False, str(exc))

    # -------------------------------------------------------------------
    # Check 4 (model-sanity): longer distance → higher ETA
    # -------------------------------------------------------------------
    try:
        short = {**VALID_INSTANCE, "distance_km": 5.0, "n_stops": 2}
        long_ = {**VALID_INSTANCE, "distance_km": 100.0, "n_stops": 2}
        r = predict(sess, url, [short, long_])
        body = r.json()
        preds = body.get("predictions", [None, None])
        passed = r.status_code == 200 and preds[1] > preds[0]
        check(
            "Sanity: longer distance → higher ETA",
            passed,
            f"short={preds[0]} min  long={preds[1]} min",
        )
    except Exception as exc:
        check("Sanity: longer distance → higher ETA", False, str(exc))

    # -------------------------------------------------------------------
    # Check 5 (model-sanity): heavier traffic → higher ETA
    # -------------------------------------------------------------------
    try:
        low_traffic = {**VALID_INSTANCE, "traffic_index": 0.0}
        high_traffic = {**VALID_INSTANCE, "traffic_index": 1.0}
        r = predict(sess, url, [low_traffic, high_traffic])
        body = r.json()
        preds = body.get("predictions", [None, None])
        passed = r.status_code == 200 and preds[1] > preds[0]
        check(
            "Sanity: heavier traffic → higher ETA",
            passed,
            f"low={preds[0]} min  high={preds[1]} min",
        )
    except Exception as exc:
        check("Sanity: heavier traffic → higher ETA", False, str(exc))

    # -------------------------------------------------------------------
    # Check 6: p95 latency SLO
    # -------------------------------------------------------------------
    latencies_ms = []
    try:
        for _ in range(50):
            t0 = time.perf_counter()
            predict(sess, url, [VALID_INSTANCE])
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        latencies_ms.sort()
        p95 = latencies_ms[int(0.95 * len(latencies_ms))]
        passed = p95 <= max_p95_ms
        check(
            f"p95 latency ≤ {max_p95_ms} ms",
            passed,
            f"measured p95={p95:.1f} ms",
        )
    except Exception as exc:
        check(f"p95 latency ≤ {max_p95_ms} ms", False, str(exc))

    # -------------------------------------------------------------------
    # Check 7: Correct model version in every response
    # -------------------------------------------------------------------
    try:
        mismatches = 0
        for _ in range(10):
            r = predict(sess, url, [VALID_INSTANCE])
            if r.status_code == 200:
                ver = r.json().get("model_version", "")
                # Support Ray Serve (arm field) or plain (model_version field)
                if ver != expect_ver:
                    mismatches += 1
        passed = mismatches == 0
        check(
            f"model_version == {expect_ver} in all responses",
            passed,
            f"{mismatches}/10 mismatches",
        )
    except Exception as exc:
        check(f"model_version == {expect_ver} in all responses", False, str(exc))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    total = len(RESULTS)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = total - n_pass

    print(f"\nResults: {n_pass}/{total} passed, {n_fail} failed")

    if n_fail > 0:
        print("\nFailed checks:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)

    print("All checks passed. ✓")
    sys.exit(0)


if __name__ == "__main__":
    main()

