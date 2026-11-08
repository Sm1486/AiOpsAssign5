"""
verify_ray.py -- Verification script for eta_ray.py (Part C2).

Deploys the sticky-canary app on a 2-CPU Ray cluster and runs five
assertion-based checks:

(a) Share to candidate within ±4 pp of weight=0.2  (1000 distinct IDs)
(b) 200 customers each get the same arm on a second call
(c) Same stickiness holds after re-deploying the identical app
(d) After raising weight 0.2 → 0.5, no candidate customer moves back to prod
(e) X-Beta: 1 header override works for 50 requests

Usage (from partC/ directory):
    python verify_ray.py
"""

import sys
import os
import time
import statistics
import requests

sys.path.insert(0, os.path.dirname(__file__))
import ray
from ray import serve
from eta_ray import build

BASE_URL = "http://localhost:8000/v1/models/eta:predict"
DATA_DIR = os.environ.get("ETA_DATA_DIR", "../data")

SAMPLE_INSTANCE = {
    "distance_km": 25.0,
    "n_stops": 5,
    "hour": 10,
    "weekday": 1,
    "traffic_index": 0.3,
    "rain": 0,
    "weight_kg": 10.0,
}


def send(customer_id: str, beta: bool = False) -> dict:
    headers = {}
    if beta:
        headers["X-Beta"] = "1"
    r = requests.post(
        BASE_URL,
        json={"customer_id": customer_id, "instances": [SAMPLE_INSTANCE]},
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def wait_for_server(max_wait: float = 60.0):
    """Poll until the server responds or timeout."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.post(
                BASE_URL,
                json={"customer_id": "ping", "instances": [SAMPLE_INSTANCE]},
                timeout=5,
            )
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("Server did not start in time")


def main():
    print("=" * 60)
    print("ETA Ray Serve Verification")
    print("=" * 60)

    # Init Ray with exactly 2 CPUs
    ray.init(num_cpus=2, ignore_reinit_error=True)

    # -----------------------------------------------------------------------
    # Deploy with weight=0.2
    # -----------------------------------------------------------------------
    print("\n[Setup] Deploying with weight=0.2 ...")
    app = build(weight=0.2)
    serve.run(app, host="0.0.0.0", port=8000)
    wait_for_server()
    print("[Setup] Server ready.")

    N = 1000
    customer_ids = [f"customer_{i:04d}" for i in range(N)]

    # -----------------------------------------------------------------------
    # (a) Share routed to candidate within ±4 pp of 0.2
    # -----------------------------------------------------------------------
    print(f"\n[Check a] Routing {N} distinct customers once ...")
    first_call_arms: dict[str, str] = {}
    for cid in customer_ids:
        resp = send(cid)
        first_call_arms[cid] = resp["arm"]

    cand_count = sum(1 for arm in first_call_arms.values() if arm == "v2")
    cand_share = cand_count / N
    print(f"  Candidate share: {cand_share:.4f}  (expected ≈ 0.2 ± 0.04)")
    assert abs(cand_share - 0.2) <= 0.04, (
        f"FAIL (a): candidate share {cand_share:.4f} not within ±0.04 of 0.2"
    )
    print("  PASS (a)")

    # -----------------------------------------------------------------------
    # (b) Stickiness: 200 customers get same arm on second call
    # -----------------------------------------------------------------------
    print("\n[Check b] Testing stickiness for 200 customers (second call) ...")
    sample_200 = customer_ids[:200]
    sticky_ok = 0
    for cid in sample_200:
        resp = send(cid)
        if resp["arm"] == first_call_arms[cid]:
            sticky_ok += 1
    print(f"  {sticky_ok}/200 got the same arm")
    assert sticky_ok == 200, f"FAIL (b): only {sticky_ok}/200 sticky"
    print("  PASS (b)")

    # -----------------------------------------------------------------------
    # (c) Stickiness survives re-deploy of identical app
    # -----------------------------------------------------------------------
    print("\n[Check c] Re-deploying identical app (weight=0.2) ...")
    app2 = build(weight=0.2)
    serve.run(app2, host="0.0.0.0", port=8000)
    wait_for_server()

    sticky_after_redeploy = 0
    for cid in sample_200:
        resp = send(cid)
        if resp["arm"] == first_call_arms[cid]:
            sticky_after_redeploy += 1
    print(f"  {sticky_after_redeploy}/200 same arm after re-deploy")
    assert sticky_after_redeploy == 200, (
        f"FAIL (c): only {sticky_after_redeploy}/200 sticky after re-deploy"
    )
    print("  PASS (c)")

    # -----------------------------------------------------------------------
    # (d) After weight 0.2 → 0.5, no candidate customer moves back to prod
    # -----------------------------------------------------------------------
    print("\n[Check d] Raising weight to 0.5; no candidate should move to prod ...")
    cand_customers = {cid for cid, arm in first_call_arms.items() if arm == "v2"}

    app3 = build(weight=0.5)
    serve.run(app3, host="0.0.0.0", port=8000)
    wait_for_server()

    regressions = 0
    for cid in cand_customers:
        resp = send(cid)
        if resp["arm"] != "v2":
            regressions += 1

    print(f"  Candidate customers who moved to prod: {regressions}")
    assert regressions == 0, f"FAIL (d): {regressions} customers moved back to prod"
    print("  PASS (d)")

    # -----------------------------------------------------------------------
    # (e) X-Beta: 1 override always routes to candidate
    # -----------------------------------------------------------------------
    print("\n[Check e] Testing X-Beta:1 override for 50 requests ...")
    # Use prod customers to verify the override overrides their normal routing
    prod_customers = [cid for cid, arm in first_call_arms.items() if arm == "v1"][:50]
    override_ok = 0
    for cid in prod_customers:
        resp = send(cid, beta=True)
        if resp["arm"] == "v2":
            override_ok += 1
    print(f"  {override_ok}/50 correctly routed to candidate via header")
    assert override_ok == 50, f"FAIL (e): only {override_ok}/50 overridden correctly"
    print("  PASS (e)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)

    serve.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()

