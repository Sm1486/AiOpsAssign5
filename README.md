# Module 4 assignment — starter (SwiftParcel ETA service)

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python eta_data.py            # writes data/eta_train.csv, data/eta_fresh.csv, data/eta_v1.joblib, data/eta_v2.joblib
    python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

Load generator:

    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 1 2 4 8 16 --duration 4 --out loadtest_results.json

See the assignment document for tasks, interface contract and rubric.
