# Module 4 assignment — starter (SwiftParcel ETA service)

## General Setup Instructions (From the original README.md)
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python eta_data.py            # writes data/eta_train.csv, data/eta_fresh.csv, data/eta_v1.joblib, data/eta_v2.joblib
    python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

Load generator:

    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 1 2 4 8 16 --duration 4 --out loadtest_results.json

See the assignment document for tasks, interface contract and rubric.

## Part A
Run the following commands
- In the first terminal run "ETA_MODEL=data/eta_v1.joblib uvicorn PartA.eta_app:app --host 0.0.0.0 --port 8000 --workers 1"
- The rest of the commands should run in the same terminal
- Run "cd PartA" and then "ETA_MODEL=../data/eta_v1.joblib pytest test_eta_app.py -v"
- Run "cd .." and then "python load_test.py --url https://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --out PartA/loadtest_results.json"
- To distinguish the differences between different batch sizes, run "python load_test.py --url https://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 1" and "python load_test.py --url https://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 100"

## Part B
- Register both the models by running "MLFLOW_TRACKING_URI=sqlite:///mlruns.db ETA_DATA_DIR=../data python3 eta_mlflow.py register" and "MLFLOW_TRACKING_URI=sqlite:///mlruns.db ETA_DATA_DIR=../data python3 eta_mlflow.py gate 2>&1 | tee gate_output.txt"
- Then in the first terminal run "MLFLOW_TRACKING_URI=sqlite:///mlruns.db mlflow models serve -m "models:/eta-model@prod" --port 5101 --no-conda"
- In the second terminal run "MLFLOW_TRACKING_URI=sqlite:///mlruns.db mlflow models serve -m "models:/eta-model@candidate" --port 5102 --no-conda"
- In the third terminal run "MLFLOW_TRACKING_URI=sqlite:///mlruns.db ETA_DATA_DIR=../data python3 eta_mlflow.py shadow http://localhost:5101/invocations http://localhost:5102/invocations 2>&1 | tee shadow_output.txt"
- To run the test cases, run "pytest test_gate.py"

## Part C
- Run "ETA_DATA_DIR=../data python3 eta_ray.py" to check its run.
- To run the tests, run "ETA_DATA_DIR=../data python3 verify_ray.py 2>&1 | tee verify_output.txt" for the tests.

## Part D
- In the first terminal run "ETA_MODEL=data/eta_v1.joblib uvicorn PartA.eta_app:app --host 0.0.0.0 --port 8000 --workers 1"
- In the second terminal run "python3 PartD/smoke_eta.py --url http://localhost:8000/v1/models/eta:predict --expect-version v1 --max-p95-ms 80"

## AI Usage Declaration
- I used AI for the format/syntax of some of the test cases and to give me a few cases for tests for variety.
- Used AI to fix some syntax and to point out things I had missed while coding.