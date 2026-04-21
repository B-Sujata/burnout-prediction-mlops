#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# entrypoint.sh — Burnout Prediction Container Entrypoint
#
# Usage (via docker run or docker-compose):
#   pipeline          → Run full ML pipeline (all 6 stages)
#   ingest            → Stage 1 only: ingest & validate data
#   preprocess        → Stage 2 only
#   engineer          → Stage 3+4 only
#   train             → Stage 5 only
#   evaluate          → Stage 6 only
#   visualize         → Stage 9 only
#   predict           → Interactive prediction CLI
#   mlflow            → Launch MLflow tracking UI on port 5000
#   test              → Run pytest test suite
#   bash              → Drop into shell (for debugging)
#   <anything else>   → Execute as a raw command
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "============================================================"
echo "  🧠 Student Burnout Prediction System — Docker Container"
echo "  Command: $1"
echo "============================================================"

case "$1" in

    pipeline)
        echo "▶ Running full ML pipeline..."
        python run_pipeline.py
        ;;

    ingest)
        echo "▶ Stage 1: Data Ingestion"
        python -c "from src.data.ingest_data import ingest_data; ingest_data()"
        ;;

    preprocess)
        echo "▶ Stage 2: Preprocessing"
        python -c "from src.data.preprocess_data import preprocess_data; preprocess_data()"
        ;;

    engineer)
        echo "▶ Stage 3+4: Feature Engineering + Burnout Score"
        python -c "from src.data.feature_engineering import engineer_features; engineer_features()"
        ;;

    train)
        echo "▶ Stage 5: Model Training"
        python -c "from src.models.train_model import train_all_models; train_all_models()"
        ;;

    evaluate)
        echo "▶ Stage 6: Model Evaluation"
        python -c "from src.models.evaluate_model import evaluate_model; evaluate_model()"
        ;;

    visualize)
        echo "▶ Stage 9: Visualization"
        python -c "from src.visualization.visualize_results import visualize_results; visualize_results()"
        ;;

    predict)
        echo "▶ Prediction CLI (raw numeric input)"
        python src/models/predict_model.py --interactive
        ;;

    assess|questionnaire)
        echo "▶ Student Self-Assessment (human-friendly questionnaire)"
        python src/advisor/questionnaire_runner.py
        ;;

    assess-advise)
        echo "▶ Student Self-Assessment with LLM Advisor"
        python src/advisor/questionnaire_runner.py --advise
        ;;

    predict-advise)
        echo "▶ Prediction CLI with LLM Advisor"
        python src/models/predict_model.py --interactive --advise
        ;;

    predict-json)
        # Pass JSON via PREDICT_INPUT env variable
        echo "▶ JSON Prediction"
        if [ -z "$PREDICT_INPUT" ]; then
            echo "ERROR: Set PREDICT_INPUT env variable with JSON string"
            exit 1
        fi
        ADVISE_FLAG=""
        if [ "$USE_ADVISOR" = "true" ]; then
            ADVISE_FLAG="--advise"
        fi
        python src/models/predict_model.py --json-input "$PREDICT_INPUT" $ADVISE_FLAG
        ;;

    mlflow)
        echo "▶ Starting MLflow UI on port 5000..."
        mlflow ui \
            --host 0.0.0.0 \
            --port 5000 \
            --backend-store-uri "${MLFLOW_TRACKING_URI:-mlruns}"
        ;;

    test)
        echo "▶ Running test suite..."
        python -m pytest tests/ -v --tb=short
        ;;

    bash|sh)
        exec /bin/bash
        ;;

    *)
        # Fallback: execute whatever was passed
        exec "$@"
        ;;
esac
