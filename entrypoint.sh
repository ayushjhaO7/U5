#!/bin/bash
set -e

echo "=================================================="
echo "  MODE: ${SERVICE_TYPE^^}"
echo "=================================================="

export SHARED_DIR="/app/shared"
mkdir -p "$SHARED_DIR"

if [ "$SERVICE_TYPE" == "analytics" ]; then
    echo "[LEAD] Initializing High-Resolution Analytics Engine..."
    
    # Smart Startup: Only run pipeline if artifacts are missing
    if [ ! -f "model.pkl" ] || [ ! -f "cleaned_crime_features.csv" ]; then
        echo "[UPDATE] Artifacts missing or new logic detected. Running pipeline..."
        python src/data/data_preparation.py
        python src/model/model_training.py
    else
        echo "✓ Artifacts detected. Skipping pipeline for instant startup."
    fi
    
    # Sync ALL data including the NEW 2001-2013 dataset for PySpark SQL Engine
    # Note: Using the filename specifically requested by the user
    cp cleaned_crime_features.csv model.pkl scaler.pkl model_meta.joblib district_risks.json "$SHARED_DIR/"
    
    # Copy the raw dataset if it exists in the fallback data/ dir
    if [ -f "data/Crimes_in_india_2001-2013.csv" ]; then
        cp "data/Crimes_in_india_2001-2013.csv" "$SHARED_DIR/"
    elif [ -f "../../Crimes_in_india_2001-2013.csv" ]; then
        cp "../../Crimes_in_india_2001-2013.csv" "$SHARED_DIR/"
    fi
    
    echo "✓ Global artifacts initialized and synced."
else
    echo "[FOLLOWER] Initializing Operational Engine (Spark SQL Matrix)..."
    until [ -f "$SHARED_DIR/model.pkl" ]; do
        echo "...waiting for lead to sync global artifacts..."
        sleep 5
    done
    echo "✓ Global artifacts detected."
fi

echo "[START] Launching ${SERVICE_TYPE^^} on port $PORT..."
python src/app/app.py
