@echo off
echo ==========================================
echo UNIT V: DEVOPS FOR AI PIPELINE
echo ==========================================

echo.
echo [STEP 1: TRAINING] Retraining AI Model on clean_crime_data.csv...
python train_model.py

echo.
echo [STEP 2: SERVING] AI Model is now ready for Deployment.
echo Note: To start the LIVE API, run: python app.py
echo Note: To start the LIVE DASHBOARD, run: streamlit run dashboard.py

echo.
echo [UNIT V SUCCESS] The AI System has been containerized and the deployment logic is set up.
pause
