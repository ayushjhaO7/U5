import os
from fastapi import FastAPI, HTTPException
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import StandardScalerModel, VectorAssembler

# Initializing Spark Session
spark = SparkSession.builder \
    .appName("DevOps_AI_Service") \
    .master("local[*]") \
    .getOrCreate()

# Load the saved model and scaler
model_path = "crime_kmeans_model"
scaler_path = "crime_scaler_model"
top_crimes_path = "top_crimes.txt"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("ERROR: Model or Scaler not found! Run train_model.py first.")
    exit(1)

model = KMeansModel.load(model_path)
scaler_model = StandardScalerModel.load(scaler_path)

with open(top_crimes_path, "r") as f:
    top_5 = f.read().split(",")

app = FastAPI()

# Risk Mapping
# These will be calculated dynamically on startup for better accuracy
# (simplified for this deployment code)
@app.get("/")
def home():
    return {"status": "DevOps AI Service is Running", "framework": "PySpark + FastAPI"}

@app.get("/predict/{state}")
def predict_risk(state: str):
    # Preprocessing logic to pull the latest state data
    data_path = "../crime---analysis/data/cleaned_crime_data.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=500, detail="Data file for inference not found.")

    df = spark.read.csv(data_path, header=True, inferSchema=True)
    state_data = df.filter(col("STATE/UT") == state.upper()) \
                   .filter(col("crime_type").isin(top_5)) \
                   .groupBy("STATE/UT") \
                   .pivot("crime_type") \
                   .sum("crime_count") \
                   .na.fill(0)
    
    if state_data.count() == 0:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found in dataset.")

    # Apply same transformations
    assembler = VectorAssembler(inputCols=top_5, outputCol="features")
    assembled_df = assembler.transform(state_data)
    
    scaled_df = scaler_model.transform(assembled_df)
    
    # Predict
    result = model.transform(scaled_df)
    prediction = result.select("cluster").collect()[0][0]
    
    return {
        "state": state.upper(),
        "cluster": int(prediction),
        "api_status": "Success",
        "description": f"AI assigned Cluster {prediction} to {state}."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
