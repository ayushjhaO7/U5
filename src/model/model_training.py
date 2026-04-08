from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.preprocessing import StandardScaler as SKLearnScaler
import joblib
import pandas as pd
import numpy as np
import json
import os

# ── Settings ──
DATA_FILE = "cleaned_crime_features.csv"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
META_FILE = "model_meta.joblib"
DISTRICT_RISK_FILE = "district_risks.json"

def train_kmeans_model(data_path, model_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data_preparation.py first.")
        return

    # ═══════════════════════════════════════
    # PHASE 1: Apache Spark Distributed Training
    # ═══════════════════════════════════════
    print(">>> Initializing Spark for District Analysis...")
    spark = SparkSession.builder \
        .appName("CrimeAI_District_Training") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    pdf = pd.read_csv(data_path)
    df = spark.createDataFrame(pdf)
    features = [c for c in df.columns if c not in ['STATE/UT', 'DISTRICT']]

    print(f">>> Features: {features}")
    print(f">>> Dataset scale: {df.count()} districts for clustering.")

    # VectorAssembler
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    assembled_df = assembler.transform(df)

    # StandardScaler
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    # K-Means (k=4 for better thresholding)
    k = 4
    print(f">>> Cluster Training (k={k})...")
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="cluster", k=k, seed=42)
    kmeans_model = kmeans.fit(scaled_df)
    result_df = kmeans_model.transform(scaled_df)

    # ═══════════════════════════════════════
    # PHASE 2: Risk Level Logic (Decreased Threshold)
    # ═══════════════════════════════════════
    spark_centers = kmeans_model.clusterCenters()
    magnitudes = {i: float(np.sum(c)) for i, c in enumerate(spark_centers)}
    sorted_clusters = sorted(magnitudes.items(), key=lambda x: x[1], reverse=True)

    # Lowering Threshold: Top 2 clusters map to "High Risk"
    # Mapping logic:
    # 0 -> High Risk (Highest centroid)
    # 1 -> High Risk (Second highest) -> Decreased threshold logic
    # 2 -> Medium Risk
    # 3 -> Low Risk
    risk_mapping = {
        sorted_clusters[0][0]: "High Risk",
        sorted_clusters[1][0]: "High Risk",
        sorted_clusters[2][0]: "Medium Risk",
        sorted_clusters[3][0]: "Low Risk"
    }

    print("\n>>> District Risk Mapping complete (Decreased Threshold Applied).")
    result_pdf = result_df.select("STATE/UT", "DISTRICT", "cluster").toPandas()
    result_pdf['Risk_Level'] = result_pdf['cluster'].map(risk_mapping)

    # ═══════════════════════════════════════
    # PHASE 3: Export Artifacts
    # ═══════════════════════════════════════
    print("\n>>> Exporting AI models and district profiles...")

    sk_scaler = SKLearnScaler()
    sk_scaler.mean_ = scaler_model.mean.toArray()
    sk_scaler.scale_ = scaler_model.std.toArray()
    sk_scaler.var_ = sk_scaler.scale_ ** 2
    sk_scaler.n_features_in_ = len(features)
    sk_scaler.n_samples_seen_ = np.int64(df.count())
    sk_scaler.feature_names_in_ = np.array(features)

    # Train sklearn proxy with k=4
    sk_kmeans = SKLearnKMeans(n_clusters=k, random_state=42, n_init=10)
    sk_kmeans.fit(sk_scaler.transform(pdf[features].values))
    sk_kmeans.cluster_centers_ = np.array(spark_centers)

    # Hierarchical risk profile for the API
    district_risks = {}
    for _, row in result_pdf.iterrows():
        key = f"{row['STATE/UT']}|{row['DISTRICT']}"
        district_risks[key] = {
            "risk_level": row['Risk_Level'],
            "cluster_id": int(row['cluster'])
        }

    with open(DISTRICT_RISK_FILE, 'w') as f:
        json.dump(district_risks, f, indent=4)

    joblib.dump(sk_kmeans, model_path)
    joblib.dump(sk_scaler, SCALER_FILE)
    meta = {
        "features": features,
        "risk_mapping": risk_mapping,
        "engine": "District-Level PySpark Clustering (k=4)"
    }
    joblib.dump(meta, META_FILE)

    print(f">>> Model re-trained for {len(district_risks)} districts.")
    spark.stop()

if __name__ == "__main__":
    train_kmeans_model(DATA_FILE, MODEL_FILE)
