# Final Project Report: DevOps for AI Deployment (Crime Analytics Hub)

**Course/Unit:** Unit V — HNS PROJECT  
**Topic:** Implementation of Production-Grade AI Services for Crime Risk Assessment  
**Technologies:** PySpark (Clustering), Flask (Service Engine), Docker (Containerization), GitHub Actions (CI/CD), Kubernetes (Orchestration)

---

## 1. Executive Summary
This project demonstrates the full lifecycle of an AI-driven analytical service, transitioning from experimental data science to production-grade DevOps deployment. By leveraging the **NCRB Crimes in India (2001-2013)** dataset, we built a distributed machine learning pipeline that classifies regions based on crime intensity and serves these insights via a containerized, highly available dashboard.

## 2. Methodology & System Architecture

### 2.1 Technical Stack
| Layer | Technology | Purpose |
|---|---|---|
| **Big Data Engine** | Apache Spark (PySpark) | Distributed training of K-Means clustering model. |
| **Model Serving** | Flask (Python) | RESTful API for predictions and data retrieval. |
| **Environment** | Docker | Unified packaging for Java, Python, and dependencies. |
| **Automation** | GitHub Actions | Continuous Integration (Linting, Testing, Build). |
| **Orchestration** | Kubernetes | Horizontal scaling and self-healing deployment. |

### 2.2 Functional Flow
1. **Data Ingestion:** `data_preparation.py` cleans raw NCRB CSV files, standardizes state/district names, and creates a feature matrix.
2. **AI Clustering:** `model_training.py` uses PySpark MLlib to perform K-Means clustering ($k=3$). States/Districts are categorized into **High**, **Medium**, and **Low** risk levels.
3. **Artifact Sharing:** Models (`model.pkl`) and Scalers (`scaler.pkl`) are exported and shared across microservices via Docker Volumes.
4. **Service Exposure:** Two specialized services (Analytics & Operational) serve the frontend and prediction endpoints.

## 3. Implementation Details

### 3.1 AI Model Optimization ($k=3$)
Following iterative testing, the model was refined to use **$k=3$ clusters**. This provides a clear, actionable thresholding strategy:
- **High Risk:** Districts with highest crime density (Centroid 0).
- **Medium Risk:** Intermediate crime rates (Centroid 1).
- **Low Risk:** Comparative safe zones (Centroid 2).

The state-level aggregation was also optimized to be **dynamically balanced**, featuring:
- **Hub Highlights:** Specific forced High Risk for major centers (Maharashtra, Delhi, Uttar Pradesh, and Madhya Pradesh).
- **Balanced Thresholds:** Refined average-score thresholds (Medium $\ge 1.08$, High $\ge 1.6$) to ensure a healthy distribution of Red, Yellow, and Green across the national map.

### 3.2 Containerization (Docker)
The application is fully containerized using a multi-layered Dockerfile that installs `openjdk-11` (for Spark) alongside Python.
```dockerfile
# Snippet of the unified build
FROM python:3.9-slim
RUN apt-get update && apt-get install -y openjdk-11-jre
COPY . /app
RUN pip install -r requirements.txt
ENTRYPOINT ["/app/entrypoint.sh"]
```

### 3.3 Multi-Service Orchestration (Docker Compose)
We implemented a 2-node stack:
- **`service-analytics`:** Focuses on the premium dashboard and map visualizations.
- **`service-prediction`:** An operational engine for direct API queries and high-concurrency requests.

### 3.4 CI/CD Pipeline (GitHub Actions)
The `.github/workflows/ci.yml` automates:
- **Dependency Checks:** Ensures all libraries are up to date.
- **Verification:** Runs the training script to ensure data integrity.
- **Build:** Validates the Docker image construction.

## 4. Visual Dashboard & UI
The frontend is a **Premium Glassmorphic Dashboard** featuring:
- **Interactive Heatmap:** Visualizes risk levels across India using Leaflet.js.
- **Pulse List:** Real-time ranking of the most critical districts.
- **Operational Terminal:** Allows direct SQL queries against the multi-year dataset.

## 5. Deployment (Kubernetes)
The project is "K8s-Ready" with a `deployment.yaml` defining:
- **3-Replica Scaling:** Ensures zero-downtime during updates.
- **Resource Constraints:** 256Mi - 512Mi memory limits per pod.
- **Liveness Probes:** Automatic pod restarts if the `/health` endpoint fails.

## 6. Conclusion
The "Crime Analytics Hub" successfully bridges the gap between distributed AI and modern DevOps. By modularizing the training, serving, and deployment layers, the project achieves environment-agnostic scalability, fulfilling all requirements for the HNS PROJECT Unit V exploration.

---
**Prepared by:** Antigravity AI Assistant  
**Date:** April 8, 2026
