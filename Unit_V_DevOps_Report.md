# Capstone Project 5 Report: DevOps for AI Deployment
**Topic:** Implementation of Containerized AI Services for Crime Analysis  
**Frameworks utilized:** PySpark (AI Engine), Flask (Serving API), Docker (Containerization), GitHub Actions (CI/CD), Kubernetes (Orchestration)

---

## 1. Introduction
This project fulfills the requirements for **Unit V: DevOps for AI**. The objective is to demonstrate the principles of "AI-as-a-Service" (AIaaS) by taking a distributed machine learning model (trained in Unit IV using PySpark MLlib) and deploying it into a production-grade infrastructure using DevOps methodologies.

## 2. Methodology & Implementation

### 2.1 Containerization (Docker)
We utilized **Docker** to containerize the application. The unified `Dockerfile` packages:
- **Java Runtime:** OpenJDK (required by Apache Spark)
- **Python 3.9:** Runtime for Flask API and PySpark
- **Dependencies:** Automated installation via `requirements.txt`
- **Application Code:** Flask API, PySpark training scripts, and Dashboard
- **Crime Data:** Real NCRB dataset copied from the parent `crime---analysis/` directory

### 2.2 Multi-Service Orchestration (Docker Compose)
The `docker-compose.yml` orchestrates two services:
1. **crime-ai:** The unified AI service that trains the PySpark model on startup, then serves the Flask API and Dashboard on port 5000.
2. **shared-volume:** A persistent Docker volume (`shared_model`) that synchronizes AI artifacts between microservices.

### 2.3 Automated Startup Pipeline (entrypoint.sh)
The container executes a 3-phase startup chain:
1. **Data Preparation:** Loads NCRB crime data, filters top 5 crime types, creates state-level feature matrix.
2. **Model Training:** Uses PySpark MLlib (VectorAssembler → StandardScaler → KMeans k=3) to classify states into High/Medium/Low risk. Exports trained parameters as scikit-learn `.pkl` files for lightweight serving.
3. **API Serving:** Starts the Flask web server with the dashboard and prediction endpoints.

### 2.4 AI Model Serving (Flask REST API)
The trained model is exposed via four HTTP endpoints:
- `GET /` — Interactive glassmorphic dashboard
- `GET /health` — Model status and feature information
- `GET /api/states` — All state risk classifications
- `POST /api/query` — Execute custom PySpark SQL queries on the 2001-2013 dataset

### 2.5 CI/CD Pipeline (GitHub Actions)
The `.github/workflows/ci.yml` automates the full build-test-deploy cycle:
- Install dependencies → Run data pipeline → Train model → Build Docker containers → Verify API health → Cleanup

### 2.6 Infrastructure-as-Code (Kubernetes)
The `pipeline/deployment.yaml` defines a production Kubernetes deployment:
- 3 replicas for horizontal scaling
- CPU/Memory resource limits
- Liveness and readiness probes on `/health`
- LoadBalancer service type for external access

### 2.7 Automated Testing (Pytest)
Unit tests in `tests/test_pipeline.py` validate:
- Data preparation output (correct columns, row count)
- Model training artifacts (model.pkl, scaler.pkl, risk_mapping)

## 3. Results
The implementation successfully demonstrates:
- **Environment Consistency:** Docker ensures the AI system runs identically on any machine.
- **Scalability:** Kubernetes enables horizontal scaling to multiple replicas.
- **Accessibility:** The Crime Analysis AI is a live web service with a premium dashboard.
- **Automation:** CI/CD pipeline automates the entire build → train → deploy → verify cycle.
- **Transparency:** Custom SQL query engine for direct analysis of historical datasets.

## 4. Conclusion
By applying DevOps principles to the Spark Crime Analysis project, we have transitioned from "Experimental Machine Learning" (Notebooks) to "Production AI Deployment" (APIs and Containers). This fulfills the Unit V requirement of developing a project focused on the lifecycle management and deployment of open-source AI frameworks.
