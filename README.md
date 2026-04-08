# Unit V — DevOps for AI: Crime Analytics Hub

This folder is the **complete, self-contained DevOps project** for the Crime Analysis capstone. It takes the PySpark K-Means model from Unit IV and deploys it as a production-grade AI service using Docker, CI/CD, and Kubernetes.

## 🚀 Architecture

```
docker-compose up --build
        │
        ▼
┌─────────────────────────────────────────────┐
│  crime-ai (Container)                       │
│  ┌──────────────────────────────────────┐   │
│  │ STEP 1: data_preparation.py          │   │
│  │   → Loads NCRB data, pivots features │   │
│  ├──────────────────────────────────────┤   │
│  │ STEP 2: model_training.py            │   │
│  │   → PySpark MLlib K-Means (k=3)      │   │
│  │   → Exports sklearn .pkl artifacts   │   │
│  ├──────────────────────────────────────┤   │
│  │ STEP 3: app.py (Flask)               │   │
│  │   → REST API: /predict, /health      │   │
│  │   → Dashboard: index.html            │   │
│  └──────────────────────────────────────┘   │
│                    │                        │
│                    ▼                        │
│  ┌──────────────────────────────────────┐   │
│  │ Artifacts (Shared Vol)                │   │
│  │   → model.pkl, scaler.pkl, CSV       │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## 📁 Folder Structure

```text
U5/
├── src/
│   ├── data/
│   │   └── data_preparation.py      # NCRB data → feature matrix
│   ├── model/
│   │   └── model_training.py        # PySpark MLlib → sklearn export
│   └── app/
│       ├── app.py                   # Flask API + Feature Rendering
│       └── static/
│           └── index.html           # Premium dashboard UI
├── tests/
│   └── test_pipeline.py             # Pytest unit tests
├── pipeline/
│   └── deployment.yaml              # Kubernetes manifests (3 replicas)
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions CI/CD pipeline
├── Dockerfile                       # Unified container (Java + PySpark + Flask)
├── docker-compose.yml               # Multi-service orchestration
├── entrypoint.sh                    # Startup chain: train → serve
├── requirements.txt                 # Python dependencies
├── Unit_V_DevOps_Report.md          # Academic report
└── README.md                        # You are here
```

## 🛠️ How to Run

### Docker Compose (Recommended)
```powershell
cd "d:\Project\HNS PROJECT\U5"
docker-compose up --build
```

Then open:
- **Dashboard:** http://localhost:5000
- **Health Check:** http://localhost:5000/health
- **API Test:**
```powershell
curl.exe -X POST http://localhost:5000/predict `
         -H "Content-Type: application/json" `
         -d '{"features": [10000, 5000, 2000, 1000, 500]}'
```

### Manual (Without Docker)
```powershell
cd "d:\Project\HNS PROJECT\U5"
pip install -r requirements.txt
python src/data/data_preparation.py
python src/model/model_training.py
python src/app/app.py
```

### Run Tests
```powershell
cd "d:\Project\HNS PROJECT\U5"
pytest tests/test_pipeline.py -v
```

## ✅ DevOps Components

| Component | File |
|---|---|
| Containerization | `Dockerfile` |
| Orchestration | `docker-compose.yml` |
| CI/CD | `.github/workflows/ci.yml` |
| Kubernetes | `pipeline/deployment.yaml` |
| REST API | `src/app/app.py` |
| Dashboard | `src/app/static/index.html` |
| Tests | `tests/test_pipeline.py` |
| Persistence Layer | Shared Volume (`shared_model`) |

---
Developed for the HNS PROJECT — Unit V DevOps Exploration.
# testing ci again