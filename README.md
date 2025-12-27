# 🔧 Predictive Maintenance MLOps Project

[![CI/CD Pipeline](https://github.com/yourusername/predictive-maintenance/actions/workflows/main.yml/badge.svg)](https://github.com/Ayushhgit/predictive-maintenance/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An **end-to-end MLOps project** for predicting Remaining Useful Life (RUL) of industrial equipment using the NASA C-MAPSS Turbofan Engine Degradation Dataset. This project demonstrates production-grade ML engineering practices including CI/CD, experiment tracking, model serving, containerization, and monitoring.


## 🎯 Project Overview

Predictive maintenance uses machine learning to predict when equipment will fail, enabling proactive maintenance scheduling. This project:

- **Predicts RUL** (Remaining Useful Life) of turbofan engines
- **Trains multiple models** (Random Forest, Gradient Boosting, LSTM, etc.)
- **Tracks experiments** with MLflow
- **Serves predictions** via REST API
- **Monitors performance** through Streamlit dashboard
- **Automates CI/CD** with GitHub Actions

### Business Value
- ⬇️ **Reduce unplanned downtime** by 30-50%
- 💰 **Lower maintenance costs** through optimized scheduling
- 📈 **Extend equipment lifespan** with timely interventions

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREDICTIVE MAINTENANCE SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Data       │───▶│   Data       │───▶│    Data      │───▶│  Model    │ │
│  │  Ingestion   │    │  Validation  │    │Transformation│    │ Training  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────┬─────┘ │
│         │                   │                   │                   │       │
│         │                   │                   │                   ▼       │
│         │                   │                   │            ┌───────────┐  │
│         │                   │                   │            │   Model   │  │
│         │                   │                   │            │Evaluation │  │
│         │                   │                   │            └─────┬─────┘  │
│         │                   │                   │                  │        │
│  ┌──────▼───────────────────▼───────────────────▼──────────────────▼─────┐  │
│  │                         MLflow Tracking Server                        │  │
│  │              (Experiments, Parameters, Metrics, Artifacts)            │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                          │
│  ┌───────────────────────────────▼───────────────────────────────────────┐  │
│  │                          Model Registry                               │  │
│  │                    (Versioning, Staging, Production)                  │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                          │
│         ┌────────────────────────┼────────────────────────┐                 │
│         ▼                        ▼                        ▼                 │
│  ┌─────────────┐         ┌─────────────┐          ┌─────────────┐          │
│  │  FastAPI    │         │  Streamlit  │          │   Batch     │          │
│  │  REST API   │         │  Dashboard  │          │ Prediction  │          │
│  └──────┬──────┘         └──────┬──────┘          └──────┬──────┘          │
│         │                       │                        │                  │
│  ┌──────▼───────────────────────▼────────────────────────▼───────────────┐  │
│  │                         Prometheus + Grafana                          │  │
│  │                      (Monitoring & Alerting)                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              INFRASTRUCTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│   │   Docker    │    │   GitHub    │    │    DVC      │    │   MongoDB   │ │
│   │  Compose    │    │   Actions   │    │   (Data)    │    │  (Storage)  │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/DL** | scikit-learn, TensorFlow/Keras, LSTM |
| **MLOps** | MLflow, DVC, Docker, GitHub Actions |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Data** | Pandas, NumPy, MongoDB |
| **Visualization** | Streamlit, Plotly, Matplotlib |
| **Testing** | pytest, pytest-cov, hypothesis |
| **Code Quality** | Black, isort, flake8, mypy, pre-commit |
| **Monitoring** | Prometheus, Grafana |

## ✨ Features

### ML Pipeline
- ✅ **Automated data ingestion** from multiple sources
- ✅ **Data validation** with quality checks and anomaly detection
- ✅ **Feature engineering** (lag features, rolling statistics)
- ✅ **Multiple model training** (RF, GB, Linear, Ridge, Lasso, SVR, LSTM)
- ✅ **Hyperparameter tuning** with GridSearchCV
- ✅ **Model evaluation** with comprehensive metrics (RMSE, MAE, R², MAPE)

### MLOps
- ✅ **Experiment tracking** with MLflow
- ✅ **Model registry** for versioning and staging
- ✅ **Data versioning** with DVC
- ✅ **CI/CD pipeline** with GitHub Actions
- ✅ **Containerization** with Docker & Docker Compose
- ✅ **Pre-commit hooks** for code quality

### Production
- ✅ **REST API** with FastAPI for real-time predictions
- ✅ **Batch prediction** pipeline for large datasets
- ✅ **Monitoring dashboard** with Streamlit
- ✅ **Health checks** and API documentation (Swagger/OpenAPI)
- ✅ **Risk level classification** (Critical, High, Medium, Low)


## 📁 Project Structure

```
predictive-maintenance/
├── .github/
│   └── workflows/
│       ├── main.yml              # CI/CD pipeline
│       └── model-training.yml    # Scheduled training
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   └── schemas.py                # Pydantic models
├── config/
│   ├── config.yaml               # Main configuration
│   └── schema.yaml               # Data schema
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── data/
│   ├── raw/                      # Raw data
│   ├── validated/                # Validated data
│   ├── transformed/              # Processed features
│   └── predictions/              # Batch predictions
├── monitoring/
│   ├── prometheus.yml            # Prometheus config
│   └── grafana/                  # Grafana dashboards
├── notebooks/
│   └── eda.ipynb                 # Exploratory analysis
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── batch_prediction.py
│   ├── pipelines/
│   │   └── training_pipeline.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── model_utils.py
│   ├── constants/
│   │   └── __init__.py
│   └── mlflow_tracking.py
├── tests/
│   ├── unit/
│   │   ├── test_data_validation.py
│   │   ├── test_model_evaluation.py
│   │   └── test_api.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── conftest.py               # Pytest fixtures
├── artifacts/
│   ├── models/                   # Trained models
│   ├── logs/                     # Application logs
│   └── reports/                  # Evaluation reports
├── .dvc/                         # DVC configuration
├── .pre-commit-config.yaml       # Pre-commit hooks
├── docker-compose.yml            # Docker services
├── Dockerfile                    # Multi-stage Dockerfile
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── pyproject.toml                # Build configuration
├── pytest.ini                    # Pytest configuration
└── README.md                     # This file
```

## 🔄 ML Pipeline

### Training Pipeline Flow

```
1. Data Ingestion    → Load raw sensor data from source
2. Data Validation   → Validate schema, types, and ranges
3. Transformation    → Feature engineering & scaling
4. Model Training    → Train multiple models
5. Model Evaluation  → Compare and select best model
6. Model Registry    → Version and stage models
```

### Models Implemented

| Model | Type | Use Case |
|-------|------|----------|
| Random Forest | Ensemble | Baseline, robust |
| Gradient Boosting | Ensemble | High accuracy |
| Linear Regression | Linear | Interpretable |
| Ridge/Lasso | Linear | Regularized |
| SVR | Kernel | Non-linear |
| LSTM | Deep Learning | Sequence modeling |


## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/predict` | Single/batch prediction |
| POST | `/predict/batch` | File-based batch prediction |
| POST | `/models/reload` | Reload models |

## 📊 Monitoring Dashboard

The Streamlit dashboard provides:

- **Overview**: Key metrics, model comparison
- **Model Performance**: Detailed metrics, visualizations
- **Predictions**: Interactive prediction interface
- **Data Explorer**: Feature distributions, correlations
- **System Health**: API status, resource usage

## 📈 Results

### Model Performance (Test Set)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 18.5 | 12.3 | 0.87 |
| Gradient Boosting | 17.2 | 11.8 | 0.89 |
| LSTM | 15.8 | 10.5 | 0.91 |
