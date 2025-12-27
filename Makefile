.PHONY: help install dev test lint format clean docker-build docker-up docker-down train api dashboard

help:
	@echo "Predictive Maintenance - Available Commands"
	@echo "============================================"
	@echo "install      - Install production dependencies"
	@echo "dev          - Install development dependencies"
	@echo "test         - Run tests with coverage"
	@echo "lint         - Run linting checks"
	@echo "format       - Format code with black and isort"
	@echo "clean        - Clean build artifacts"
	@echo "docker-build - Build Docker images"
	@echo "docker-up    - Start Docker services"
	@echo "docker-down  - Stop Docker services"
	@echo "train        - Run training pipeline"
	@echo "api          - Start API server"
	@echo "dashboard    - Start Streamlit dashboard"

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 src/ api/ tests/
	mypy src/

format:
	black .
	isort .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

train:
	python -m src.pipelines.training_pipeline

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000
