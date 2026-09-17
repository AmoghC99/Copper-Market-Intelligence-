<<<<<<< HEAD
# Copper Market Intelligence & Forecasting Platform

Purpose

This repository is the Stage 1 scaffold for the "Copper Market Intelligence & Forecasting Platform" — a modular, reproducible Python project that will ingest public copper market data, perform feature engineering, build forecasting models, and expose results through a dashboard and a grounded AI analyst.

Current status (Stage 1)

- Repository scaffolded with `src/` package layout and minimal utilities.
- Configuration module and logging helper implemented.
- Minimal `pytest` test confirming package imports.
- GitHub Actions workflow added to run lint and tests.

Planned stages

1. Scaffold (this stage)
2. Data ingestion adapters and ETL pipelines
3. Feature engineering and EDA
4. Baseline and statistical forecasting (naive, SARIMAX)
5. Machine-learning forecasting (XGBoost) and explainability
6. Scenario analysis and evaluation
7. Grounded AI analyst (RAG) and vector store
8. Streamlit dashboard and deployment

Getting started (developer)

- Create a virtual environment using Python 3.11+.

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
# or
.\.venv\Scripts\activate     # cmd
```

- Install development dependencies (recommended):

```
python -m pip install -U pip
python -m pip install -e .[dev]
```

- Run tests:

```
$env:PYTHONPATH = "src"  # PowerShell, if not installed editable
python -m pytest -q
```

Notes

- This stage intentionally avoids data ingestion, modelling, large frameworks, or external services.
- Keep secrets out of the repository; use environment variables and `.env` files (see `.env.example`).
=======
# Copper Market Intelligence & Forecasting Platform

Purpose

This repository is the Stage 1 scaffold for the "Copper Market Intelligence & Forecasting Platform" — a modular, reproducible Python project that will ingest public copper market data, perform feature engineering, build forecasting models, and expose results through a dashboard and a grounded AI analyst.

Current status (Stage 1)

- Repository scaffolded with `src/` package layout and minimal utilities.
- Configuration module and logging helper implemented.
- Minimal `pytest` test confirming package imports.
- GitHub Actions workflow added to run lint and tests.

Planned stages

1. Scaffold (this stage)
2. Data ingestion adapters and ETL pipelines
3. Feature engineering and EDA
4. Baseline and statistical forecasting (naive, SARIMAX)
5. Machine-learning forecasting (XGBoost) and explainability
6. Scenario analysis and evaluation
7. Grounded AI analyst (RAG) and vector store
8. Streamlit dashboard and deployment

Getting started (developer)

- Create a virtual environment using Python 3.11+.

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
# or
.\.venv\Scripts\activate     # cmd
```

- Install development dependencies (recommended):

```
python -m pip install -U pip
python -m pip install -e .[dev]
```

- Run tests:

```
$env:PYTHONPATH = "src"  # PowerShell, if not installed editable
python -m pytest -q
```

Notes

- This stage intentionally avoids data ingestion, modelling, large frameworks, or external services.
- Keep secrets out of the repository; use environment variables and `.env` files (see `.env.example`).
>>>>>>> 08f19bad2376f7959fa366be1049c98c47bebb89
