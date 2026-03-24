# 🔮 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![MLflow](https://img.shields.io/badge/MLflow-3.10.1-red)
![SHAP](https://img.shields.io/badge/SHAP-0.45.0-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

> An end-to-end Machine Learning system that predicts which telecom customers are likely to cancel their subscription — with full exploratory data analysis, feature engineering, model training, experiment tracking, explainability, API deployment, and an interactive web interface.

---

## 🎬 Demo Video

[![Customer Churn Prediction Demo](https://img.shields.io/badge/▶%20Watch%20Demo-Google%20Drive-red?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1D8vEtF3k5XUZUHImbbczpb0sHexxjvrW/view?usp=sharing)
---

## 📌 Problem Statement

Customer churn is one of the most critical challenges in the telecom industry. Acquiring a new customer costs 5x more than retaining an existing one. This project builds an intelligent ML system that analyzes 7,043 customer records, identifies patterns that lead to churn, predicts who will leave next, explains why they will leave, and serves predictions via a live API and Web UI — helping the company take action before the customer leaves.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Dataset Size | 7,043 customers |
| Features Used | 30 |
| Best Model | XGBoost |
| Accuracy | 75.73% |
| Recall (Churners) | 71.58% |
| Churn Rate in Data | 26.5% |

---

## 🔍 Business Insights Discovered

| # | Insight | Impact |
|---|---------|--------|
| 1 | New customers with low tenure churn significantly more | 🔴 High |
| 2 | Customers paying above $65/month are at higher churn risk | 🔴 High |
| 3 | Month-to-month contract customers churn the most | 🔴 High |
| 4 | Fiber optic internet customers churn more than DSL | 🔴 High |
| 5 | Senior citizens churn at almost double the average rate 41.6% | 🟠 Medium |
| 6 | Electronic check payment users churn more than auto-pay users | 🟠 Medium |
| 7 | Phone service has almost no impact on churn | 🟢 Low |

---
## 🏗️ Project Architecture
````
customer-churn/
│
├── churn_eda.ipynb           # Stage 1+2: EDA & Data Cleaning
├── churn_model.ipynb         # Stage 3+4+5: Training, MLflow, SHAP
│
├── app/
│   ├── main.py               # FastAPI application
│   ├── model.pkl             # Saved XGBoost model
│   └── requirements.txt      # API dependencies
│
├── frontend/
│   └── index.html            # Interactive Web UI
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── cleaned_churn.csv
│
├── .gitignore
└── README.md
````

## 🛠️ Tech Stack

| Category | Tool | Purpose |
|----------|------|---------|
| Language | Python 3.12 | Core development |
| Data Analysis | Pandas, NumPy | Data manipulation |
| Visualization | Matplotlib, Seaborn | EDA charts |
| ML Models | Scikit-learn, XGBoost | Model training |
| Experiment Tracking | MLflow | Track all experiments |
| Explainability | SHAP | Explain predictions |
| Deployment | FastAPI + Uvicorn | REST API |
| Frontend | HTML, CSS, JavaScript | Web interface |
| Version Control | Git + GitHub | Code management |

---

## 📊 ML Pipeline

Raw Data 7043 rows 21 columns
          ↓
Stage 1 — Exploratory Data Analysis
          ↓
Stage 2 — Data Cleaning and Feature Engineering
          ↓
Stage 3 — Model Training XGBoost and RandomForest
          ↓
Stage 4 — MLflow Experiment Tracking
          ↓
Stage 5 — SHAP Explainability
          ↓
Stage 6 — FastAPI Deployment
          ↓
Live Prediction API and Web UI ✅

---

## 📈 Model Performance

| Model | Accuracy | Recall Churners |
|-------|----------|-----------------|
| Random Forest | 79.00% | 53.62% |
| XGBoost default | 79.84% | 53.62% |
| XGBoost tuned | 75.73% | 71.58% ✅ |

> Why tuned XGBoost? Missing a churner costs the business more than a false alarm. We optimized for Recall over raw accuracy using scale_pos_weight=4 to handle class imbalance.

---

## 🔍 SHAP Explainability

Every prediction comes with a full explanation showing which features caused the churn decision and by how much. The API returns top reasons for churn and top reasons against churn for every single customer prediction.

---

## 🚀 How To Run Locally

### 1. Clone Repository
git clone https://github.com/abhichiku18/customer-churn.git
cd customer-churn

### 2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

### 3. Install Dependencies
pip install -r app/requirements.txt

### 4. Run FastAPI
cd app
uvicorn main:app --reload

### 5. Open Web UI
Open frontend/index.html in your browser

### 6. View MLflow Dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /predict | Predict churn |
| GET | /docs | API documentation |

---

## 📉 Class Imbalance Handling

Dataset had 73.5% non-churners and 26.5% churners. Model was biased towards majority class. Fixed using scale_pos_weight=4 which improved Recall from 53% to 71%.

---

## 💼 What This Project Demonstrates

| Skill | How |
|-------|-----|
| Data Analysis | EDA with 6 charts and 7 business insights |
| Feature Engineering | Encoding, scaling, handling nulls |
| ML Modeling | Trained and compared multiple models |
| Production Mindset | MLflow tracking like real companies |
| Explainability | SHAP explains every single prediction |
| Deployment | FastAPI working live REST API |
| Frontend | Interactive web UI for non-technical users |
| Communication | Clean README and GitHub portfolio |

---

## 📦 Dataset

| Detail | Info |
|--------|------|
| Name | Telco Customer Churn |
| Source | Kaggle |
| Rows | 7,043 customers |
| Columns | 21 features |
| Target | Churn Yes or No |

---

## 🧠 Resume Line

Built an end-to-end ML pipeline for customer churn prediction on 7,000+ records using XGBoost, with MLflow experiment tracking, SHAP explainability, and FastAPI deployment — achieving 75%+ accuracy and 71%+ recall on churners.

---

## 👤 Author

Your Name Here
GitHub: https://github.com/abhichiku18
LinkedIn: https://www.linkedin.com/in/abhichiku/
Email: abhichiku2004@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project helpful please give it a star ⭐
