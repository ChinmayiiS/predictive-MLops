# 🚀 Printer Predictive Maintenance (MLOps + AWS)

## 📌 Overview
End-to-end machine learning system to predict printer failures using sensor data.

## ⚙️ Tech Stack
- Python, Scikit-learn, XGBoost
- MLflow
- FastAPI
- Docker
- AWS (ECR + EC2)
- Streamlit Dashboard

## 🔥 Features
- Feature engineering
- Model training + MLflow tracking
- REST API deployment
- Real-time prediction
- AWS deployment
- Interactive dashboard

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python data/generate_data.py
python src/train.py
uvicorn api.app:app --reload
streamlit run dashboard/app.py
