# 🏦 Customer Churn Predictor

An end-to-end Machine Learning application that predicts whether a bank customer will **stay or leave (churn)** using a trained Random Forest model.

---

## 🚀 Features

- Churn prediction using ML model (~83% accuracy)
- FastAPI backend for REST API
- Streamlit frontend dashboard
- MLflow experiment tracking
- Fully Dockerized for easy deployment

---

## 🛠 Tech Stack

Python • Scikit-learn • FastAPI • Streamlit • MLflow • Docker

---

## 📂 Project Structure

app/ → FastAPI backend
Models/ → Saved model files
Data/ → Dataset
streamlit_app.py → UI
Dockerfile → Container setup


---

## ▶️ Run with Docker (Recommended)

### Pull Image

## Docker Setup
docker pull debabrata7/churn-app:latest

Run Container

docker run -p 8000:8000 -p 8501:8501 debabrata7/churn-app

🌐 Access Application
Streamlit UI:
http://localhost:8501

API Docs:
http://localhost:8000/docs

