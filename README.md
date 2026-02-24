# 🏦 Customer Churn Predictor

Predict whether a bank customer will **stay or leave (churn)** using a trained Random Forest model.

---

## 🚀 Features
- Random Forest model (~83% accuracy) with saved artifacts in `Models/`
- FastAPI backend (`app/main.py`) for predictions and model info
- Streamlit dashboard (`streamlit_app.py`) for interactive scoring
- MLflow tracking notebooks in `Notebooks/`
- Single-container Docker/Compose for easy run

---

## 🛠 Tech Stack
Python • scikit-learn • FastAPI • Streamlit • MLflow • Docker

---

## 📂 Project Structure
- `app/` — FastAPI backend
- `Models/` — Trained model + feature metadata
- `Data/` — Dataset (`Customer-Churn-Records.csv`)
- `streamlit_app.py` — Streamlit UI
- `Dockerfile` / `docker-compose.yml` — Container + orchestration

---

## ▶️ Quick Start (Docker Compose — recommended)
```bash
docker compose up --build
```
Then open:
- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

Compose uses a single service `app` that runs both FastAPI and Streamlit.

### Using a prebuilt image
```bash
docker pull debabrata7/churn-app:latest
docker run -p 8000:8000 -p 8501:8501 debabrata7/churn-app:latest
```

---

## 🧪 Local Development (no Docker)
```bash
python -m venv venv
venv\\Scripts\\activate          # Windows
pip install -r requirements.txt

uvicorn app.main:app --reload    # starts API on 8000
streamlit run streamlit_app.py   # starts UI on 8501
```
If API and UI run separately, point the UI to the API host:
```bash
$env:API_URL="http://127.0.0.1:8000/predict"
$env:HEALTH_URL="http://127.0.0.1:8000/health"
```

---

## 🔌 API Endpoints (FastAPI)
- `GET /` — service info
- `GET /health` — health + model loaded flags
- `POST /predict` — returns `churn_prediction`, probabilities, confidence
- `GET /model-info` — model metrics and feature list

---

## 🗒 Notes
- Data imbalance: `Exited` positive class is ~20% (2,038/10,000).
- Feature list is stored in `Models/feature_names.pkl`; payloads must match these columns (the Streamlit app handles this for you).

---

