from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from typing import Annotated
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a bank customer will churn using Random Forest model (83.2% accuracy)",
    version="2.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# LOAD SAVED MODEL FILES
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Models")


try:
    model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    preprocessing_info = joblib.load(os.path.join(MODEL_DIR, "preprocessing_info.pkl"))
    
    print("✅ Model loaded successfully!")
    print(f"   Model Type: Random Forest")
    print(f"   Features: {len(feature_names)}")
    print(f"   Expected features: {feature_names}")
    
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}")
    print(f"   Make sure the following files exist in {MODEL_DIR}:")
    print(f"   - random_forest_model.pkl")
    print(f"   - feature_names.pkl")
    print(f"   - preprocessing_info.pkl")
    raise

# ---------------------------
# INPUT SCHEMA
# ---------------------------
class CustomerData(BaseModel):
    """
    Customer input data matching the trained Random Forest model.
    All fields are required. Use correct values for one-hot encoded features.
    """
    
    # Numeric features
    CreditScore: Annotated[int, Field(..., ge=300, le=900, description="Customer credit score (300-900)")]
    Age: Annotated[int, Field(..., ge=18, le=100, description="Age of the customer")]
    Tenure: Annotated[int, Field(..., ge=0, le=10, description="Years with the bank (0-10)")]
    Balance: Annotated[float, Field(..., ge=0.0, description="Account balance")]
    NumOfProducts: Annotated[int, Field(..., ge=1, le=4, description="Number of bank products (1-4)")]
    HasCrCard: Annotated[int, Field(..., ge=0, le=1, description="Has credit card: 1=Yes, 0=No")]
    IsActiveMember: Annotated[int, Field(..., ge=0, le=1, description="Active member: 1=Yes, 0=No")]
    EstimatedSalary: Annotated[float, Field(..., ge=0.0, description="Estimated annual salary")]
    Satisfaction_Score: Annotated[int, Field(..., ge=1, le=5, description="Customer satisfaction score (1-5)")]
    Point_Earned: Annotated[int, Field(..., ge=0, description="Reward points earned")]
    
    # Geography (one-hot encoded: France is baseline, both=0)
    Geography_Germany: Annotated[int, Field(..., ge=0, le=1, description="From Germany: 1=Yes, 0=No")]
    Geography_Spain: Annotated[int, Field(..., ge=0, le=1, description="From Spain: 1=Yes, 0=No")]
    
    # Gender (one-hot encoded: Female is baseline)
    Gender_Male: Annotated[int, Field(..., ge=0, le=1, description="Gender: 1=Male, 0=Female")]
    
    # Card Type (one-hot encoded: DIAMOND is baseline, all=0)
    Card_Type_GOLD: Annotated[int, Field(..., ge=0, le=1, description="Gold card: 1=Yes, 0=No")]
    Card_Type_PLATINUM: Annotated[int, Field(..., ge=0, le=1, description="Platinum card: 1=Yes, 0=No")]
    Card_Type_SILVER: Annotated[int, Field(..., ge=0, le=1, description="Silver card: 1=Yes, 0=No")]

    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 650,
                "Age": 35,
                "Tenure": 5,
                "Balance": 100000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0,
                "Satisfaction_Score": 3,
                "Point_Earned": 500,
                "Geography_Germany": 0,
                "Geography_Spain": 0,
                "Gender_Male": 1,
                "Card_Type_GOLD": 1,
                "Card_Type_PLATINUM": 0,
                "Card_Type_SILVER": 0
            }
        }


# ---------------------------
# HOME ROUTE
# ---------------------------
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API (Random Forest - 83.2% accuracy)",
        "status": "running",
        "model": "RandomForestClassifier",
        "features": len(feature_names),
        "docs": "Visit /docs for interactive API documentation"
    }


@app.get("/health")
def health_check():
    """Health check endpoint for Docker/monitoring"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": feature_names is not None
    }


# ---------------------------
# PREDICT ROUTE
# ---------------------------
@app.post("/predict")
def predict(data: CustomerData):
    """
    Predict customer churn based on input features.
    
    Returns:
        - churn_prediction: 0 (will stay) or 1 (will churn)
        - result: Human-readable prediction
        - churn_probability: Probability of churning (0-1)
        - stay_probability: Probability of staying (0-1)
        - confidence: High/Medium/Low based on probability
    """
    
    try:
        # Step 1: Convert Pydantic model to dict
        input_dict = data.dict()
        
        # Step 2: Rename fields to match exact training column names
        # (Pydantic doesn't allow spaces in field names)
        input_dict["Satisfaction Score"] = input_dict.pop("Satisfaction_Score")
        input_dict["Point Earned"] = input_dict.pop("Point_Earned")
        input_dict["Card Type_GOLD"] = input_dict.pop("Card_Type_GOLD")
        input_dict["Card Type_PLATINUM"] = input_dict.pop("Card_Type_PLATINUM")
        input_dict["Card Type_SILVER"] = input_dict.pop("Card_Type_SILVER")
        
        # Step 3: Create DataFrame and ensure column order matches training
        df = pd.DataFrame([input_dict])
        
        # Add missing columns with 0 (for one-hot encoded features not provided)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training exactly
        df = df[feature_names]
        
        # Step 4: Make prediction (no scaling needed for Random Forest)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        churn_prob = float(probability[1])
        stay_prob = float(probability[0])
        
        # Determine confidence level
        if churn_prob > 0.7 or churn_prob < 0.3:
            confidence = "High"
        elif churn_prob > 0.6 or churn_prob < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "churn_prediction": int(prediction),
            "result": "Will Churn" if prediction == 1 else "Will Stay",
            "churn_probability": round(churn_prob, 4),
            "stay_probability": round(stay_prob, 4),
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ---------------------------
# MODEL INFO ROUTE
# ---------------------------
@app.get("/model-info")
def model_info():
    """Get information about the trained model"""
    return {
        "model_type": "RandomForestClassifier",
        "test_accuracy": 0.832,
        "f1_score": 0.632,
        "precision": 0.571,
        "recall": 0.706,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_date": "2026-02-15",
        "description": "Random Forest model trained on bank customer churn data with 83.2% accuracy"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
