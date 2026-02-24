import os
import streamlit as st
import requests
import pandas as pd

# -----------------------------------------------
# PAGE SETUP
# -----------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------------------------
# STYLING
# -----------------------------------------------
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .step-header {
        background: #ffffff;
        border-left: 5px solid #4f8ef7;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 1.5rem 0 0.8rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a202c;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .hint {
        font-size: 0.78rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 8px;
    }

    div[data-testid="stButton"] > button {
        background: #4f8ef7;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        margin-top: 1rem;
    }
    div[data-testid="stButton"] > button:hover {
        background: #2f6de0;
    }

    .box-churn {
        background: #fff5f5;
        border: 2px solid #fc8181;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .box-stay {
        background: #f0fff4;
        border: 2px solid #68d391;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-emoji { font-size: 3rem; }
    .result-heading { font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0; }
    .result-msg { font-size: 0.95rem; color: #555; }
    .churn-color { color: #e53e3e; }
    .stay-color { color: #276749; }

    .prob-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .prob-num { font-size: 2rem; font-weight: 800; }
    .prob-lbl { font-size: 0.8rem; color: #666; margin-top: 4px; }
    .red { color: #e53e3e; }
    .green { color: #276749; }

    .status-ok { color: #276749; font-size: 0.85rem; font-weight: 600; }
    .status-err { color: #e53e3e; font-size: 0.85rem; font-weight: 600; }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .conf-high { background: #c6f6d5; color: #22543d; }
    .conf-medium { background: #feebc8; color: #744210; }
    .conf-low { background: #fed7d7; color: #742a2a; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:8000/health")


# -----------------------------------------------
# HEADER
# -----------------------------------------------
st.title("🏦 Customer Churn Predictor")
st.write("Predict whether a bank customer will **stay** or **leave** using our AI model with **83.2% accuracy**.")
st.divider()

# -----------------------------------------------
# API STATUS CHECK
# -----------------------------------------------
try:
    r = requests.get(HEALTH_URL.replace("/health", "/"), timeout=2)
    if r.status_code == 200:
        st.markdown('<p class="status-ok">🟢 Connected to prediction API</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">🔴 API connection issue</p>', unsafe_allow_html=True)
except:
    st.markdown('<p class="status-err">🔴 API is offline</p>', unsafe_allow_html=True)
    st.error("⚠️ **Start the FastAPI server first:**\n\n1. Open terminal\n2. Navigate to: `cd app`\n3. Run: `uvicorn main:app --reload`")
    st.stop()

st.write("")

# -----------------------------------------------
# STEP 1 — CUSTOMER INFORMATION
# -----------------------------------------------
st.markdown('<div class="step-header">Step 1 — Customer Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    
with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])

country = st.selectbox("Country", ["France", "Germany", "Spain"])

# -----------------------------------------------
# STEP 2 — FINANCIAL INFORMATION
# -----------------------------------------------
st.markdown('<div class="step-header">Step 2 — Financial Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650,
                            help="300 = Poor | 600 = Average | 900 = Excellent")
    
with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=5000.0, format="%.0f")

salary = st.number_input("Estimated Yearly Salary ($)", min_value=0.0, value=60000.0, step=5000.0, format="%.0f")

# -----------------------------------------------
# STEP 3 — BANKING RELATIONSHIP
# -----------------------------------------------
st.markdown('<div class="step-header">Step 3 — Banking Relationship</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Years with Bank", min_value=0, max_value=10, value=5)
    
with col2:
    num_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=1,
                                help="Savings account, credit card, loan, etc.")

col1, col2 = st.columns(2)

with col1:
    has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    
with col2:
    is_active = st.selectbox("Active Member?", ["Yes", "No"])

# -----------------------------------------------
# STEP 4 — CARD & SATISFACTION
# -----------------------------------------------
st.markdown('<div class="step-header">Step 4 — Card Type & Satisfaction</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])
    
with col2:
    satisfaction = st.select_slider(
        "Satisfaction Level",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {
            1: "😤 Very Unhappy",
            2: "😕 Unhappy",
            3: "😐 Neutral",
            4: "😊 Happy",
            5: "😄 Very Happy"
        }[x]
    )

points = st.number_input("Reward Points Earned", min_value=0, max_value=1500, value=400, step=50)

# -----------------------------------------------
# PREDICT BUTTON
# -----------------------------------------------
st.divider()
predict_clicked = st.button("🔍 Predict Churn")

# -----------------------------------------------
# PREDICTION RESULT
# -----------------------------------------------
if predict_clicked:
    
    # Build payload matching API schema
    payload = {
        "CreditScore": int(credit_score),
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": 1 if has_credit_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": float(salary),
        "Satisfaction_Score": int(satisfaction),
        "Point_Earned": int(points),
        "Geography_Germany": 1 if country == "Germany" else 0,
        "Geography_Spain": 1 if country == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Card_Type_GOLD": 1 if card_type == "GOLD" else 0,
        "Card_Type_PLATINUM": 1 if card_type == "PLATINUM" else 0,
        "Card_Type_SILVER": 1 if card_type == "SILVER" else 0,
    }
    
    with st.spinner("🤖 Analyzing customer profile..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                prediction = data["churn_prediction"]
                churn_prob = data["churn_probability"]
                stay_prob = data["stay_probability"]
                confidence = data.get("confidence", "Medium")
                
                st.divider()
                st.subheader("📊 Prediction Result")
                
                # Main result box
                if prediction == 1:
                    conf_class = f"conf-{confidence.lower()}"
                    st.markdown(f"""
                        <div class="box-churn">
                            <div class="result-emoji">⚠️</div>
                            <div class="result-heading churn-color">Customer Will CHURN</div>
                            <div class="result-msg">
                                <strong>{churn_prob*100:.1f}%</strong> probability of leaving<br>
                                <span class="confidence-badge {conf_class}">Confidence: {confidence}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.warning("💡 **Recommendation:** Consider retention strategies - special offers, personalized support, or investigate satisfaction issues.")
                else:
                    conf_class = f"conf-{confidence.lower()}"
                    st.markdown(f"""
                        <div class="box-stay">
                            <div class="result-emoji">✅</div>
                            <div class="result-heading stay-color">Customer Will STAY</div>
                            <div class="result-msg">
                                <strong>{stay_prob*100:.1f}%</strong> probability of staying<br>
                                <span class="confidence-badge {conf_class}">Confidence: {confidence}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("💡 **Recommendation:** Maintain current relationship quality and continue providing excellent service.")
                
                # Probability breakdown
                st.write("")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                        <div class="prob-card">
                            <div class="prob-num red">{churn_prob*100:.1f}%</div>
                            <div class="prob-lbl">🔴 Churn Risk</div>
                        </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                        <div class="prob-card">
                            <div class="prob-num green">{stay_prob*100:.1f}%</div>
                            <div class="prob-lbl">🟢 Retention</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Input summary
                st.write("")
                with st.expander("📋 View Input Summary"):
                    summary = pd.DataFrame({
                        "Feature": [
                            "Age", "Country", "Gender", "Credit Score",
                            "Account Balance", "Yearly Salary", "Years with Bank",
                            "Number of Products", "Has Credit Card", "Active Member",
                            "Card Type", "Satisfaction", "Reward Points"
                        ],
                        "Value": [
                            f"{age} years", country, gender, f"{credit_score}",
                            f"${balance:,.0f}", f"${salary:,.0f}", f"{tenure} year(s)",
                            f"{num_products}", has_credit_card, is_active,
                            card_type, f"{satisfaction}/5 ⭐", f"{points} points"
                        ]
                    })
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                
            else:
                st.error(f"❌ Prediction failed with error code {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure FastAPI is running on http://127.0.0.1:8000")
            
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Footer
st.divider()
st.caption("🤖 Powered by Random Forest ML Model | 📊 83.2% Test Accuracy | 🎯 0.632 F1 Score")
