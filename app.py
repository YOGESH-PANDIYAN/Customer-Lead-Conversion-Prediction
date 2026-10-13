
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_lead_model.pkl"

# Page configuration
st.set_page_config(
    page_title="Customer Lead Conversion Predictor",
    page_icon="📊",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# Dashboard heading
st.title("📊 Customer Lead Conversion Predictor")

st.write(
    "An interactive Machine Learning application "
    "for estimating customer lead conversion."
)

st.warning(
    "Portfolio demonstration only. This model was trained "
    "on synthetic data and is not validated for real "
    "business decisions."
)

st.divider()

# Model information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Records", "2,000")

with col2:
    st.metric("Test Accuracy", "71.25%")

with col3:
    st.metric("Best Algorithm", "Logistic Regression")

st.divider()

st.subheader("Enter Customer Details")

# Input form
with st.form("lead_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=25
        )

        location = st.selectbox(
            "Location",
            ["Chennai", "Madurai", "Coimbatore"]
        )

        budget = st.number_input(
            "Property Budget (INR)",
            min_value=0,
            value=10000000,
            step=100000
        )

        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "Villa", "Plot"]
        )

        lead_source = st.selectbox(
            "Lead Source",
            ["Website", "Google", "Facebook"]
        )

    with col2:
        calls = st.number_input(
            "Number of Calls",
            min_value=0,
            value=3
        )

        follow_ups = st.number_input(
            "Number of Follow-ups",
            min_value=0,
            value=4
        )

        site_visit = st.selectbox(
            "Site Visit Completed?",
            ["No", "Yes"]
        )

        response_time = st.number_input(
            "Response Time (Hours)",
            min_value=0,
            value=6
        )

    submitted = st.form_submit_button(
        "Predict Lead Conversion",
        type="primary",
        use_container_width=True
    )

# Prediction
if submitted:

    new_lead = pd.DataFrame([{
        "age": age,
        "location": location.lower(),
        "budget": budget,
        "property_type": property_type.lower(),
        "lead_source": lead_source.lower(),
        "calls": calls,
        "follow_ups": follow_ups,
        "site_visit": 1 if site_visit == "Yes" else 0,
        "response_time_hours": response_time
    }])

    prediction = model.predict(new_lead)[0]
    probability = model.predict_proba(new_lead)[0][1]

    st.divider()
    st.subheader("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.success("Likely to Convert")
        else:
            st.error("Unlikely to Convert")

    with result_col2:
        st.metric(
            "Estimated Conversion Probability",
            f"{probability * 100:.2f}%"
        )

    st.progress(float(probability))

    st.caption(
        "The probability is an uncalibrated model estimate "
        "from synthetic training data."
    )

    st.subheader("Customer Input Summary")
    st.dataframe(
        new_lead,
        use_container_width=True,
        hide_index=True
    )
