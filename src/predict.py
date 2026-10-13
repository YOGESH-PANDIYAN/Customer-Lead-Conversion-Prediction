
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/lead_conversion_model.pkl")

# New customer details
new_lead = pd.DataFrame([{
    "age": 25,
    "location": "Madurai",
    "budget": 10000000,
    "property_type": "Plot",
    "lead_source": "Website",
    "calls": 3,
    "follow_ups": 4,
    "site_visit": 0,
    "response_time_hours": 6
}])

# Predict conversion
prediction = model.predict(new_lead)

# Calculate probabilities
probability = model.predict_proba(new_lead)[0]

print("\nCustomer Lead Prediction")
print("--------------------------")

if prediction[0] == 1:
    print("Prediction: Likely to Convert")
else:
    print("Prediction: Unlikely to Convert")

print(f"Conversion Probability: {probability[1] * 100:.2f}%")
print(f"Non-Conversion Probability: {probability[0] * 100:.2f}%")
