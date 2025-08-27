import streamlit as st
import pickle
import numpy as np

# Load the trained Logistic Regression model
model = pickle.load(open("model.pkl", "rb"))

st.title("🎓 Student Placement Prediction App")

st.write("This app predicts whether a student will be placed based on **CGPA** and **IQ**.")

# Taking inputs
cgpa = st.number_input("Enter CGPA:", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ:", min_value=50, max_value=200, step=1)

# Prediction button
if st.button("Predict Placement"):
    input_data = np.array([[cgpa, iq]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # probability of placement

    if prediction == 1:
        st.success(f"✅ Student is likely to be **Placed** (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Student is **Not Placed** (Probability: {prob:.2f})")
