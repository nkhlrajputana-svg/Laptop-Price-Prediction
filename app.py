import joblib
import pandas as pd
import streamlit as st
import pickle



# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("📊 Machine Learning Prediction App")

# Laptop Image
st.image("https://images.unsplash.com/photo-1517336714731-489689fd1ca8", caption="AI Model Running on Laptop", width=700)
st.write("Enter input values and get prediction from your trained model")

# ---- USER INPUT FORM ----
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        ram = st.number_input("RAM")
        
    with col2:
        inches = st.number_input("INCHES ")
        

    submitted = st.form_submit_button("Predict")

# ---- PREDICTION ----
if submitted:
    input_data = pd.DataFrame([[ram, inches]])
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Value: ${prediction}")

st.markdown("---")
st.caption("Built with Streamlit")
