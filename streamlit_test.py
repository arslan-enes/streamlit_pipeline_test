import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained pipeline
pipeline = joblib.load('pipeline_example.joblib')

# Create a Streamlit web interface
st.title("Predictive Model with Streamlit")

# Create input widgets for user input
age = st.slider("Age:", min_value=18, max_value=100, value=30)
income = st.slider("Income:", min_value=0, max_value=200000, value=50000)
education = st.selectbox("Education:", ['Bachelors', 'Masters', 'PhD', 'High School'])
gender = st.radio("Gender:", ['Male', 'Female'])

# Create a DataFrame from user input
user_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'education': [education],
    'gender': [gender]
})

# Make predictions using the loaded pipeline
if st.button("Predict"):
    prediction = pipeline.predict(user_data)
    prediction_proba = pipeline.predict_proba(user_data)[:, 1]

    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.write("The model predicts that the target is 1 (Positive).")
    else:
        st.write("The model predicts that the target is 0 (Negative).")

    st.subheader("Prediction Probability:")
    st.write(f"The probability of the positive class is: {prediction_proba[0]:.2f}")

# Add some additional information or explanations if needed
st.write("""
This is a simple Streamlit app that uses a pre-trained scikit-learn pipeline to make predictions based on user input. Adjust the sliders and select options, then click the 'Predict' button to see the model's prediction and prediction probability.
""")
