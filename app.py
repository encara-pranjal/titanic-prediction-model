import streamlit as st
import pandas as pd
import joblib
from feature_engineering import TitanicFeatureEngineer

try:
    pipeline = joblib.load('best_model_pipeline.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Titanic Survival Prediction App")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
with col2:
    sibsp = st.number_input("Siblings/Spouses (SibSp)", min_value=0, value=0, step=1)
    parch = st.number_input("Parents/Children (Parch)", min_value=0, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, value=15.0, step=1.0)
    embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

if st.button("Predict Survival"):
    default_title = "Mr." if sex == "male" else "Miss."
    fake_name = f"Unknown, {default_title} Unknown"

    input_data = pd.DataFrame([{
        'PassengerId': 0, 'Name': fake_name, 'Ticket': 'Unknown', 'Cabin': None,
        'Pclass': pclass, 'Sex': sex, 'Age': float(age),
        'SibSp': int(sibsp), 'Parch': int(parch), 'Fare': float(fare), 'Embarked': embarked
    }])

    prediction = pipeline.predict(input_data)
    if prediction[0] == 1:
        st.success("Likely to Survive!")
    else:
        st.error("Unlikely to Survive")
