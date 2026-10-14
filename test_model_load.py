import joblib
import pandas as pd
from feature_engineering import TitanicFeatureEngineer

pipeline = joblib.load('best_model_pipeline.pkl')

sample_passenger = pd.DataFrame([{
    'PassengerId': 1, 'Pclass': 3, 'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male',
    'Age': 22.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25,
    'Cabin': None, 'Embarked': 'S'
}])

prediction = pipeline.predict(sample_passenger)
print(f"Test Successful: Prediction = {prediction[0]}")
