import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from feature_engineering import TitanicFeatureEngineer

# 1. Load Data
df = pd.read_csv('train.csv')
X = df.drop(columns=['Survived'])
y = df['Survived']

# 2. Re-create Pipeline (with updated class logic)
num_features = ['Age', 'Fare', 'individual_fare', 'FamilySize']
cat_features = ['Sex', 'Embarked', 'Pclass', 'Title', 'Family_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

full_pipeline = Pipeline([
    ('eng', TitanicFeatureEngineer()),
    ('prep', preprocessor),
    ('model', LogisticRegression(class_weight='balanced', random_state=42))
])

# 3. Fit & Save
full_pipeline.fit(X, y)
joblib.dump(full_pipeline, 'best_model_pipeline.pkl')

print("--- Data Engineering Fix Applied and Pipeline Regenerated ---")
