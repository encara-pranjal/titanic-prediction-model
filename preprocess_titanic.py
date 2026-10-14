import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('train.csv')

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def engineer_features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.split(',').str.get(1).str.strip().str.split(' ').str.get(0)
    df['Title'] = df['Title'].replace(['Rev.', 'Dr.', 'Col.', 'Major.', 'Don.', 'Capt.', 'the', 'Jonkheer.', 'Dona.'], 'Other')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['individual_fare'] = df['Fare'] / df['FamilySize']
    df['Family_type'] = pd.cut(df['FamilySize'], bins=[0, 1, 4, 11], labels=['alone', 'small', 'large'])
    return df.drop(columns=['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], errors='ignore')

X_train_eng = engineer_features(X_train)
X_test_eng = engineer_features(X_test)

num_features = ['Age', 'Fare', 'individual_fare', 'FamilySize']
cat_features = ['Sex', 'Embarked', 'Pclass', 'Title', 'Family_type']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

X_train_final = preprocessor.fit_transform(X_train_eng)
X_test_final = preprocessor.transform(X_test_eng)

print("--- Chunk 2 Preprocessing Report ---")
print(f"Training shape: {X_train_final.shape}")
print(f"Testing shape: {X_test_final.shape}")
print(f"Missing values after preprocessing: {np.isnan(X_train_final).sum()}")
