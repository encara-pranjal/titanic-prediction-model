import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()

        # 1. Ensure 'Name' is string and handle missing values
        X['Name'] = X['Name'].fillna('Unknown, Mr. Unknown')
        X['Name'] = X['Name'].astype(str)

        # Title extraction
        X['Title'] = X['Name'].str.split(',').str.get(1).str.strip().str.split(' ').str.get(0)
        X['Title'] = X['Title'].replace(['Rev.', 'Dr.', 'Col.', 'Major.', 'Don.', 'Capt.', 'the', 'Jonkheer.', 'Dona.'], 'Other')

        # Ensure numeric columns are numeric for safety
        X['SibSp'] = pd.to_numeric(X['SibSp'], errors='coerce').fillna(0)
        X['Parch'] = pd.to_numeric(X['Parch'], errors='coerce').fillna(0)
        X['Fare'] = pd.to_numeric(X['Fare'], errors='coerce').fillna(0)

        # Family features
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['individual_fare'] = X['Fare'] / X['FamilySize']
        X['Family_type'] = pd.cut(X['FamilySize'], bins=[0, 1, 4, 11], labels=['alone', 'small', 'large'])

        # Drop redundant columns
        return X.drop(columns=['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], errors='ignore')
