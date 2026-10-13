import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load Data
df = pd.read_csv('train.csv')

# 2. Split Data (Stratified 80/20) - SAME AS PREVIOUS CHUNKS
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Engineering Helper
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

# 4. Pipeline Definition
num_features = ['Age', 'Fare', 'individual_fare', 'FamilySize']
cat_features = ['Sex', 'Embarked', 'Pclass', 'Title', 'Family_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# 5. Fit & Transform
X_train_final = preprocessor.fit_transform(X_train_eng)
X_test_final = preprocessor.transform(X_test_eng)

# 6. Train & Evaluate Models with Confusion Matrix
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(class_weight='balanced', random_state=42)
}

results = []
print("--- Chunk 3: Detailed Evaluation ---")
for name, model in models.items():
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

    print(f"\n{name} Confusion Matrix:\n{cm}")

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
results_df.to_csv('model_comparison.csv', index=False)
print("\n--- Final Summary Table ---")
print(results_df.to_string(index=False))
