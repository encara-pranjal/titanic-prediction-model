# Titanic Survival Prediction

Ye ek simple Machine Learning project hai jahan hum predict karte hain ki ek passenger Titanic ship par survive karega ya nahi.

## Project Structure
* `train.csv` — Training dataset (891 passengers).
* `titanic.ipynb` — Exploratory Data Analysis (EDA) notebook.
* `preprocess_titanic.py` — Data cleaning & feature engineering steps.
* `train_models.py` — 5 algorithms ko train aur evaluate karne ka script.
* `save_model.py` — Best model + preprocessing pipeline ko save karne ka script.
* `feature_engineering.py` — `TitanicFeatureEngineer` class (pipeline ke andar use hoti hai).
* `best_model_pipeline.pkl` — Saved Logistic Regression pipeline (preprocessing included).
* `model_comparison.csv` — Sabhi 5 models ke actual metrics.
* `test_model_load.py` — Quick check: model load ho raha hai ya nahi.
* `app.py` — Streamlit web app for real-time predictions.
* `requirements.txt` — Python dependencies.

## Setup & Installation

### 1. Python install karein (agar nahi hai)
Python 3.10 ya usse naya chahiye. Check karne ke liye:
```bash
python --version
```

### 2. Project folder mein jayein
```bash
cd model training
```

### 3. Virtual environment banayein (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux
```

### 4. Requirements install karein
```bash
pip install -r requirements.txt
```

## Kaise Run Karein (Local)

### Model verify karein
```bash
python train_models.py     # 5 models train karke model_comparison.csv regenerate karega
python test_model_load.py  # saved pipeline load karke ek sample prediction dega
```

### Streamlit app start karein
```bash
streamlit run app.py
```
Phir browser mein **http://localhost:8501** open karein. Passenger ki details bharo aur "Predict Survival" button dabao.

## Models Compared
Humne 5 algorithms test kiye hain, sab same stratified 80/20 train-test split pe:

1. Logistic Regression (**Winner** ✅)
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. K-Nearest Neighbors (KNN)

## Comparison Results (actual metrics)

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.8324** | 0.7671 | **0.8116** | **0.7887** |
| SVM | 0.8268 | 0.7794 | 0.7681 | 0.7737 |
| Decision Tree | 0.8101 | 0.7778 | 0.7101 | 0.7424 |
| Random Forest | 0.8045 | 0.7576 | 0.7246 | 0.7407 |
| KNN | 0.8045 | 0.7742 | 0.6957 | 0.7328 |

Yehi values `model_comparison.csv` mein save hain aur `train_models.py` se reproduce hoti hain.

## Deployment Status
⚠️ **App abhi sirf local chal raha hai — online deploy Nahi kiya gaya hai.**

Koi live URL nahi hai. Deployment (Streamlit Community Cloud / Hugging Face Spaces) abhi pending hai.

## Notes
- Preprocessing (imputation, scaling, one-hot encoding) saved pipeline ke andar hai, isliye `app.py` raw passenger data directly pass kar sakta hai.
- `titanic.ipynb` ke kuch cells mein local file paths hain (jaise `C:\Users\encar\Downloads\test.csv`) — doosre computer par wo cells fail karenge. ML scripts (`train_models.py`, `save_model.py`, `app.py`) in paths par depend nahi karte.
