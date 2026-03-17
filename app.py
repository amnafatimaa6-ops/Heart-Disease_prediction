# 1️ Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 2️ Load dataset (make sure heart.csv is in same folder)
df = pd.read_csv("heart.csv")

# 3️ Preprocessing (same as training)

# Replace 0 values
df['Cholesterol'] = df['Cholesterol'].replace(0, df[df['Cholesterol'] != 0]['Cholesterol'].mean())
df['RestingBP'] = df['RestingBP'].replace(0, df[df['RestingBP'] != 0]['RestingBP'].mean())

# One-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Convert to int
df_encoded = df_encoded.astype(int)

# Scale numerical features
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Split features and target
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# 4️ Train models
log_model = LogisticRegression()
log_model.fit(X, y)

tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X, y)

# 5️ Streamlit UI
st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

# User inputs
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
bp = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxhr = st.slider("Max Heart Rate", 60, 200, 120)
angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# 6 Convert input into dataframe
input_dict = {
    'Age': age,
    'RestingBP': bp,
    'Cholesterol': chol,
    'FastingBS': fbs,
    'MaxHR': maxhr,
    'Oldpeak': oldpeak,
    'Sex_M': 1 if sex == "M" else 0,
    'ChestPainType_ATA': 1 if cp == "ATA" else 0,
    'ChestPainType_NAP': 1 if cp == "NAP" else 0,
    'ChestPainType_TA': 1 if cp == "TA" else 0,
    'RestingECG_Normal': 1 if ecg == "Normal" else 0,
    'RestingECG_ST': 1 if ecg == "ST" else 0,
    'ExerciseAngina_Y': 1 if angina == "Y" else 0,
    'ST_Slope_Flat': 1 if slope == "Flat" else 0,
    'ST_Slope_Up': 1 if slope == "Up" else 0
}

input_df = pd.DataFrame([input_dict])

# Ensure same column order
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale input
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

#  Prediction button
if st.button("Predict"):

    # Logistic Regression prediction
    log_pred = log_model.predict(input_df)[0]
    log_prob = log_model.predict_proba(input_df)[0][1]

    # Decision Tree prediction
    tree_pred = tree_model.predict(input_df)[0]
    tree_prob = tree_model.predict_proba(input_df)[0][1]

    # Display results
    st.subheader("📊 Results")

    st.write("### Logistic Regression")
    st.write("Prediction:", "Heart Disease" if log_pred == 1 else "No Heart Disease")
    st.write(f"Confidence: {log_prob:.2f}")

    st.write("### Decision Tree")
    st.write("Prediction:", "Heart Disease" if tree_pred == 1 else "No Heart Disease")
    st.write(f"Confidence: {tree_prob:.2f}")
