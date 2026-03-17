import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load saved models
# -----------------------------
log_model = joblib.load("log_model.pkl")
tree_model = joblib.load("tree_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Predictor")
st.title("❤️ Heart Disease Prediction App")

st.write("Fill patient details below:")

# Inputs
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
bp = st.slider("Resting BP", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar (1 = High)", [0, 1])
ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxhr = st.slider("Max Heart Rate", 60, 200, 120)
angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -----------------------------
# Prepare input
# -----------------------------
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

# Match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    log_prob = log_model.predict_proba(input_df)[0][1]
    tree_prob = tree_model.predict_proba(input_df)[0][1]

    st.subheader("📊 Results")

    st.write(f"**Logistic Regression:** {'Heart Disease' if log_prob>0.5 else 'No Disease'} ({log_prob:.2f})")
    st.write(f"**Decision Tree:** {'Heart Disease' if tree_prob>0.5 else 'No Disease'} ({tree_prob:.2f})")

    # -----------------------------
    # Comparison Chart
    # -----------------------------
    st.subheader("📈 Model Comparison")

    fig, ax = plt.subplots()
    ax.bar(["Logistic", "Decision Tree"], [log_prob, tree_prob])
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # -----------------------------
    # Feature Importance (Tree)
    # -----------------------------
    st.subheader("🔥 Feature Importance (Decision Tree)")

    importance = tree_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    ax2.barh(feat_df["Feature"], feat_df["Importance"])
    ax2.invert_yaxis()
    st.pyplot(fig2)
