import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("heart.csv")

# Preprocessing
df['Cholesterol'] = df['Cholesterol'].replace(0, df[df['Cholesterol'] != 0]['Cholesterol'].mean())
df['RestingBP'] = df['RestingBP'].replace(0, df[df['RestingBP'] != 0]['RestingBP'].mean())

df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.astype(int)

# Scaling
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Split
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# Train models
log_model = LogisticRegression()
log_model.fit(X, y)

tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X, y)

joblib.dump(log_model, "log_model.pkl")
joblib.dump(tree_model, "tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Models saved!")
