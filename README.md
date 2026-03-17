# Heart-Disease_prediction
❤️ Heart Disease Prediction Project
📌 Overview

This project focuses on predicting the likelihood of heart disease using machine learning techniques. The dataset contains medical attributes of patients, and the goal is to classify whether a person has heart disease (1) or not (0).

Two models were implemented and compared:

Logistic Regression

Decision Tree

📊 Dataset Description

The dataset contains 918 records and 12 features:

🔑 Features and Their Impact

Age: Older individuals have a higher risk of heart disease.

Sex: Males generally show higher risk compared to females.

ChestPainType: Type of chest pain (e.g., typical angina, atypical, asymptomatic). Strong indicator of heart problems.

RestingBP (Blood Pressure): High blood pressure increases strain on the heart.

Cholesterol: High cholesterol leads to artery blockage.

FastingBS: High blood sugar may indicate diabetes, which increases heart risk.

RestingECG: Abnormal ECG results can signal heart issues.

MaxHR (Maximum Heart Rate): Lower max heart rate during exercise can indicate poor heart health.

ExerciseAngina: Chest pain during exercise is a strong warning sign.

Oldpeak: ST depression measure — higher values indicate abnormal heart activity.

ST_Slope: Shape of ECG curve; flat/downward slopes often indicate disease.

HeartDisease (Target):

0 → No disease

1 → Heart disease

🔍 Exploratory Data Analysis (EDA)

We performed EDA to understand the dataset:

Checked dataset shape and structure

Identified data types (numerical & categorical)

Verified missing values (none found)

Detected duplicate records (none found)

Visualized distributions using histograms

Observed class balance for the target variable

🧹 Data Preprocessing

To prepare the data for modeling:

Handled invalid values

Replaced 0 values in Cholesterol and RestingBP with mean values

Encoding categorical variables

Used pd.get_dummies() to convert categories into numeric form

Feature Scaling

Applied StandardScaler to normalize numerical features

Final dataset

Converted all values to numeric format for model compatibility

🤖 Models Used
1️⃣ Logistic Regression

A linear model that predicts probability using weighted features

Works well with structured and linearly separable data

2️⃣ Decision Tree

A rule-based model that splits data using conditions

Captures non-linear relationships but can overfit

📈 Model Evaluation
✅ Accuracy

Logistic Regression: 88%

Decision Tree: 79%

📊 Confusion Matrix

A confusion matrix shows how well the model predicted:

	Predicted 0	Predicted 1
Actual 0	True Negatives	False Positives
Actual 1	False Negatives	True Positives

👉 Helps understand:

How many correct predictions were made

Types of errors (false positives & false negatives)

 ROC Curve (Receiver Operating Characteristic)

Plots True Positive Rate vs False Positive Rate

Shows model performance across all thresholds

 AUC (Area Under Curve):

Closer to 1 → better model

Around 0.5 → random guessing

🧠 Why Logistic Regression Performed Better

Logistic Regression achieved higher accuracy because:

The dataset has relatively linear relationships

Feature scaling improved its performance

It generalizes better on smaller datasets

The Decision Tree:

Likely overfitted the training data

Uses rigid splits, which may miss smooth patterns

🚀 Conclusion

Logistic Regression proved to be more effective for this dataset

Decision Tree can be improved using techniques like pruning or ensemble methods (e.g., Random Forest)

Proper preprocessing and understanding of data played a key role in model performance
