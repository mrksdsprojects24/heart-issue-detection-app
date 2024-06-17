#!pip install streamlit pandas scikit-learn

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.linear_model import LogisticRegression

#df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
# Due to problems with SMOTE on my laptop, we ran SMOTE on colab and downloaded the oversampled dataframe onto our laptop.
df = pd.read_csv("oversampled_data.csv")

df = df.dropna()
X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']

# Training a Logistic Regression Model
model = LogisticRegression(solver="liblinear")
model.fit(X, y)


import streamlit as st

st.title("Heart Disease Risk Assessment app by Krishna")

st.write("Please enter your information using the slider and checkboxes to calculate risk of heart issue.")

# Get user input
general_health = 6 - st.slider("General Health (1-5)", 1, 5, 3)
diabetes = st.checkbox("Diagnosed with Diabetes?")
high_bp = st.checkbox("High Blood Pressure?")
high_cholestoral = st.checkbox("High Cholesterol?")
smoker = st.checkbox("Do you smoke?")
heavy_drinker = 1 - st.checkbox("Do you drink more than the accepted limits?")

# Convert user input to a DataFrame
user_data = pd.DataFrame({
    "Diabetes_binary": [int(diabetes)],       # User-input
    "HighBP": [int(high_bp)],                 # User-input
    "HighChol": [int(high_cholestoral)],      # User-input
    "CholCheck": [df['CholCheck'].median()],
    "BMI": [df['BMI'].median()],
    "Smoker": [int(smoker)],                  # User-input
    "Stroke": [df['Stroke'].median()],
    "PhysActivity": [df['PhysActivity'].median()],
    "Fruits": [df['Fruits'].median()],
    "Veggies": [df['Veggies'].median()],
    "HvyAlcoholConsump": [int(heavy_drinker)],         # User-input
    "AnyHealthcare": [df['AnyHealthcare'].median()],
    "NoDocbcCost": [df['NoDocbcCost'].median()],
    "GenHlth": [general_health],              # User-input
    "MentHlth": [df['MentHlth'].median()],
    "PhysHlth": [df['PhysHlth'].median()],
    "DiffWalk": [df['DiffWalk'].median()],
    "Sex": [df['Sex'].median()],
    "Age": [df['Age'].median()],
    "Education": [df['Education'].median()],
    "Income": [df['Income'].median()],
})

# Make predictions
prediction = model.predict_proba(user_data)[:, 1]
risk_score = round(prediction[0] * 100, 2)
if risk_score > 10:
    risk_score = risk_score + 10

# Display results
st.write("As per my machine learning algorithm trained on a dtataset created from the Behavioral Risk Factor Surveillance System (BRFSS) by the CDC (Center for Disease Control and Prevention), your estimated risk of heart disease is:")
st.progress(risk_score / 100)
st.write(f"{risk_score}%")

if risk_score < 20:
    st.success("Very little risk")
elif risk_score < 40:
    st.success("Low risk")
elif risk_score < 60:
    st.warning("Moderate risk - Do moderate exercise!")
elif risk_score < 75:
    st.warning("High risk - Do exercises regularly and watch your calaries!")
else:
    st.error("Very high risk. Please consult a doctor!")
