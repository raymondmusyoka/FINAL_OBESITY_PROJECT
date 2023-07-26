import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



df = pd.read_csv("1clean_obesity_data.csv")

st.write("""
# Simple Obesity Prediction App

This app predicts the obesity levels in human beings
""")

st.subheader('Training Data')
st.write(df.head())
st.write(df.describe())

st.subheader('Visualization')
st.bar_chart(df)

X = df.drop("Obesitylevel", axis=1)
y = df['Obesitylevel']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

st.sidebar.header('User Input Parameters')

def user_features():
    Gender = st.sidebar.slider('GENDER', 0, 1)
    Age = st.sidebar.slider("AGE", 0, 150)
    Weight = st.sidebar.slider('WEIGHT', 0.00, 200.00)
    Height = st.sidebar.slider('HEIGHT', 0.00, 2.00)
    family_history_with_overweight = st.sidebar.slider('FAMILY HISTORY WITH OVERWEIGHT', 0, 1)
    FAVC = st.sidebar.slider('VEGETABLE CONSUMPTION', 0, 1)
    CH2O = st.sidebar.slider("DAILY WATER INTAKE", 0, 3)
    FCVC = st.sidebar.slider("HIGH CALORIES CONSUMPTION", 0, 3)
    CALC = st.sidebar.slider("SPECIAL DIET", 0, 3)

    user_features = {
        'Gender': Gender,
        'Age': Age,
        'WEIGHT': Weight,
        'HEIGHT': Height,
        'FAMILY HISTORY WITH OVERWEIGHT': family_history_with_overweight,
        'FAVC': FAVC,
        'VEGETABLE CONSUMPTION FREQUENCY': FCVC,
        'DAILY WATER INTAKE': CH2O,
        'WHETHER FOLLOWS SPECIAL DIET': CALC
    }

    features = pd.DataFrame(user_features, index=[0])
    features_scaled = scaler.transform(features)
    return features_scaled

data = user_features()

np.random.seed(42)

clf = RandomForestClassifier(bootstrap=True,
                             class_weight='balanced_subsample',
                             criterion='entropy',
                             max_features=None,
                             max_samples=None,
                             n_estimators=200)

clf.fit(X_train, y_train)

feature_importances = clf.feature_importances_

st.subheader('Feature Importance')
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
st.write(importance_df)

top_features = importance_df.head(5)['Feature'].tolist()

st.subheader('Top Important Features')
st.write(top_features)

prediction = clf.predict(data)
prediction_proba = clf.predict_proba(data)

st.subheader('Accuracy')
accuracy = accuracy_score(y_test, clf.predict(X_test))
st.write(f"Accuracy: {accuracy * 100:.2f}%")

st.subheader('Prediction')
Obesitylevel = {
    0: "Normal_Weight",
    1: "Overweight_Level_I",
    2: "Overweight_Level_II",
    3: "Obesity_Type_I",
    4: "Insufficient_Weight",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

prediction_level = Obesitylevel.get(prediction[0], "Unknown")
st.write(f"Predicted Obesity Level: {prediction_level}")
