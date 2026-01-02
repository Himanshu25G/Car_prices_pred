# 🚗 Car Price Prediction App

An end-to-end Machine Learning web application that predicts the **selling price of a car** based on user inputs such as present price, kilometers driven, fuel type, seller type, transmission, owner and age.

The project demonstrates **proper ML engineering practices** using:
- ColumnTransformer
- Pipeline
- Gradient Boosting Regressor
- Streamlit for deployment

---

## 🔍 Features

- Predict car selling price in real-time
- Handles numerical & categorical data correctly
- Uses Scikit-learn Pipeline & ColumnTransformer
- Clean, simple Streamlit UI
- Easily deployable on Streamlit Cloud

---

## 🧠 Machine Learning Workflow
User Input
↓
ColumnTransformer
├── Numerical Features → StandardScaler
└── Categorical Features → map
↓
GradientBoostingRegressor
↓
Predicted Car Price



