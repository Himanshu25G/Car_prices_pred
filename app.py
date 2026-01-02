# demo app for car price prediction model
import streamlit as st
import joblib
import numpy as np
import pandas as pd


# load the trained model
model = joblib.load("car_price_model.pkl")


st.title("Car Price Prediction App")
st.write("Enter the details of the car to predict its selling price.")

# input fields
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1 ,max_value=100.0)
km_driven = st.number_input("Kms Driven (in lakhs)", min_value=0.0, step=0.1 )
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", options=["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", options=["Manual", "Automatic"])
owner = st.selectbox("Number of Previous Owners", options=[0, 1, 2, 3])
age = st.number_input("Age of the Car (in years)", min_value=0, step=1)



# encode categorical (have done ordinal encoding in the training data) inputs
fuel_type_encoded = 0 if fuel_type == "Petrol" else 1 if fuel_type == "Diesel" else 2
seller_type_encoded = 0 if seller_type == "Dealer" else 1
transmission_encoded = 0 if transmission == "Manual" else 1
owner_encoded = owner  # already numeric


# predict button
if st.button("Predict Selling Price"):
    input_data = (present_price, km_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, owner_encoded, age)
    input_array = np.array(input_data).reshape(1, -1)
    predicted_price = model.predict(input_array)
    st.success(f"The predicted selling price of the car is Rs. {predicted_price[0]:.2f} lakhs")



st.write("Developed by Himanshu Sharma")
st.write("Data Science Enthusiast")
st.write("GitHub: [Himanshu25](https://github.com/Himanshu25)")
st.write("LinkedIn: [Himanshu Ranjan](https://www.linkedin.com/in/himanshu-ranjan-25g/)")

st.write("Feedback and contributions are welcome!")



         





