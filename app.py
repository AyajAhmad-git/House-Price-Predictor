import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")


st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://www.shutterstock.com/image-photo/row-new-construction-homes-share-260nw-2503222899.jpg"
             alt="California Housing" width="500">
        <h1 style="color:#2c3e50; margin-top:20px;">California House Price Predictor</h1>
        <hr style="border: 1px solid #ccc;">
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Enter the details below to predict Median House Value")


longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)


# Predict Button

if st.button("Predict"):

    # Create dataframe
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    # Transform using pipeline
    prepared_data = pipeline.transform(input_df)

    # Predict
    prediction = model.predict(prepared_data)[0]

    # Output Box
    st.markdown(
        f"""
        <div style="padding:20px; background:#ecf0f1; border-radius:10px;">
            <h2 style="color:#27ae60;">Predicted House Price:</h2>
            <h1 style="color:#2c3e50;">${prediction:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <br><br>
    <hr>
    <div style="text-align:center; color:gray;">
        <p>Developed by Ayaj Ahmad | Powered by Streamlit & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)