import streamlit as st
import pandas as pd
import joblib

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

numeric_features = [
    "bedrooms",
    "bathrooms",
    "car_park",
    "land_size",
    "distance_to_cbd"
]

categorical_features = [
    "suburb",
    "property_type"
]

st.title("Sydney Housing Price Prediction")

st.write(
    "Enter property details below to estimate the sale price."
)

suburb = st.selectbox(
    "Suburb",
    ["Liverpool", "Marrickville", "Mosman"]
)

property_type = st.selectbox(
    "Property Type",
    ["Apartment", "House", "Townhouse"]
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=20,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=15,
    value=2
)

car_park = st.number_input(
    "Car Spaces",
    min_value=0,
    max_value=10,
    value=1
)

land_size = st.number_input(
    "Land Size (m²)",
    min_value=1.0,
    value=300.0
)

distance_to_cbd = st.number_input(
    "Distance to Sydney CBD (km)",
    min_value=0.0,
    value=15.0
)

if st.button("Predict Sale Price"):

    input_data = pd.DataFrame({
        "suburb": [suburb.lower()],
        "property_type": [property_type],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "car_park": [car_park],
        "land_size": [land_size],
        "distance_to_cbd": [distance_to_cbd]
    })

    numeric_scaled = scaler.transform(
        input_data[numeric_features]
    )

    numeric_df = pd.DataFrame(
        numeric_scaled,
        columns=numeric_features
    )

    categorical_encoded = encoder.transform(
        input_data[categorical_features]
    )

    categorical_df = pd.DataFrame(
        categorical_encoded,
        columns=encoder.get_feature_names_out(
            categorical_features
        )
    )

    final_input = pd.concat(
        [numeric_df, categorical_df],
        axis=1
    )

    prediction = model.predict(final_input)[0]

    st.success(
        f"Estimated Sale Price: ${prediction:,.0f}"
    )