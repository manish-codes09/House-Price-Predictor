import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files
model = joblib.load("house_price_model.pkl")
encoder = joblib.load("encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

# Load dataset only for dropdown values
df = pd.read_csv("data.csv")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")

st.write("Enter house details and predict the price.")

# Numeric Inputs
bedrooms = st.slider(
    "Bedrooms",
    min_value=0,
    max_value=9,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=0.0,
    max_value=5.75,
    value=2.0,
    step=0.25
)

sqft_living = st.number_input(
    "Sqft Living",
    min_value=370,
    max_value=7320,
    value=2000
)

sqft_lot = st.number_input(
    "Sqft Lot",
    min_value=638,
    max_value=1074218,
    value=5000
)

floors = st.number_input(
    "Floors",
    min_value=1.0,
    max_value=3.5,
    value=1.0,
    step=0.5
)

view = st.slider(
    "View",
    min_value=0,
    max_value=4,
    value=0
)

condition = st.slider(
    "Condition",
    min_value=1,
    max_value=5,
    value=3
)
# City Dropdown
city = st.selectbox(
    "Select City",
    sorted(df["city"].unique())
)

# Show only statezips related to selected city
available_zips = sorted(
    df[df["city"] == city]["statezip"].unique()
)

statezip = st.selectbox(
    "Select StateZip",
    available_zips
)

# Prediction
if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "view": [view],
        "condition": [condition],
        "city": [city],
        "statezip": [statezip]
    })

    # Encode city and statezip
    encoded = encoder.transform(
        input_df[["city", "statezip"]]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            ["city", "statezip"]
        )
    )

    # Merge numeric + encoded
    final_df = pd.concat(
        [
            input_df.drop(
                ["city", "statezip"],
                axis=1
            ),
            encoded_df
        ],
        axis=1
    )

    # Add missing columns
    for col in model_columns:
        if col not in final_df.columns:
            final_df[col] = 0

    # Correct column order
    final_df = final_df[model_columns]

    # Predict
    prediction_log = model.predict(final_df)[0]

    # Reverse log transform
    predicted_price = np.expm1(prediction_log)

    st.success(
        f"Predicted House Price: ${predicted_price:,.0f}"
    )