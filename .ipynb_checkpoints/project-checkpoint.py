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
st.markdown("""
<div style="
background: linear-gradient(135deg, #4f46e5, #7c3aed);
padding: 40px;
border-radius: 25px;
text-align: center;
color: white;
margin-bottom: 25px;
box-shadow: 0px 10px 30px rgba(0,0,0,0.25);
border: 1px solid rgba(255,255,255,0.1);
">

<h1 style="
margin: 0;
font-size: 48px;
font-weight: 800;
">
🏠 House Price Predictor
</h1>

<p style="
font-size: 20px;
margin-top: 10px;
color: #e2e8f0;
">
Predict House Prices Using Machine Learning
</p>

<div style="
display: inline-block;
padding: 10px 20px;
background: rgba(255,255,255,0.15);
border-radius: 30px;
font-size: 14px;
font-weight: 600;
margin-top: 10px;
">
🤖 ML Powered • 📊 Real Estate Analytics • 🚀 Streamlit App
</div>

</div>
""", unsafe_allow_html=True)



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
    
        
    st.markdown(f"""
<div style="
background:linear-gradient(135deg,#10b981,#22c55e);
padding:30px;
border-radius:20px;
text-align:center;
color:white;
margin-top:20px;
">
<h2>Predicted House Price</h2>
<h1>${predicted_price:,.0f}</h1>
</div>
""", unsafe_allow_html=True)

