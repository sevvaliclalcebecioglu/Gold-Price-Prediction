import streamlit as st
import pandas as pd
import joblib

# Load model & columns
model = joblib.load("data/best_model.pkl")
columns = joblib.load("data/columns.pkl")

# Load cleaned dataset (already engineered)
df = pd.read_csv("data/cleaned_gold_price_data.csv")
df = df.dropna().reset_index(drop=True)

st.set_page_config(
    page_title="Gold Price Prediction",
    layout="centered"
)

st.title("💰 Gold Price Prediction")
st.write("Regression-based gold price prediction using engineered financial features.")

st.divider()

# Show latest known price
latest_row = df.iloc[-1]

st.metric(
    label="📌 Latest Known Gold Price (USD PM)",
    value=f"{latest_row['USD (PM)']:.2f}"
)

st.divider()

st.subheader("🔮 Model-based price estimation")

if st.button("Predict"):
    # Build input from latest row
    input_df = pd.DataFrame([latest_row])

    # Drop target column
    input_df = input_df.drop(columns=["USD (PM)"])

    # Match training column order
    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]

    st.success(
        f"💎 Predicted Gold Price (USD PM): **{prediction:.2f}**"
    )
