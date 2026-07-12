import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==========================
# Load Model and Files
# ==========================

model = load_model(r"C:/Users/admin/Downloads/movie_success_ann.keras")
scaler = joblib.load(r"C:/Users/admin/Downloads/scaler.pkl")
encoder = joblib.load(r"C:/Users/admin/Downloads/genre_encoder.pkl")

# ==========================
# Streamlit Page
# ==========================

st.set_page_config(page_title="Movie Success Prediction", page_icon="🎬")

st.title("🎬 Movie Success Prediction using ANN")

st.write("Enter movie details below.")

# ==========================
# User Inputs
# ==========================

budget = st.number_input("Budget ($)", min_value=1000, value=100000000)

duration = st.number_input("Duration (minutes)", min_value=60, value=120)

genre = st.selectbox(
    "Genre",
    encoder.classes_
)

imdb_score = st.slider("IMDb Score", 1.0, 10.0, 7.0)

num_voted_users = st.number_input(
    "Number of Voted Users",
    min_value=0,
    value=50000
)

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    # Encode genre
    genre = encoder.transform([genre])[0]

    # Create input array
    data = np.array([[
        budget,
        duration,
        genre,
        imdb_score,
        num_voted_users
    ]])

    # Scale
    data = scaler.transform(data)

    # Prediction
    probability = model.predict(data)[0][0]

    if probability >= 0.5:
        st.success("🎉 Prediction: HIT Movie")
    else:
        st.error("❌ Prediction: FLOP Movie")

    st.write(f"Confidence: {probability*100:.2f}%")