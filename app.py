import streamlit as st
from model_helper import predict
import os

st.title("Vehicle Damage Detection")

# Upload image
uploaded_file = st.file_uploader("Upload the file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded File", use_container_width=True)

    if st.button("Predict"):
        # Save to temp file
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        prediction = predict(temp_path)  # Pass path to model
        st.success(f"Predicted Class: {prediction}")

        # Optionally delete temp file afterward
        os.remove(temp_path)
