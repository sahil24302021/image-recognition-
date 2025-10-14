import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- MODEL LOADING & PREDICTION FUNCTIONS ---

# Use a cache to load the model only once
@st.cache_resource
def load_model():
    """Loads the pre-trained MobileNetV2 model."""
    # Load a pre-trained model from Keras
    # It's trained on the ImageNet dataset, which has 1000 object categories.
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    """Prepares the image for the model."""
    # MobileNetV2 expects images of size 224x224
    image = image.resize((224, 224))
    # Convert image to a numpy array
    image_array = np.array(image)
    # Add a 'batch' dimension for the model
    image_array = np.expand_dims(image_array, axis=0)
    # Preprocess the image for MobileNetV2 (scales pixel values)
    processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return processed_image

def predict(image):
    """Makes a prediction using the loaded model."""
    model = load_model()
    processed_image = preprocess_image(image)
    # Get model's prediction
    prediction = model.predict(processed_image)
    # Decode the prediction into human-readable labels
    decoded_prediction = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=3)[0]
    return decoded_prediction

# --- STREAMLIT WEB APP INTERFACE ---

# Set up the title and a small description
st.title("🖼️ Image Recognition App")
st.write("Upload an image, and the AI will predict what it is!")

# Create the file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # If a file is uploaded, open it as an image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("🧠 **Thinking...**")

    # Perform prediction
    predictions = predict(image)

    # Display the results
    st.subheader("🤖 Here's what I think it is:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i+1}. **{label.replace('_', ' ').title()}** with {score:.2%} confidence.")