import cv2 # type: ignore # OpenCV for image handling
import numpy as np # type: ignore # Numerical operations
import streamlit as st # type: ignore # Web app framework
from tensorflow.keras.applications.mobilenet_v2 import ( # type: ignore
    MobileNetV2,
    preprocess_input,
    decode_predictions
) # Not training from scratch, using a pre-trained model (MobileNetV2)
from PIL import Image # type: ignore # For image processing


def load_model():
    model = MobileNetV2(weights='imagenet') # Convolutional Neural Network pre-trained on ImageNet (Weights are like learned values)
    return model

def preprocess_image(image):
    img = np.array(image) # Convert image to numpy array
    img = cv2.resize(img, (224, 224)) # Resize image to 224x224 pixels
    img = preprocess_input(img) # Preprocess image for MobileNet
    img = np.expand_dims(img, axis=0) # Add batch (10, 20, 30 images at a time) dimension
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image) # Preprocess the image
        predictions = model.predict(processed_image) # Get predictions from the model
        decoded_predictions = decode_predictions(predictions, top=3)[0] # Decode top 3 numeric predictions and convert into string labels
        
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered") # Set page configuration for Streamlit app
    st.write("Upload an image and let AI tell you what is in it!")

    @st.cache_resource # Cache the model to avoid reloading on every interaction
    def load_cached_model(): # Load and cache the model
        return load_model() # Load the pre-trained model
    
    model = load_cached_model() # Load the cached model

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"]) # File uploader for images

    if uploaded_file is not None: # If an image is uploaded
        image = st.image(
            uploaded_file, caption="Uploaded Image.", use_container_width=True # Display the uploaded image
        )
        btn = st.button("Classify Image") # Button to classify the image
        if btn:
            with st.spinner("Analyzing Image..."): # Show spinner while processing
                image = Image.open(uploaded_file) # Open the uploaded image
                predictions = classify_image(model, image) # Classify the image

                if predictions:
                    st.subheader("Predictions") # Display predictions
                    for _, label, score in predictions: # Loop through predictions
                        st.write(f"**{label}**:{score:.2%}") # Show label and confidence score

if __name__ == "__main__":
    main()
