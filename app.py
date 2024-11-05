import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="centered"
)

# Constants - exactly matching your training configuration
IMAGE_SIZE = 256
CHANNELS = 3
MODEL_PATH = "/Users/architmurgudkar/tensorflow-test/Projects/PotatoDiseaseClassification/Models/3"

# Create preprocessing layers exactly as in training
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)
])

# Load the saved model
@st.cache_resource
def load_model():
    try:
        # Load model with custom_objects if needed
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Class names in the exact order as training
CLASS_NAMES = ['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy']

# Preprocess image using the same preprocessing as training
def preprocess_image(image):
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to array
    image_array = np.array(image)
    
    # Create batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    # Use the same preprocessing as training
    image_array = resize_and_rescale(image_array)
    
    return image_array

def predict(model, image_array):
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0])) * 100
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    st.title("🥔 Potato Disease Classification")
    st.markdown("""
    Upload an image of a potato plant leaf to detect if it's:
    * Healthy
    * Affected by Early Blight
    * Affected by Late Blight
    """)
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable Debug Mode")
    
    # Show model architecture in debug mode
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check the model path and try again.")
        return
    
    if debug_mode:
        st.sidebar.write("Model Architecture:")
        model.summary(print_fn=lambda x: st.sidebar.write(x))
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with st.spinner('Processing image...'):
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            if debug_mode:
                st.write("Debug Information:")
                st.write(f"Original image size: {image.size}")
                st.write(f"Input image shape: {processed_image.shape}")
                st.write(f"Input image dtype: {processed_image.dtype}")
                st.write(f"Input value range: [{processed_image.min()}, {processed_image.max()}]")
                
                # Display preprocessed image
                st.write("Preprocessed Image:")
                st.image(processed_image[0], caption='Preprocessed Image')
            
            # Make prediction
            predicted_class, confidence, raw_predictions = predict(model, processed_image)
            
            if predicted_class and confidence:
                # Display raw predictions in debug mode
                if debug_mode:
                    st.write("Raw prediction scores:")
                    for class_name, score in zip(CLASS_NAMES, raw_predictions):
                        st.write(f"{class_name}: {score:.4f}")
                
                # Create columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Diagnosis")
                    if predicted_class == 'Potato Healthy':
                        st.success(predicted_class)
                    else:
                        st.error(predicted_class)
                
                with col2:
                    st.markdown("### Confidence")
                    st.info(f"{confidence:.2f}%")
                
                # Additional information based on diagnosis
                st.markdown("### Recommendations")
                if predicted_class == 'Potato Early Blight':
                    st.markdown("""
                    * Remove affected leaves to prevent spread
                    * Apply appropriate fungicide
                    * Ensure proper spacing between plants for air circulation
                    * Water at the base of plants to keep leaves dry
                    """)
                elif predicted_class == 'Potato Late Blight':
                    st.markdown("""
                    * Remove and destroy affected plants immediately
                    * Apply copper-based fungicide
                    * Improve drainage in the field
                    * Monitor weather conditions for high humidity
                    """)
                else:
                    st.markdown("""
                    * Continue regular maintenance
                    * Monitor plants regularly
                    * Maintain good agricultural practices
                    """)
                
                if debug_mode:
                    st.markdown("### Model Verification")
                    st.write("To verify the model is loaded correctly, please check:")
                    st.write("1. Model architecture matches training configuration")
                    st.write("2. Class names are in correct order")
                    st.write("3. Preprocessing steps match training exactly")

if __name__ == "__main__":
    main()