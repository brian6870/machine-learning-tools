import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained MNIST model"""
    model_path = 'src/models/mnist_cnn_model.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure 'mnist_cnn_model.h5' exists in the current directory.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image for prediction with better quality"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # First, let's see the original uploaded image
    st.subheader("Image Processing Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Original Image**")
        st.image(image, caption=f'Size: {image.size}', use_container_width=True)
    
    # Step 1: Resize to 28x28 using high-quality resampling
    image_28x28 = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    with col2:
        st.write("**Resized to 28x28**")
        st.image(image_28x28, caption='LANCZOS resampling', use_container_width=True)
    
    # Step 2: Convert to numpy array and normalize
    image_array = np.array(image_28x28) / 255.0
    
    # Step 3: Auto-detect and invert if needed (MNIST expects white digits on black background)
    # If most pixels are bright, invert the image
    if np.mean(image_array) > 0.5:
        image_array = 1.0 - image_array
        st.info("Image automatically inverted (dark digits on light background detected)")
    
    with col3:
        st.write("**Final Processed**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(image_array, cmap='gray')
        ax.set_title('Model Input')
        ax.axis('off')
        st.pyplot(fig)
    
    # Show pixel statistics
    st.write(f"**Pixel Statistics:** Min: {image_array.min():.3f}, Max: {image_array.max():.3f}, Mean: {image_array.mean():.3f}")
    
    # Reshape for model input
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

def enhance_image_quality(image):
    """Optional image enhancement for better results"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    image = ImageOps.autocontrast(image, cutoff=2)
    
    # Optional: Apply slight sharpening
    image = image.filter(ImageFilter.SHARPEN)
    
    return image

def plot_prediction_distribution(predictions):
    """Create a bar chart of prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    digits = range(10)
    probabilities = predictions[0]
    
    # Find the predicted digit
    predicted_digit = np.argmax(probabilities)
    
    # Create colors - highlight the predicted digit
    colors = ['red' if i == predicted_digit else 'skyblue' for i in digits]
    
    bars = ax.bar(digits, probabilities, color=colors, alpha=0.7)
    ax.set_xlabel('Digits')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities (Red = Predicted Digit)')
    ax.set_xticks(digits)
    ax.set_ylim(0, 1)
    
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    return fig

def show_home():
    """Display home page content"""
    st.header("MNIST Handwritten Digit Classification")
    st.write("This application demonstrates a Convolutional Neural Network (CNN) trained on the MNIST dataset for recognizing handwritten digits (0-9).")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Features")
        st.write("- Real-time digit classification")
        st.write("- Confidence scoring")
        st.write("- Interactive visualization")
        st.write("- Model architecture inspection")
        
        st.subheader("How to Use")
        st.write("1. **Upload Image**: Upload a clear image of a handwritten digit")
        st.write("2. **Model Info**: View model architecture and performance details")
        
        st.subheader("Image Requirements")
        st.write("- Clear, dark digit on light background")
        st.write("- Single digit per image")
        st.write("- Minimum size: 50x50 pixels")
        st.write("- Supported formats: PNG, JPG, JPEG")
    
    with col2:
        st.subheader("Model Performance")
        st.metric("Test Accuracy", "98.9%")
        st.metric("Model Type", "CNN")
        st.metric("Training Data", "60,000 images")

def show_upload_image(model):
    """Handle image upload and prediction"""
    st.header("Digit Classification from Image")
    
    uploaded_file = st.file_uploader(
        "Upload an image of a handwritten digit", 
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of a single digit (0-9). Dark digit on light background works best."
    )
    
    if uploaded_file is not None:
        try:
            # Display original image
            original_image = Image.open(uploaded_file)
            
            st.subheader("Uploaded Image")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(original_image, caption=f'Original Size: {original_image.size}', use_container_width=True)
            
            with col2:
                st.write("**Tips for better results:**")
                st.write("- Use dark ink on white background")
                st.write("- Ensure the digit is centered")
                st.write("- Avoid blurry or low-contrast images")
                st.write("- Crop to show only one digit")
            
            # Add optional enhancement
            enhance = st.checkbox("Enhance image quality (recommended for poor quality images)")
            
            if st.button("Classify Digit", type="primary"):
                with st.spinner("Processing image..."):
                    # Apply enhancement if requested
                    if enhance:
                        processed_image = enhance_image_quality(original_image)
                    else:
                        processed_image = original_image
                    
                    # Preprocess image with visualization
                    model_input = preprocess_image(processed_image)
                    
                    if model is not None:
                        prediction = model.predict(model_input, verbose=0)
                        predicted_digit = np.argmax(prediction[0])
                        confidence = np.max(prediction[0])
                        
                        st.subheader("Classification Results")
                        
                        # Display results prominently
                        result_col1, result_col2 = st.columns([1, 2])
                        
                        with result_col1:
                            st.success(f"**Prediction: {predicted_digit}**")
                            st.info(f"**Confidence: {confidence:.2%}**")
                            
                            if confidence > 0.9:
                                st.balloons()
                                st.success("High confidence prediction! 🎉")
                            elif confidence > 0.7:
                                st.warning("Moderate confidence prediction")
                            else:
                                st.error("Low confidence prediction - try a clearer image")
                        
                        with result_col2:
                            # Show probability distribution
                            st.subheader("Prediction Distribution")
                            fig = plot_prediction_distribution(prediction)
                            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def get_layer_output_shape(layer):
    """Safely get output shape from different layer types"""
    try:
        if hasattr(layer, 'output_shape'):
            return layer.output_shape
        elif hasattr(layer, 'output'):
            return layer.output.shape
        else:
            return "N/A"
    except:
        return "N/A"

def show_model_info(model):
    """Display model information and architecture"""
    st.header("Model Information")
    
    if model is not None:
        st.subheader("Model Architecture")
        
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        model_summary = "\n".join(string_list)
        
        with st.expander("View Model Summary"):
            st.text(model_summary)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Architecture")
            st.write("- Type: Convolutional Neural Network")
            st.write("- Input Shape: 28×28×1")
            st.write("- Output: 10 classes (0-9)")
            st.write(f"- Total Layers: {len(model.layers)}")
        
        with col2:
            st.subheader("Training")
            st.write("- Framework: TensorFlow/Keras")
            st.write("- Optimizer: Adam")
            st.write("- Loss Function: Sparse Categorical Crossentropy")
            st.write("- Epochs: 15")
        
        with col3:
            st.subheader("Performance")
            st.metric("Test Accuracy", "98.9%")
            st.metric("Precision", "98.7%")
            st.metric("Recall", "98.8%")
        
        st.subheader("Layer Details")
        layer_info = []
        for i, layer in enumerate(model.layers):
            output_shape = get_layer_output_shape(layer)
            layer_info.append({
                "Layer": i + 1,
                "Name": layer.name,
                "Type": layer.__class__.__name__,
                "Output Shape": str(output_shape)
            })
        
        st.table(layer_info)
        
        st.subheader("Model Parameters")
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", f"{total_params:,}")
        with col2:
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        with col3:
            st.metric("Non-Trainable Parameters", f"{non_trainable_params:,}")
        
    else:
        st.error("Model not available. Please ensure the model file is properly loaded.")

def main():
    """Main application function"""
    st.title("🔢 MNIST Digit Classifier")
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            model = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.model_loaded = True
            else:
                st.error("Failed to load model. Please check the model file.")
                return
    
    model = st.session_state.model
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["Home", "Upload Image", "Model Info"]
    )
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Upload Image":
        show_upload_image(model)
    elif app_mode == "Model Info":
        show_model_info(model)
    
    st.sidebar.markdown("---")
    st.sidebar.write("AI Toolkit")

if __name__ == "__main__":
    plt.style.use('default')
    main()