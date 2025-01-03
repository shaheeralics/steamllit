import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageDraw

# Set page configuration
st.set_page_config(
    page_title="Yellow Rust Disease Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the SageMaker endpoint URL
ENDPOINT_URL = "https://<your-endpoint-name>.sagemaker.<region>.amazonaws.com"  # Replace with your endpoint details

# Theme selection
st.sidebar.title("Settings and Preferences")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        "<style>body { background-color: #0e1117; color: white; }</style>",
        unsafe_allow_html=True,
    )

# Title
st.title("Yellow Rust Disease Classification Dashboard")

# User Input Section
st.sidebar.header("Upload Image of Plant Leaf")
image_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if image_file:
    st.subheader("Uploaded Image")
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for SageMaker
    def preprocess_image(img):
        img = img.resize((224, 224))  # Adjust to model input size
        img_array = np.array(img) / 255.0  # Normalize
        return np.expand_dims(img_array, axis=0).tolist()  # Add batch dimension and convert to list

    processed_image = preprocess_image(image)

    # Send request to SageMaker endpoint
    st.write("Sending image to the SageMaker endpoint for prediction...")
    headers = {"Content-Type": "application/json"}
    payload = {"instances": processed_image}  # Adjust payload format based on your model requirements

    try:
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        predictions = response.json()["predictions"][0]  # Extract predictions from the response
        class_names = ["0", "MR", "MRMS", "MS", "R", "S"]  # Replace with your class labels
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display results
        st.header("Prediction Results")
        st.write(f"**Predicted Status:** {predicted_class}")
        st.write(f"**Confidence Level:** {confidence:.2f}%")

        # Severity Level
        severity = "High" if confidence > 80 else "Moderate" if confidence > 50 else "Low"
        st.write(f"**Severity Level:** {severity}")

        # Highlighting a region (Example: bounding box)
        st.subheader("Highlighted Regions")
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 150, 150], outline="red", width=3)  # Example bounding box
        st.image(image, caption="Highlighted Regions", use_column_width=True)

        # Disease Insights
        st.header("Disease Insights")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {predictions[i] * 100:.2f}%")

        # Model Performance Metrics (Admin Only)
        if st.checkbox("Show Model Performance Metrics (Admin Only)"):
            st.write("**Accuracy:** 95.6%")  # Replace with actual metric
            st.write("**Precision:** 94.7%")  # Replace with actual metric
            st.write("**Recall:** 93.5%")  # Replace with actual metric

        # User Notifications
        if severity == "High":
            st.warning("High severity detected! Immediate action is recommended.")

    except Exception as e:
        st.error(f"Error during inference: {e}")

# Footer
st.sidebar.info("Powered by Streamlit and Amazon SageMaker")

