import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Load the custom YOLOv10 model
@st.cache_resource
def load_model():
    model_path = "fine_tuned_yolov10.pt"  # Path to the uploaded model
    try:
        model = YOLO(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Function to process the image and get predictions
def get_predictions(model, image):
    with torch.no_grad():
        results = model(image)  # Directly pass the PIL image; YOLO handles preprocessing
    return results[0]  # Get the first Results object from the list

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, results):
    draw = ImageDraw.Draw(image)

    # Extract boxes, confidence, and class IDs
    boxes = results.boxes  # This is an ultralytics.engine.results.Boxes object

    if boxes is not None and len(boxes) > 0:
        for box in boxes.data:
            xmin, ymin, xmax, ymax = box[:4]
            confidence = box[4].item()
            class_id = int(box[5].item())
            label = f"{results.names[class_id]}: {confidence:.2f}"

            # Draw the bounding box
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
            draw.text((xmin, ymin - 10), label, fill="red")
    else:
        st.write("No objects detected.")

    return image

# Streamlit UI
st.title("YOLOv10 Image Detection")

# Upload image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None and model:
    # Convert the file to a PIL image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get predictions
    st.write("Processing Image...")
    try:
        results = get_predictions(model, image)

        # Draw bounding boxes
        processed_image = draw_bounding_boxes(image.copy(), results)

        # Display the image with bounding boxes
        st.image(processed_image, caption="Detected Image", use_column_width=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.write("Upload an image to get predictions.")


