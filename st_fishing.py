import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("MobileNet_fish_finetuned.h5")

# Use the same class labels from your notebook
class_labels = np.load("class_labels.npy", allow_pickle=True).tolist()

# Confidence threshold for "Unknown"
THRESHOLD = 60  

st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image to predict its category")


# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):

        # Preprocess (resize as per your model input)
        img_resized = img.resize((224, 224))  
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        preds = model.predict(img_array)[0]
        predicted_idx = np.argmax(preds)
        predicted_label = class_labels[predicted_idx]
        confidence = preds[predicted_idx] * 100

        # Threshold check
        if confidence < THRESHOLD:
            predicted_label = "❌ Unknown / Not a Fish"
            st.error(f"This does not look like a fish from the trained classes. (Confidence {confidence:.2f}%)")
        else:
            predicted_label = class_labels[predicted_idx]
            st.subheader(f"✅ Predicted Category: {predicted_label}")
            st.write(f"**Confidence:** {confidence:.2f}%")

       # Show confidence scores for all classes
        st.write("### Confidence Scores for Each Class")
        st.bar_chart({label: float(score) for label, score in zip(class_labels, preds)})
