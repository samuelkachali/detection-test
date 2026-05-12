import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱")

# --- LOAD MODEL ---
@st.cache_resource # Caches model so it doesn't reload every time you click
def load_my_model():
    return tf.keras.models.load_model("crop_disease_v4.keras")

model = load_my_model()

# --- LABELS (Match your Kaggle IDs exactly!) ---
CROP_NAMES = ['bean', 'maize', 'soya'] # Add all 3 here
DISEASE_NAMES = ['Maize Gray Leaf Spot', 'Soybeans Healthy', 'Soybeans Bacterial Pustule', 'Soybeans Frog Eye Leaf Spot', 'Maize Blight', 'Soybeans Sudden Death Syndrome', 'Beans Angular Leaf Spot', 'Soybeans Rust', 'Beans Rust', 'Soybeans Target Leaf Spot', 'Maize Common Rust', 'Maize Healthy', 'Beans Healthy', 'Soybeans Yellow Mosaic'] # Add all 14 here

# --- UI DESIGN ---
st.title("🌱 AI Crop Disease Detection")
st.markdown("Upload a leaf photo to identify the crop and detect potential diseases.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    # 2. Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Predict
    with st.spinner('Analyzing patterns...'):
        preds = model.predict(img_array)
        
        c_idx = np.argmax(preds[0])
        d_idx = np.argmax(preds[1])
        c_conf = np.max(preds[0])
        d_conf = np.max(preds[1])

    # 4. Show Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Crop", CROP_NAMES[c_idx], f"{c_conf:.1%}")
    with col2:
        st.metric("Condition", DISEASE_NAMES[d_idx], f"{d_conf:.1%}")

    if d_conf < 0.5:
        st.warning("Low confidence: The symptoms are unclear. Please try a clearer photo.")