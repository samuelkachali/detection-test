import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱")

# --- LOAD MODEL ---
@st.cache_resource 
def load_my_model():
    return tf.keras.models.load_model("crop_disease_v4.keras")

model = load_my_model()

# --- LABELS ---
CROP_NAMES = ['bean', 'maize', 'soya']

DISEASE_NAMES = [
    'Maize Gray Leaf Spot', 'Soybeans Healthy', 'Soybeans Bacterial Pustule', 
    'Soybeans Frog Eye Leaf Spot', 'Maize Blight', 'Soybeans Sudden Death Syndrome', 
    'Beans Angular Leaf Spot', 'Soybeans Rust', 'Beans Rust', 
    'Soybeans Target Leaf Spot', 'Maize Common Rust', 'Maize Healthy', 
    'Beans Healthy', 'Soybeans Yellow Mosaic'
]

# --- UI DESIGN ---
st.title("🌱 AI Crop Disease Detection - v1.0")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analyzing...'):
        preds = model.predict(img_array)
        
        c_idx = np.argmax(preds[0])
        d_idx = np.argmax(preds[1])
        
        c_conf = np.max(preds[0])
        d_conf = np.max(preds[1])

        predicted_crop = CROP_NAMES[c_idx]
        predicted_disease = DISEASE_NAMES[d_idx]

    # --- REVISED VALIDATION LOGIC ---
    is_valid = True
    error_message = ""

    # 1. Lowered Threshold (70% is safer for a baseline model)
    if c_conf < 0.70:
        is_valid = False
        error_message = f"Low confidence ({c_conf:.1%}). Please provide a clearer leaf photo."

    # 2. Flexible Semantic Check
    # This checks if the first word of the disease matches the crop
    crop_val = predicted_crop.lower() # 'bean', 'maize', or 'soya'
    disease_val = predicted_disease.lower() # e.g., 'soybeans healthy'

    # Mapping common variations
    if crop_val == 'soya' and 'soybean' in disease_val:
        pass # This is a match
    elif crop_val == 'bean' and 'bean' in disease_val:
        pass # This is a match
    elif crop_val == 'maize' and 'maize' in disease_val:
        pass # This is a match
    else:
        is_valid = False
        error_message = f"Mismatch: Predicted {crop_val} but symptoms look like {predicted_disease}."

    # --- DISPLAY ---
    if not is_valid:
        st.error("⚠️ Identification Failed")
        st.info(f"Reason: {error_message}")
    else:
        st.success("Analysis Complete!")
        col1, col2 = st.columns(2)
        col1.metric("Crop", predicted_crop.title(), f"{c_conf:.1%}")
        col2.metric("Condition", predicted_disease, f"{d_conf:.1%}")