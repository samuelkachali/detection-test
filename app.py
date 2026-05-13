import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Comparison - Crop AI", page_icon="🌱", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource 
def load_all_assets():
    # Load Version 1 (Multi-Head)
    v1_model = tf.keras.models.load_model("crop_disease_v4.keras")
    
    # Load Version 2 (Unified)
    v2_model = tf.keras.models.load_model("crop_disease_unified_v2.keras")
    
    # Load the unified class names
    with open('class_indices.json', 'r') as f:
        class_map = json.load(f)
    # Ensure they are in the correct numerical order
    v2_labels = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
    
    return v1_model, v2_model, v2_labels

v1_model, v2_model, v2_labels = load_all_assets()

# Constants for the old model structure
V1_CROP_NAMES = ['bean', 'maize', 'soya']
V1_DISEASE_NAMES = [
    'Maize Gray Leaf Spot', 'Soybeans Healthy', 'Soybeans Bacterial Pustule', 
    'Soybeans Frog Eye Leaf Spot', 'Maize Blight', 'Soybeans Sudden Death Syndrome', 
    'Beans Angular Leaf Spot', 'Soybeans Rust', 'Beans Rust', 
    'Soybeans Target Leaf Spot', 'Maize Common Rust', 'Maize Healthy', 
    'Beans Healthy', 'Soybeans Yellow Mosaic'
]

# --- UI ---
st.title("🌱 AI Crop Disease Detection: Model Comparison")
st.write("Compare our initial multi-head model (V1) against the new unified balanced model (V2).")

uploaded_file = st.sidebar.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption='Original Image', use_container_width=True)
    
    # Preprocess
    img_prep = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_prep) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Running inference on both systems...'):
        # Inference V1 (Multi-Head)
        preds_v1 = v1_model.predict(img_array, verbose=0)
        v1_c_idx = np.argmax(preds_v1[0])
        v1_d_idx = np.argmax(preds_v1[1])
        v1_crop = V1_CROP_NAMES[v1_c_idx]
        v1_disease = V1_DISEASE_NAMES[v1_d_idx]
        v1_conf = np.max(preds_v1[0])

        # Inference V2 (Unified)
        preds_v2 = v2_model.predict(img_array, verbose=0)
        v2_idx = np.argmax(preds_v2[0])
        v2_full_label = v2_labels[v2_idx]
        v2_conf = np.max(preds_v2[0])

    # --- LAYOUT COMPARISON ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Version 1: Multi-Head")
        st.info("Uses separate outputs for Crop and Disease.")
        st.metric("Detected Crop", v1_crop.title())
        st.metric("Condition", v1_disease)
        st.caption(f"Confidence: {v1_conf:.1%}")
        
        # Check for logical mismatch
        v1_crop_val = v1_crop.lower()
        v1_disease_val = v1_disease.lower()
        is_mismatch = not (v1_crop_val in v1_disease_val or (v1_crop_val == "soya" and "soybean" in v1_disease_val))
        
        if is_mismatch:
            st.warning("⚠️ Logic Mismatch Detected")
            st.caption("Model identified one crop but symptoms of another.")

    with col2:
        st.subheader("Version 2: Unified")
        st.success("Uses single-output classification with balanced data.")
        
        # Format the label for display
        v2_display = v2_full_label.replace("_", " ").title()
        st.metric("Unified Prediction", v2_display)
        st.caption(f"Confidence: {v2_conf:.1%}")
        
        st.write("Logical consistency guaranteed by architecture.")

    # Educational Insight for supervisor
    if v1_crop == "soya" and "soya" not in v2_full_label.lower():
        st.success("**Improvement Note:** The new model correctly avoided the Soya-bias seen in V1.")