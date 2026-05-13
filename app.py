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
        
        # 1. Determine the Crop First
        c_probs = preds[0][0]
        c_idx = np.argmax(c_probs)
        c_conf = c_probs[c_idx]
        predicted_crop = CROP_NAMES[c_idx]

        # 2. APPLY MASKING TO DISEASE PROBABILITIES
        d_probs = preds[1][0].copy() # Work on a copy of the disease probabilities
        
        crop_keywords = {
            'soya': 'soybean',
            'bean': 'bean',
            'maize': 'maize'
        }
        target_keyword = crop_keywords[predicted_crop]

        # Zero out any disease that doesn't belong to the detected crop
        for i, disease_name in enumerate(DISEASE_NAMES):
            if target_keyword not in disease_name.lower():
                d_probs[i] = 0.0  # Masking: Setting the probability to zero

        # 3. Pick the best disease from the REMAINING valid options
        d_idx = np.argmax(d_probs)
        d_conf = d_probs[d_idx]
        predicted_disease = DISEASE_NAMES[d_idx]

    # --- VALIDATION LOGIC ---
    is_valid = True
    error_message = ""

    # Confidence check
    if c_conf < 0.70:
        is_valid = False
        error_message = f"Low confidence ({c_conf:.1%}). Please provide a clearer leaf photo."
    
    # Safety Check: If after masking, the highest disease probability is 0, 
    # it means the model is completely confused.
    if d_conf == 0:
        is_valid = False
        error_message = "The model could not find a matching disease for this crop type."

    # --- DISPLAY ---
    if not is_valid:
        st.error("⚠️ Identification Failed")
        st.info(f"Reason: {error_message}")
    else:
        st.success("Analysis Complete!")
        col1, col2 = st.columns(2)
        col1.metric("Crop", predicted_crop.title(), f"{c_conf:.1%}")
        col2.metric("Condition", predicted_disease, f"{d_conf:.1%}")