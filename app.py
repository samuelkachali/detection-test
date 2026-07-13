import os
import gc
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import Image
import streamlit as st
from core import load_all_assets, predict, format_disease_name, format_label_name, resolve_consensus_labels, V1_CROP_NAMES, V1_DISEASE_NAMES

st.set_page_config(page_title="Model Comparison Pipeline - Crop AI", page_icon="🌱", layout="wide")

st.title("🌱 AI Crop Disease Detection: Pipeline Comparison Engine")
st.write("Cross-framework analysis: Standard CNN Classifier Layers vs Semantic Vision Foundations vs Object Detection.")

with st.spinner("Loading models..."):
    models = load_all_assets()

uploaded_file = st.sidebar.file_uploader("Upload leaf image for system evaluation", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.sidebar.image(image, caption="Original Input Image", width="stretch")
    with st.spinner("Running pipeline..."):
        result = predict(image, models)
    if not result.get("valid"):
        st.error(f"❌ **Invalid Domain:** {result.get('reason')}")
        st.stop()
    st.subheader("🔎 Detection Results")
    st.success("The pipeline completed successfully and produced the following predictions.")
    crop_name = format_crop_name(result["v1_crop"])
    disease_name = format_disease_name(result["v2_full_label"])
    yolo_name = format_label_name(result["yolo_label"])
    col1, col2, col3 = st.columns(3)
    col1.metric("YOLO Class", yolo_name, f"{result['yolo_confidence']:.0%}")
    col2.metric("Consensus Crop", result["consensus_crop"], f"{result['v1_confidence']:.0%}")
    col3.metric("Consensus Disease", result["consensus_disease"], f"{result['v2_confidence']:.0%}")
    st.write(f"**Primary crop:** {result['consensus_crop']}")
    st.write(f"**Primary disease:** {result['consensus_disease']}")
    st.write(f"**YOLO anchor confidence:** {result['yolo_confidence']:.2%}")
    st.caption("YOLO is being used as the master anchor for the final displayed crop and disease labels.")
    gc.collect()
