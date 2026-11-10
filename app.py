import streamlit as st
import torch
import numpy as np
import cv2
from pathlib import Path
import json

from mri_quality_final import UNet, QualityNet, MRIPipeline, load_demo_image

st.set_page_config(page_title="MRI Quality Assessment & Enhancement", layout="wide")

st.title("Group 06 — Cranial MRI Quality Assessment & Enhancement")
st.markdown("Upload a brain MRI slice (PNG, JPG, or NIfTI) to enhance image quality and estimate quality metrics.")

@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Folder where your trained weights (enhancer_best.pt, quality_best.pt) will be located
    model_folder = Path("./runs/model_weights") 
    try:
        pipeline = MRIPipeline.from_folder(model_folder, device=device)
        return pipeline
    except Exception as e:
        return None

pipeline = load_pipeline()

uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["png", "jpg", "jpeg", "tif", "nii", "nii.gz"])

if uploaded_file is not None:
    temp_path = Path("temp_uploaded_file.nii.gz" if uploaded_file.name.endswith(".gz") else "temp_uploaded_file.png")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        image_size = 256  # Standard resolution from your project config
        normalized_img = load_demo_image(temp_path, image_size)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input (Degraded/Original)")
            st.image(normalized_img, caption="Input MRI Slice", use_column_width=True, clamp=True, channels="GRAY")
            
        if pipeline is not None:
            with st.spinner("Running U-Net enhancement & QualityNet estimation..."):
                enhanced, report = pipeline.process(normalized_img)
                
            with col2:
                st.subheader("Enhanced Output (U-Net)")
                st.image(enhanced, caption="Enhanced MRI Slice", use_column_width=True, clamp=True, channels="GRAY")
                
            st.markdown("### Quality & Performance Report")
            st.json(report)
        else:
            st.warning("Model weights (`enhancer_best.pt` and `quality_best.pt`) are missing. Please place your trained `.pt` files inside a folder named `runs/model_weights/` in your repository.")
            
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")