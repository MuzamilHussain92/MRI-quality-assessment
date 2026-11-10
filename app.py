import streamlit as st
import torch
import numpy as np
import cv2
from pathlib import Path
import json
import matplotlib.cm as cm

# Import model pipeline functions from your backend module
from mri_quality_final import (
    UNet, QualityNet, MRIPipeline, load_demo_image, true_metrics
)

# Page Configuration - Dark PACS Workstation Theme
st.set_page_config(
    page_title="Group 06 | Cranial MRI Workstation",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection for professional dark styling
st.markdown("""
    <style>
    .main { background-color: #060911; color: #F8FAFC; }
    .stApp { background-color: #060911; }
    .css-12oz5g7 { padding-top: 1rem; }
    .card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.12);
        padding: 1.25rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title { font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 600; }
    .metric-value { font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #FFFFFF; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Path to your uploaded model weights
    model_folder = Path("./runs/model_weights")
    try:
        pipeline = MRIPipeline.from_folder(model_folder, device=device)
        return pipeline, device
    except Exception as e:
        return None, device

pipeline, device = load_pipeline()

# --- TOP HEADER BAR ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("### 🧠 CRANIAL MRI QUALITY & ENHANCEMENT WORKSTATION")
    st.caption("Automated No-Reference PSNR/SSIM Regression & Residual U-Net Artifact Restoration (Group 06)")

with col_head2:
    st.markdown("Status: <span style='color:#10B981; font-weight:bold;'>● PIPELINE READY</span>", unsafe_allow_html=True)

st.markdown("---")

# --- SIDEBAR: CONTROLS & TELEMETRY ---
with st.sidebar:
    st.header("PACS Controls")
    
    uploaded_file = st.file_uploader(
        "Upload Cranial MRI Slice",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "nii", "nii.gz"]
    )
    
    st.markdown("---")
    st.subheader("Radiological Windowing")
    brightness = st.slider("Brightness (Level)", 50, 160, 100, 5)
    contrast = st.slider("Contrast (Width)", 50, 180, 100, 5)
    
    st.markdown("---")
    st.markdown("**Model Specs:**")
    st.text("• Architecture: Residual U-Net\n• IQA Engine: QualityNet\n• Dataset: OpenNeuro ds000221")

# --- MAIN WORKSTATION VIEWPORT ---
if uploaded_file is not None:
    # Save temporary file securely to disk for processing
    temp_path = Path("temp_upload_" + uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        image_size = 256  # Standard project resolution
        normalized_img = load_demo_image(temp_path, image_size)
        
        if pipeline is not None:
            with st.spinner("Running Residual U-Net Enhancement & QualityNet Estimation..."):
                enhanced, report = pipeline.process(normalized_img)
                
            # Apply dynamic windowing adjustment filters for display
            display_in = np.clip((normalized_img * (contrast / 100.0)) * (brightness / 100.0), 0, 1)
            display_out = np.clip((enhanced * (contrast / 100.0)) * (brightness / 100.0), 0, 1)
            
            # --- TABS FOR VIEW MODES ---
            tab_side, tab_diff = st.tabs(["Side-by-Side Comparison", "Magma Error Difference Map"])
            
            with tab_side:
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown("**Uploaded Input MRI**")
                    st.image(display_in, use_column_width=True, clamp=True, channels="GRAY")
                with col_img2:
                    st.markdown("**Enhanced Output (U-Net)**")
                    st.image(display_out, use_column_width=True, clamp=True, channels="GRAY")
                    
            with tab_diff:
                st.markdown("**Absolute Residual Correction Map (|Enhanced − Input|)**")
                diff_map = np.clip(np.abs(enhanced - normalized_img) / 0.25, 0, 1)
                magma_colored = (cm.get_cmap("magma")(diff_map)[..., :3]).astype(np.float32)
                st.image(magma_colored, use_column_width=True, clamp=True)

            st.markdown("---")
            
            # --- METRICS & QUALITYNET CLASSIFICATION PANELS ---
            m_col1, m_col2, m_col3 = st.columns(3)
            
            bef_est = report["before_estimated"]
            aft_est = report["after_estimated"]
            
            with m_col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Quality Band Classification</div>', unsafe_allow_html=True)
                st.markdown(f"Input: **{bef_est['quality_label']}** → Enhanced: **<span style='color:#10B981;'>{aft_est['quality_label']}</span>**", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with m_col2:
                ssim_gain = aft_est['estimated_ssim'] - bef_est['estimated_ssim']
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Estimated SSIM</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{aft_est["estimated_ssim"]:.3f} <span style="font-size:0.9rem; color:#38BDF8;">(+{ssim_gain:.3f})</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with m_col3:
                psnr_gain = aft_est['estimated_psnr_db'] - bef_est['estimated_psnr_db']
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Estimated PSNR</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{aft_est["estimated_psnr_db"]:.2f} dB <span style="font-size:0.9rem; color:#38BDF8;">(+{psnr_gain:.2f} dB)</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
                
        else:
            st.error("Model weights checkpoint folder (`runs/model_weights/`) could not be loaded. Please ensure weight files are properly uploaded.")
            
    except Exception as e:
        st.error(f"Error processing uploaded MRI: {e}")
else:
    # --- EMPTY STATE ---
    st.info("👈 Please upload a cranial MRI image or scan file using the sidebar panel to begin diagnostic assessment.")