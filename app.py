import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Set wide layout so your PACS workstation layout fits nicely
st.set_page_config(
    page_title="Group 06 | Cranial MRI IQA Workstation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide default Streamlit header/footer elements for a clean desktop look
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding: 0 !important; max-width: 100% !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load and render your custom index.html UI
try:
    html_path = Path("index.html")
    if html_path.exists():
        html_code = html_path.read_text(encoding="utf-8")
        # Render the custom UI inside an iframe with full width and viewport height
        components.html(html_code, height=920, scrolling=True)
    else:
        st.error("Error: 'index.html' not found in the project directory. Please upload it.")
except Exception as e:
    st.error(f"Error loading UI: {e}")