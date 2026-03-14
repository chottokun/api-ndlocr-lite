import streamlit as st
import httpx
import time
import json
import io
import os
import base64
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional

# --- Configuration & Styling ---
st.set_page_config(
    page_title="NDLOCR-Lite Dashboard",
    page_icon="📖",
    layout="wide"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 1.5rem;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .metric-value {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-ready { background-color: #dcfce7; color: #166534; }
    .status-error { background-color: #fee2e2; color: #991b1b; }
    .status-warning { background-color: #fef9c3; color: #854d0e; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Constants & State ---
SAMPLE_DIR = "extern/ndlocr-lite/resource"
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# --- Helper Functions ---

def draw_bounding_boxes(image: Image.Image, ocr_data: Dict[str, Any], show_labels: bool = True) -> Image.Image:
    """Draws OCR bounding boxes on the image."""
    draw_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    for page in ocr_data.get("pages", []):
        for line in page.get("lines", []):
            box = line.get("boundingBox")
            if not box or len(box) < 2: continue
            
            # Flatten box if needed [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
            poly = [coord for pt in box for coord in pt]
            draw.polygon(poly, outline="#2563eb", width=3)
            
            if show_labels:
                text = f"{line.get('confidence', 0):.2f}"
                # Get text size using textbbox
                bbox = draw.textbbox((box[0][0], box[0][1] - 20), text, font=font)
                draw.rectangle(bbox, fill="#2563eb")
                draw.text((box[0][0], box[0][1] - 20), text, fill="white", font=font)
                
    return draw_img

def run_ocr(api_url: str, file_bytes: bytes, filename: str, mime_type: str) -> Optional[Dict[str, Any]]:
    """Calls the synchronous OCR API."""
    try:
        files = {"file": (filename, file_bytes, mime_type)}
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{api_url}/v1/ocr", files=files)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
    return None

def check_health(api_url: str) -> Dict[str, Any]:
    """Checks API health."""
    try:
        with httpx.Client() as client:
            response = client.get(f"{api_url}/health")
            if response.status_code == 200:
                return response.json()
    except:
        pass
    return {"status": "offline", "engine_ready": False}

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ndl-lab/ndlocr-lite/main/ndlocr-lite-gui/ndl-lab-logo.png", width=150)
    st.header("NDLOCR-Lite API")
    
    api_base_url = st.text_input("API Base URL", value=DEFAULT_API_URL)
    
    health = check_health(api_base_url)
    if health["status"] == "ok":
        if health.get("engine_ready"):
            st.markdown('<span class="status-badge status-ready">● Engine Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-warning">● Engine Starting</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-error">● API Offline</span>', unsafe_allow_html=True)

    st.divider()
    
    st.subheader("📁 Sample Gallery")
    if os.path.exists(SAMPLE_DIR):
        sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(('.jpg', '.png')) and not f.startswith('viz_')]
        selected_sample = st.selectbox("Select a sample image", ["None"] + sample_files)
        if selected_sample != "None" and st.button("Load Sample"):
            with open(os.path.join(SAMPLE_DIR, selected_sample), "rb") as f:
                st.session_state.uploaded_file = {
                    "bytes": f.read(),
                    "name": selected_sample,
                    "type": "image/jpeg"
                }
                st.rerun()
    else:
        st.caption("Sample directory not found.")

    st.divider()
    
    if st.session_state.history:
        st.subheader("🕒 History")
        for i, h in enumerate(reversed(st.session_state.history)):
            if st.button(f"{h['name']} ({h['time']})", key=f"hist_{i}"):
                st.session_state.current_result = h
                st.rerun()

# --- Main UI ---
st.markdown('<h1 class="main-header">📖 NDLOCR-Lite Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">High-performance Japanese OCR testing environment.</p>', unsafe_allow_html=True)

upload_tab, result_tab, raw_tab = st.tabs(["📤 Upload", "🔍 Analysis", "📄 Raw Data"])

with upload_tab:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.session_state.uploaded_file = {
                "bytes": uploaded_file.getvalue(),
                "name": uploaded_file.name,
                "type": uploaded_file.type
            }

        if "uploaded_file" in st.session_state:
            up = st.session_state.uploaded_file
            st.image(up["bytes"], caption=f"Selected: {up['name']}", use_container_width=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Process Settings")
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        show_labels = st.checkbox("Show Confidence Labels", value=True)
        
        if "uploaded_file" in st.session_state:
            if st.button("🚀 Run OCR", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    start_time = time.time()
                    up = st.session_state.uploaded_file
                    result = run_ocr(api_base_url, up["bytes"], up["name"], up["type"])
                    
                    if result:
                        elapsed = time.time() - start_time
                        
                        # Calculate avg confidence
                        confs = []
                        for p in result.get("pages", []):
                            for l in p.get("lines", []):
                                confs.append(l.get("confidence", 0))
                        
                        avg_conf = sum(confs) / len(confs) if confs else 0
                        
                        history_entry = {
                            "name": up["name"],
                            "time": time.strftime("%H:%M:%S"),
                            "elapsed": f"{elapsed:.2f}s",
                            "confidence": f"{avg_conf:.2%}",
                            "result": result,
                            "original_image": up["bytes"]
                        }
                        
                        st.session_state.current_result = history_entry
                        st.session_state.history.append(history_entry)
                        if len(st.session_state.history) > 5:
                            st.session_state.history.pop(0)
                        
                        st.success("Analysis complete!")
                        st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)

with result_tab:
    if st.session_state.current_result:
        curr = st.session_state.current_result
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Source File", curr["name"])
        with m2:
            st.metric("Avg Confidence", curr["confidence"])
        with m3:
            st.metric("Processing Time", curr["elapsed"])
            
        st.divider()
        
        view_col, text_col = st.columns(2)
        
        with view_col:
            st.subheader("Visual Result")
            img = Image.open(io.BytesIO(curr["original_image"]))
            if show_boxes:
                viz_img = draw_bounding_boxes(img, curr["result"], show_labels)
                st.image(viz_img, use_container_width=True)
            else:
                st.image(img, use_container_width=True)
                
        with text_col:
            st.subheader("Extracted Content")
            markdown_text = "\n\n".join([p.get("markdown", "") for p in curr["result"].get("pages", [])])
            st.markdown(markdown_text)
            
            st.divider()
            st.subheader("Export")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📥 Download Markdown", markdown_text, file_name=f"ocr_{curr['name']}.md", mime="text/markdown")
            with col_d2:
                st.download_button("📥 Download JSON", json.dumps(curr["result"], ensure_ascii=False, indent=2), file_name=f"ocr_{curr['name']}.json", mime="application/json")
    else:
        st.info("Run an OCR process to see results here.")

with raw_tab:
    if st.session_state.current_result:
        st.json(st.session_state.current_result["result"])
    else:
        st.info("No data available.")

# --- Footer ---
st.divider()
st.caption("NDLOCR-Lite Test UI | Powered by FastAPI & Streamlit")

