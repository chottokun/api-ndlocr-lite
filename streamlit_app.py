import streamlit as st
import httpx
import time
import json
import io

st.set_page_config(
    page_title="NDLOCR-Lite Test UI",
    page_icon="📖",
    layout="wide"
)

st.title("📖 NDLOCR-Lite Test UI")
st.markdown("Simple interface to test the NDLOCR-Lite API.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_base_url = st.text_input("API Base URL", value="http://localhost:8001")
    
    st.divider()
    st.markdown("### API Health")
    if st.button("Check Health"):
        try:
            response = httpx.get(f"{api_base_url}/health")
            if response.status_code == 200:
                health = response.json()
                st.success(f"Status: {health.get('status')}")
                if health.get('engine_ready'):
                    st.info("Engine: Ready")
                else:
                    st.warning("Engine: Not Initialized")
            else:
                st.error(f"Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

# Main Tabs
tab1, tab2 = st.tabs(["Synchronous OCR", "Asynchronous Jobs"])

def display_ocr_result(result):
    if "pages" in result:
        for i, page in enumerate(result["pages"]):
            st.subheader(f"Page {i+1}")
            st.markdown(page.get("markdown", ""))
            
            with st.expander(f"View Raw JSON (Page {i+1})"):
                st.json(page)
    else:
        st.json(result)

# Tab 1: Synchronous OCR
with tab1:
    st.header("Synchronous OCR")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="sync_upload")
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        st.caption(f"File: {uploaded_file.name} / Size: {len(file_bytes)} bytes / Type: {uploaded_file.type}")
        
        try:
            st.image(file_bytes, caption="Uploaded Image", width="stretch")
        except Exception as e:
            st.warning(f"Image preview failed: {e}")
        
        if st.button("Run OCR", key="run_sync"):
            with st.spinner("Processing..."):
                try:
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    response = httpx.post(f"{api_base_url}/v1/ocr", files=files, timeout=120.0)
                    
                    if response.status_code == 200:
                        st.success("OCR Completed")
                        display_ocr_result(response.json())
                    else:
                        st.error(f"API Error: {response.status_code}")
                        try:
                            st.json(response.json())
                        except Exception:
                            st.code(response.text)
                except Exception as e:
                    st.error(f"Request failed: {e}")

# Tab 2: Asynchronous Jobs
with tab2:
    st.header("Asynchronous Jobs")
    job_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="job_upload")
    
    if job_file is not None:
        job_bytes = job_file.getvalue()
        st.caption(f"File: {job_file.name} / Size: {len(job_bytes)} bytes / Type: {job_file.type}")
        
        try:
            st.image(job_bytes, caption="Uploaded Image", width="stretch")
        except Exception as e:
            st.warning(f"Image preview failed: {e}")
        
        if st.button("Create Job", key="create_job"):
            with st.spinner("Creating job..."):
                try:
                    files = {"file": (job_file.name, job_bytes, job_file.type)}
                    response = httpx.post(f"{api_base_url}/v1/ocr/jobs", files=files, timeout=120.0)
                    
                    if response.status_code == 200:
                        job_info = response.json()
                        job_id = job_info["job_id"]
                        st.info(f"Job Created: {job_id}")
                        
                        # Polling logic
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        max_attempts = 120 # 2 minutes
                        for attempt in range(max_attempts):
                            status_response = httpx.get(f"{api_base_url}/v1/ocr/jobs/{job_id}")
                            if status_response.status_code == 200:
                                job_data = status_response.json()
                                status = job_data["status"]
                                status_placeholder.write(f"Current Status: **{status}**")
                                
                                if status == "completed":
                                    progress_bar.progress(100)
                                    st.success("Job Completed!")
                                    display_ocr_result(job_data["result"])
                                    break
                                elif status == "failed":
                                    st.error(f"Job Failed: {job_data.get('error')}")
                                    break
                                
                                progress_bar.progress(min((attempt + 1) * 2, 95))
                            else:
                                st.error(f"Failed to fetch job status: {status_response.status_code}")
                                break
                            
                            time.sleep(1)
                        else:
                            st.warning("Polling timed out.")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        try:
                            st.json(response.json())
                        except Exception:
                            st.code(response.text)
                except Exception as e:
                    st.error(f"Request failed: {e}")
