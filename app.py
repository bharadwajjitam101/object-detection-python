import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from object_detector import ObjectDetector
import time
import os

# Set environment variable for macOS camera authorization
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

# Set page config
st.set_page_config(
    page_title="Object Detection System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .detection-summary {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Object Detection System")
st.write("Select a mode to start detecting objects")

# Initialize the object detector
@st.cache_resource
def load_detector():
    try:
        return ObjectDetector()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

detector = load_detector()

if detector is None:
    st.error("Failed to initialize the object detector. Please check your internet connection and try again.")
    st.stop()

# Mode selection
mode = st.sidebar.selectbox(
    "Select Detection Mode",
    ["Image Detection", "Video Detection", "Real-time Camera Detection"]
)

def display_detection_summary(summary):
    """Display detection summary in a nice format."""
    if not summary:
        st.warning("No objects detected.")
        return
        
    st.markdown("### Detection Summary")
    st.markdown('<div class="detection-summary">', unsafe_allow_html=True)
    
    # Sort objects by count
    sorted_items = sorted(summary.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for object_name, data in sorted_items:
        count = data['count']
        avg_score = sum(data['scores']) / len(data['scores'])
        st.markdown(f"**{object_name.title()}**")
        st.markdown(f"- Count: {count}")
        st.markdown(f"- Average Confidence: {avg_score:.2%}")
        st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Image Detection Mode
if mode == "Image Detection":
    st.header("Image Object Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            # Display original image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Detect Objects", key="detect_image"):
                with st.spinner("Detecting objects..."):
                    boxes, scores, classes = detector.detect_objects(image)
                    
                    if len(boxes) > 0:
                        result_image = detector.draw_detections(image.copy(), boxes, scores, classes)
                        st.image(result_image, caption="Detected Objects", use_container_width=True)
                        
                        # Get and display detection summary
                        summary = detector.get_detection_summary(boxes, scores, classes)
                        display_detection_summary(summary)
                    else:
                        st.warning("No objects detected in the image.")
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Video Detection Mode
elif mode == "Video Detection":
    st.header("Video Object Detection")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            if st.button("Process Video", key="process_video"):
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                progress_bar = st.progress(0)
                
                # Create a placeholder for detection summary
                summary_placeholder = st.empty()
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = 0
                
                # Dictionary to store all detections
                all_detections = {}
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    boxes, scores, classes = detector.detect_objects(frame)
                    result_frame = detector.draw_detections(frame.copy(), boxes, scores, classes)
                    
                    # Update detection summary
                    frame_summary = detector.get_detection_summary(boxes, scores, classes)
                    for obj_name, data in frame_summary.items():
                        if obj_name in all_detections:
                            all_detections[obj_name]['count'] += data['count']
                            all_detections[obj_name]['scores'].extend(data['scores'])
                        else:
                            all_detections[obj_name] = data.copy()
                    
                    # Display the frame
                    stframe.image(result_frame, channels="BGR", use_container_width=True)
                    
                    # Update progress
                    processed_frames += 1
                    progress = processed_frames / total_frames
                    progress_bar.progress(progress)
                    
                    # Update summary display
                    with summary_placeholder.container():
                        display_detection_summary(all_detections)
                    
                    # Add a small delay to make the video playback smoother
                    time.sleep(0.01)
                
                cap.release()
                st.success("Video processing completed!")
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

# Real-time Camera Detection Mode
else:
    st.header("Real-time Camera Detection")
    
    # Initialize session state for camera control
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    # Start/Stop camera buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Camera", key="start_camera", disabled=st.session_state.camera_active):
            st.session_state.camera_active = True
    
    with col2:
        if st.button("Stop Camera", key="stop_camera", disabled=not st.session_state.camera_active):
            st.session_state.camera_active = False
    
    if st.session_state.camera_active:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to access camera. Please check your camera permissions.")
                st.session_state.camera_active = False
                st.stop()
            
            stframe = st.empty()
            # summary_placeholder = st.empty()  # Remove summary placeholder
            
            # Dictionary to store all detections (no longer needed)
            # all_detections = {}
            
            while cap.isOpened() and st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                # Process frame
                boxes, scores, classes = detector.detect_objects(frame)
                result_frame = detector.draw_detections(frame.copy(), boxes, scores, classes)
                
                # Display the frame
                stframe.image(result_frame, channels="BGR", use_container_width=True)
                
                # No summary display for real-time camera
                
                # Add a small delay to reduce CPU usage
                time.sleep(0.01)
            
            cap.release()
            
        except Exception as e:
            st.error(f"Error accessing camera: {str(e)}")
            st.session_state.camera_active = False 