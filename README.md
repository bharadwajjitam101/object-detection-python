# Object Detection System

A real-time object detection system built with Python, Streamlit, and TensorFlow Lite. This application can detect objects in images, videos, and real-time camera feeds.

## Features

- Image object detection
- Video object detection
- Real-time camera detection
- Beautiful and intuitive user interface
- Detection summary with object counts and confidence scores
- Support for multiple input formats (JPG, JPEG, PNG, MP4, AVI)

## Prerequisites

- Use Python 3.11.9
- Webcam (for real-time detection)
- Internet connection (for initial model download)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd object-detection-python
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. The application will open in your default web browser. If it doesn't, navigate to the URL shown in the terminal (typically http://localhost:8501).

3. Select your desired detection mode from the sidebar:
   - Image Detection
   - Video Detection
   - Real-time Camera Detection

### Image Detection
1. Click "Choose an image..." to upload an image file (supported formats: JPG, JPEG, PNG)
2. Click "Detect Objects" to process the image
3. View the results and detection summary

### Video Detection
1. Click "Choose a video..." to upload a video file (supported formats: MP4, AVI)
2. Click "Process Video" to start processing
3. Watch the processed video with detections in real-time
4. View the detection summary as the video plays

### Real-time Camera Detection
1. Click "Start Camera" to begin real-time detection
2. Allow camera access when prompted by your browser
3. View the real-time detection results
4. Click "Stop Camera" when finished

## Troubleshooting

### SSL Certificate Issues
If you encounter SSL certificate errors during model download, you can try one of these solutions:

1. Update your Python packages:
```bash
pip install --upgrade certifi
```

2. Install SSL certificates:

For macOS:
```bash
# Run the certificate installation script
/Applications/Python\ 3.11/Install\ Certificates.command
```

For Windows:
```bash
# Navigate to Python installation directory (adjust path if needed)
cd C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python311\Scripts
# Run the certificate installation script
python Install\ Certificates.command
```

3. Or set the following environment variable before running the application:
```bash
export PYTHONHTTPSVERIFY=0  # On Windows, use: set PYTHONHTTPSVERIFY=0
```

### Camera Access Issues
If you're having trouble accessing the camera:

1. Ensure your webcam is properly connected and not in use by another application
2. Check your browser's camera permissions
3. For macOS users, the application automatically sets the required environment variable for camera authorization

## Project Structure

```
object-detection-python/
├── app.py                 # Main Streamlit application
├── object_detector.py     # Core detection logic
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Dependencies

- streamlit
- opencv-python
- numpy
- Pillow
- tensorflow
- tensorflow-hub

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 