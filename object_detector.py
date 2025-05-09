import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import ssl
import certifi

class ObjectDetector:
    def __init__(self):
        # Fix SSL certificate issues
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Load the TFLite model
        try:
            # Using a more accurate model
            self.model = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        # COCO dataset class names
        self.class_names = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
            39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
            43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
            49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
            54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
            59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
            64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
            73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
            78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
            84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
            89: 'hair drier', 90: 'toothbrush'
        }
        
    def get_object_name(self, class_id):
        return self.class_names.get(class_id, f"Unknown object {class_id}")
        
    def preprocess_image(self, image):
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'numpy'):
            image = image.numpy()
        
        # Ensure image is in RGB format
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                # Check if it's BGR
                if isinstance(image, np.ndarray):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Resize image if it's too large
        max_dimension = 640
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
        
    def detect_objects(self, image):
        try:
            # Preprocess the image
            image = self.preprocess_image(image)
            
            # Convert image to tensor
            input_tensor = tf.convert_to_tensor(image)
            input_tensor = input_tensor[tf.newaxis, ...]
            
            # Perform detection
            detections = self.model(input_tensor)
            
            # Process detections
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(np.int32)
            
            # Lower confidence threshold for better detection
            confidence_threshold = 0.3  # Lowered from 0.5
            
            # Filter detections based on confidence threshold
            mask = scores >= confidence_threshold
            
            # Apply non-maximum suppression to remove overlapping boxes
            selected_indices = tf.image.non_max_suppression(
                boxes, scores, max_output_size=50, iou_threshold=0.5, score_threshold=confidence_threshold
            ).numpy()
            
            boxes = boxes[selected_indices]
            scores = scores[selected_indices]
            classes = classes[selected_indices]
            
            return boxes, scores, classes
            
        except Exception as e:
            print(f"Error in detect_objects: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def draw_detections(self, image, boxes, scores, classes):
        try:
            # Make a copy of the image
            image = image.copy()
            
            # Convert to RGB if it's not already
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            height, width = image.shape[:2]
            
            for box, score, class_id in zip(boxes, scores, classes):
                # Convert normalized coordinates to pixel coordinates
                ymin, xmin, ymax, xmax = box
                xmin = int(xmin * width)
                xmax = int(xmax * width)
                ymin = int(ymin * height)
                ymax = int(ymax * height)
                
                # Draw bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Get object name and create label
                object_name = self.get_object_name(class_id)
                label = f'{object_name}: {score:.2f}'
                
                # Add label with background for better visibility
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (xmin, ymin - label_height - 10), (xmin + label_width, ymin), (0, 255, 0), -1)
                cv2.putText(image, label, (xmin, ymin - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            return image
            
        except Exception as e:
            print(f"Error in draw_detections: {e}")
            return image
            
    def get_detection_summary(self, boxes, scores, classes):
        """Returns a summary of detected objects with their counts and confidence scores."""
        summary = {}
        for box, score, class_id in zip(boxes, scores, classes):
            object_name = self.get_object_name(class_id)
            if object_name in summary:
                summary[object_name]['count'] += 1
                summary[object_name]['scores'].append(score)
            else:
                summary[object_name] = {
                    'count': 1,
                    'scores': [score]
                }
        return summary 