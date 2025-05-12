#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import dlib
import sys
import os
import io
from ultralytics import YOLO

# Setup for encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add path to GazeTracking GitHub repository
gaze_tracking_path = "GazeTracking"
sys.path.append(gaze_tracking_path)

try:
    from gaze_tracking import GazeTracking
except ImportError:
    print("GazeTracking module not found.")
    print("Please clone https://github.com/antoinelame/GazeTracking")
    sys.exit(1)

class CheatingDetector:
    def __init__(self):
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize gaze tracking
        self.gaze = GazeTracking()
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO("yolov8n.pt")  # Use small model
            self.yolo_enabled = True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            print("Continuing without object detection.")
            self.yolo_enabled = False
        
        # Cheating related object class IDs (based on COCO dataset)
        self.cheating_objects = {
            67: "Cell Phone",
            73: "Book",
            63: "Laptop",
            74: "Mouse",
            62: "Monitor",
            66: "Remote",
            28: "Person"
        }
        
        # Initialize Dlib face detector and landmark predictor
        try:
            self.detector = dlib.get_frontal_face_detector()
            landmark_path = os.path.join(gaze_tracking_path, "trained_models", "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(landmark_path):
                print(f"Landmark model not found: {landmark_path}")
                print("Please check if shape_predictor_68_face_landmarks.dat file exists in the trained_models folder of GazeTracking repository.")
                raise FileNotFoundError(f"Landmark file not found: {landmark_path}")
            
            self.predictor = dlib.shape_predictor(landmark_path)
            self.head_pose_enabled = True
        except Exception as e:
            print(f"Failed to initialize head direction estimation: {e}")
            print("Continuing without head direction estimation.")
            self.head_pose_enabled = False
        
        # 3D model points (for head direction estimation)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Camera matrix
        self.camera_matrix = np.array(
            [[650, 0, 320],
             [0, 650, 240],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Camera distortion coefficients
        self.dist_coeffs = np.zeros((4, 1))
        
        # Initialize cheating counters
        self.gaze_cheating_counter = 0
        self.head_cheating_counter = 0
        self.last_warning_time = 0
        
        # Define screen ROI (Region of Interest)
        self.roi_left = 100
        self.roi_top = 100
        self.roi_right = 540
        self.roi_bottom = 380
        
        # No face detected counter
        self.no_face_counter = 0
        
        # Frame processing skip counter (for performance)
        self.frame_skip = 0
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Save cheating events
        self.cheating_events = []
        self.last_event_time = time.time()
    
    def get_head_pose(self, shape):
        """Estimate head direction from Dlib landmarks"""
        # Extract specific points from landmarks
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),     # Nose tip
            (shape.part(8).x, shape.part(8).y),       # Chin
            (shape.part(36).x, shape.part(36).y),     # Left eye left corner
            (shape.part(45).x, shape.part(45).y),     # Right eye right corner
            (shape.part(48).x, shape.part(48).y),     # Left mouth corner
            (shape.part(54).x, shape.part(54).y)      # Right mouth corner
        ], dtype=np.float64)
        
        # Calculate rotation vector and translation vector
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        
        # Rotation matrix and Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            cv2.hconcat((pose_mat, np.array([[0, 0, 0, 1]]).T))
        )
        
        return euler_angles
    
    def add_cheating_event(self, event_type, details=""):
        """Record cheating event"""
        current_time = time.time()
        # Record events with minimum 2 second interval
        if current_time - self.last_event_time > 2:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            event = {
                "Time": timestamp,
                "Type": event_type,
                "Details": details
            }
            self.cheating_events.append(event)
            self.last_event_time = current_time
            
            # Delete old events if too many
            if len(self.cheating_events) > 20:
                self.cheating_events.pop(0)
    
    def detect_cheating(self, frame):
        """Detect cheating"""
        original_frame = frame.copy()
        
        # Gaze tracking analysis
        self.gaze.refresh(frame)
        
        is_cheating = False
        warning_text = ""
        
        # 1. Check face detection status
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            self.no_face_counter += 1
            if self.no_face_counter > 30:  # If face not detected for about 1 second
                is_cheating = True
                warning_text += "Face not detected! "
                self.add_cheating_event("No Face", "Failed to detect face on screen")
                
                # Display face detection failure message
                cv2.putText(frame, "Face not detected!", (50, 150), 
                          self.font, 1.0, (0, 0, 255), 2)
        else:
            self.no_face_counter = 0
        
        # 2. Gaze tracking check
        gaze_text = "Analyzing gaze..."
        gaze_color = (147, 58, 31)  # Default color
        
        if self.gaze.is_blinking():
            gaze_text = "Blinking"
            gaze_color = (0, 255, 0)  # Normal state - green
        elif self.gaze.is_right():
            gaze_text = "Looking Right"
            gaze_color = (0, 165, 255)  # Warning state - orange
            self.gaze_cheating_counter += 1
        elif self.gaze.is_left():
            gaze_text = "Looking Left"
            gaze_color = (0, 165, 255)  # Warning state - orange
            self.gaze_cheating_counter += 1
        elif self.gaze.is_center():
            gaze_text = "Looking Center"
            gaze_color = (0, 255, 0)  # Normal status - green
            # Decrease counter gradually
            self.gaze_cheating_counter = max(0, self.gaze_cheating_counter - 1)
        
        # Display probability information (more accurate feedback)
        right_prob = self.gaze.horizontal_ratio()
        if right_prob is not None:
            gaze_text += f" ({100 - int(right_prob * 100)}% | {int(right_prob * 100)}%)"
        
        # Display detected gaze state
        cv2.putText(frame, "Gaze: " + gaze_text, (20, 60), self.font, 0.7, gaze_color, 2)
        
        # Consider cheating if gaze off-center for more than 3 seconds (90 frames)
        if self.gaze_cheating_counter >= 90:
            is_cheating = True
            warning_text += "Gaze off-center! "
            self.add_cheating_event("Suspicious Gaze", gaze_text)
        
        # 3. Dlib face landmarks and head direction estimation (if enabled)
        if self.head_pose_enabled and len(faces) > 0:
            try:
                shape = self.predictor(gray, faces[0])
                
                # Draw landmarks (for debugging and visualization)
                for n in range(68):
                    x = shape.part(n).x
                    y = shape.part(n).y
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Estimate head direction
                euler_angles = self.get_head_pose(shape)
                pitch, yaw, roll = [float(angle) for angle in euler_angles]
                
                # Display head direction information
                pitch_text = f"Pitch: {pitch:.1f}°"
                yaw_text = f"Yaw: {yaw:.1f}°"
                roll_text = f"Roll: {roll:.1f}°"
                
                # Set color based on angle
                pitch_color = (0, 255, 0) if abs(pitch) < 25 else (0, 0, 255)
                yaw_color = (0, 255, 0) if abs(yaw) < 25 else (0, 0, 255)
                roll_color = (0, 255, 0) if abs(roll) < 15 else (0, 0, 255)
                
                cv2.putText(frame, pitch_text, (20, 100), self.font, 0.6, pitch_color, 2)
                cv2.putText(frame, yaw_text, (20, 130), self.font, 0.6, yaw_color, 2)
                cv2.putText(frame, roll_text, (20, 160), self.font, 0.6, roll_color, 2)
                
                # Visualize head direction vector
                # Nose tip coordinates
                nose_tip = (shape.part(30).x, shape.part(30).y)
                
                # Calculate and draw yaw direction vector (blue)
                yaw_length = 50  # Vector length
                yaw_dx = yaw_length * np.sin(np.radians(yaw))
                yaw_vector = (int(nose_tip[0] + yaw_dx), nose_tip[1])
                cv2.line(frame, nose_tip, yaw_vector, (255, 0, 0), 2)
                
                # Increase cheating counter if head yaw angle exceeds ±25 degrees
                if abs(yaw) > 25:
                    self.head_cheating_counter += 1
                else:
                    # Decrease counter if in normal range (gradually)
                    self.head_cheating_counter = max(0, self.head_cheating_counter - 1)
                
                # Consider cheating if head direction abnormal for more than 2 seconds (60 frames)
                if self.head_cheating_counter >= 60:
                    is_cheating = True
                    warning_text += f"Abnormal head direction (Yaw: {yaw:.1f}°)! "
                    self.add_cheating_event("Head Direction", f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°")
            
            except Exception as e:
                cv2.putText(frame, "Head direction estimation error", (20, 200), 
                          self.font, 0.7, (0, 0, 255), 1)
                print(f"Error during head direction estimation: {e}")
        
        # 4. YOLOv8 object detection (with frame-based skipping for performance)
        if self.yolo_enabled and self.frame_skip == 0:
            try:
                # Run YOLO on original frame (improve small object detection accuracy)
                yolo_results = self.yolo_model(original_frame, conf=0.45)  # Adjust confidence threshold
                
                for r in yolo_results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        
                        if cls_id in self.cheating_objects and conf > 0.45:
                            obj_name = self.cheating_objects[cls_id]
                            
                            # Handle person class (detect additional people outside screen)
                            if cls_id == 28:  # Person
                                # Only consider as cheating if not near main face
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                if len(faces) > 0:
                                    face = faces[0]
                                    face_center_x = (face.left() + face.right()) // 2
                                    face_center_y = (face.top() + face.bottom()) // 2
                                    box_center_x = (x1 + x2) // 2
                                    box_center_y = (y1 + y2) // 2
                                    
                                    # Calculate distance from main face
                                    distance = np.sqrt((face_center_x - box_center_x)**2 + 
                                                     (face_center_y - box_center_y)**2)
                                    
                                    # Consider different person if distance is far enough
                                    if distance < 100:  # Threshold (pixels)
                                        continue  # Ignore if close to main face
                            
                            is_cheating = True
                            warning_text += f"{obj_name} detected! "
                            self.add_cheating_event("Object Detected", obj_name)
                            
                            # Draw box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{obj_name}: {conf:.2f}", 
                                      (x1, y1 - 10), self.font, 0.5, (0, 0, 255), 2)
                
                # Set next skip frame count (run every 3 frames)
                self.frame_skip = 3
            except Exception as e:
                cv2.putText(frame, "Object detection error", (20, 220), self.font, 0.7, (0, 0, 255), 1)
                print(f"Error during YOLO processing: {e}")
        else:
            # Decrease skip counter
            self.frame_skip = max(0, self.frame_skip - 1)
        
        # 5. Define and display screen ROI
        cv2.rectangle(frame, (self.roi_left, self.roi_top), 
                    (self.roi_right, self.roi_bottom), (0, 165, 255), 2)

        # 6. Display counters (enhanced visual feedback)
        gaze_bar_length = int((self.gaze_cheating_counter / 90) * 100)
        head_bar_length = int((self.head_cheating_counter / 60) * 100)
        
        # Progress bar background
        cv2.rectangle(frame, (400, 30), (400 + 100, 40), (100, 100, 100), -1)
        cv2.rectangle(frame, (400, 50), (400 + 100, 60), (100, 100, 100), -1)
        
        # Progress bar fill
        gaze_color = (0, 165, 255) if gaze_bar_length < 80 else (0, 0, 255)
        head_color = (0, 165, 255) if head_bar_length < 80 else (0, 0, 255)
        
        cv2.rectangle(frame, (400, 30), (400 + min(gaze_bar_length, 100), 40), gaze_color, -1)
        cv2.rectangle(frame, (400, 50), (400 + min(head_bar_length, 100), 60), head_color, -1)
        
        # Labels
        cv2.putText(frame, "Gaze", (350, 38), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Head", (350, 58), self.font, 0.5, (255, 255, 255), 1)
        
        # 7. Display warning for detected cheating
        if is_cheating:
            current_time = time.time()
            if current_time - self.last_warning_time > 2:  # Console warning every 2 seconds
                print("Possible cheating detected!", warning_text)
                self.last_warning_time = current_time
                
            # Display warning message on screen (noticeable)
            cv2.putText(frame, "Cheating Detected!", (20, 30), self.font, 1.0, (0, 0, 255), 2)
            
            # Draw warning box
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 400), (630, 470), (0, 0, 150), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # Semi-transparent effect
            
            # Warning text
            wrapped_warning = self._wrap_text(warning_text, 80)
            y_offset = 420
            for line in wrapped_warning:
                cv2.putText(frame, line, (20, y_offset), self.font, 0.6, (255, 255, 255), 1)
                y_offset += 20
        
        # 8. Display cheating records (recent event history at bottom of screen)
        if len(self.cheating_events) > 0 and not is_cheating:  # Only show when no current warning
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 400), (630, 470), (100, 100, 100), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # Semi-transparent effect
            
            # Title
            cv2.putText(frame, "Recent Events:", (20, 420), self.font, 0.7, (255, 255, 255), 1)
            
            # Show only last 3 events
            y_offset = 440
            for event in self.cheating_events[-3:]:
                event_text = f"{event['Time']} - {event['Type']}: {event['Details']}"
                cv2.putText(frame, event_text, (30, y_offset), self.font, 0.5, (200, 200, 200), 1)
                y_offset += 20
        
        # 9. Guidance message
        cv2.putText(frame, "Q: Quit", (560, 20), self.font, 0.5, (200, 200, 200), 1)
        
        # Add annotations for preprocessed frame, head direction estimation and eye positions
        frame = self.gaze.annotated_frame()
        
        return frame
    
    def _wrap_text(self, text, max_width):
        """Wrap text to specified maximum width"""
        words = text.split(' ')
        lines = []
        current_line = ''
        
        for word in words:
            test_line = current_line + ' ' + word if current_line else word
            if len(test_line) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def run(self):
        """Main execution loop"""
        print("Starting cheating detection system...")
        print("Press 'q' to quit.")
        
        # Variables for FPS measurement
        prev_frame_time = 0
        new_frame_time = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot read frame from webcam.")
                break
                
            # Mirror frame (for more intuitive user experience)
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            
            # Detect cheating
            try:
                frame = self.detect_cheating(frame)
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps}", (560, 40), self.font, 0.5, (200, 200, 200), 1)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Display error information
                cv2.putText(frame, f"Processing error: {str(e)[:50]}", (10, 30), 
                          self.font, 0.5, (0, 0, 255), 1)
            
            # Display frame
            cv2.imshow("Online Exam Cheating Detection", frame)
            
            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        print("Cheating detection system terminated.")

if __name__ == "__main__":
    try:
        detector = CheatingDetector()
        detector.run()
    except Exception as e:
        print(f"Error occurred: {e}")
