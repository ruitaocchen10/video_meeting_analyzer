import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
import time
from collections import deque

@dataclass
class PostureMetrics:
    """Data class to store posture analysis metrics"""
    shoulder_angle: float
    head_tilt: float
    forward_lean: float
    is_slouching: bool
    is_tilted: bool
    is_leaning: bool
    # New metrics
    head_motion_score: float
    hand_motion_score: float
    eye_contact_maintained: bool
    eye_contact_duration: float
    is_head_moving: bool
    is_hand_fidgeting: bool
    timestamp: float
    issues: List[str]

class PostureDetector:
    """Detects and analyzes posture using MediaPipe Pose"""
    
    def __init__(self):
        # Initialize MediaPipe Pose and Hands
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Initialize face mesh for eye tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Thresholds for posture issues
        self.FORWARD_LEAN_THRESHOLD = 0.15
        self.HEAD_TILT_MIN = 165
        self.HEAD_TILT_MAX = 195
        self.SHOULDER_ANGLE_MIN = 165
        self.SHOULDER_ANGLE_MAX = 195
        
        # New thresholds
        self.HEAD_MOTION_THRESHOLD = 15.0  # Pixels per frame
        self.HAND_MOTION_THRESHOLD = 20.0  # Pixels per frame
        self.EYE_CONTACT_DURATION_THRESHOLD = 2.0  # Seconds
        #self.EYE_CENTER_THRESHOLD = 0.15  # Ratio threshold for eye centering

        self.LEFT_EYE_MIN = 2.75
        self.LEFT_EYE_MAX = 2.85
        self.RIGHT_EYE_MIN = -1.95
        self.RIGHT_EYE_MAX = -1.875
        
        # For smoothing measurements
        self.metric_history = {
            'shoulder_angle': [],
            'head_tilt': [],
            'forward_lean': []
        }
        self.history_size = 5
        
        # Head motion tracking
        self.head_position_history = deque(maxlen=10)
        self.head_motion_buffer = deque(maxlen=30)  # Track last 30 frames (~1 sec)
        
        # Hand motion tracking
        self.hand_positions_history = deque(maxlen=10)
        self.hand_motion_buffer = deque(maxlen=30)
        
        # Eye contact tracking
        self.eye_center_history = deque(maxlen=10)
        self.looking_away_start_time = None
        self.total_looking_away_time = 0.0
        
        # For debugging iris positions
        self.last_left_iris = None
        self.last_right_iris = None
        
    def calculate_angle(self, point1: tuple, point2: tuple, point3: tuple) -> float:
        """Calculate angle between three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_slope_angle(self, point1: tuple, point2: tuple) -> float:
        """Calculate the angle of a line from horizontal"""
        x1, y1 = point1
        x2, y2 = point2
        
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return abs(angle)
    
    def smooth_metric(self, metric_name: str, value: float) -> float:
        """Apply moving average smoothing to reduce jitter"""
        self.metric_history[metric_name].append(value)
        
        if len(self.metric_history[metric_name]) > self.history_size:
            self.metric_history[metric_name].pop(0)
        
        return np.mean(self.metric_history[metric_name])
    
    def calculate_head_motion(self, nose_coords: tuple) -> float:
        """Calculate head movement across frames"""
        self.head_position_history.append(nose_coords)
        
        if len(self.head_position_history) < 2:
            return 0.0
        
        # Calculate displacement from previous frame
        prev_pos = self.head_position_history[-2]
        curr_pos = self.head_position_history[-1]
        
        displacement = np.sqrt(
            (curr_pos[0] - prev_pos[0])**2 + 
            (curr_pos[1] - prev_pos[1])**2
        )
        
        self.head_motion_buffer.append(displacement)
        
        # Average motion over buffer period
        avg_motion = np.mean(list(self.head_motion_buffer))
        return avg_motion
    
    def calculate_hand_motion(self, hand_landmarks_list) -> float:
        """Calculate hand movement and fidgeting"""
        if not hand_landmarks_list:
            self.hand_positions_history.append(None)
            return 0.0
        
        # Calculate center point of all detected hands
        hand_centers = []
        for hand_landmarks in hand_landmarks_list:
            # Use wrist as reference point
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            hand_centers.append((wrist.x, wrist.y))
        
        # Average position of all hands
        avg_hand_pos = np.mean(hand_centers, axis=0) if hand_centers else None
        
        self.hand_positions_history.append(avg_hand_pos)
        
        if len(self.hand_positions_history) < 2 or avg_hand_pos is None:
            return 0.0
        
        # Calculate displacement
        if self.hand_positions_history[-2] is not None:
            prev_pos = self.hand_positions_history[-2]
            curr_pos = avg_hand_pos
            
            # Convert to pixel displacement (assuming 1280x720)
            displacement = np.sqrt(
                ((curr_pos[0] - prev_pos[0]) * 1280)**2 + 
                ((curr_pos[1] - prev_pos[1]) * 720)**2
            )
            
            self.hand_motion_buffer.append(displacement)
        
        # Average motion over buffer period
        avg_motion = np.mean(list(self.hand_motion_buffer)) if self.hand_motion_buffer else 0.0
        return avg_motion
    
    def check_eye_contact(self, face_landmarks, image_shape) -> tuple:
        """
        Check if eyes are looking at camera
        Returns: (is_looking, duration_looking_away, left_iris_relative, right_iris_relative)
        """
        if not face_landmarks:
            return True, 0.0, None, None
        
        h, w = image_shape[:2]
        
        # Key iris landmarks (left and right iris centers)
        # MediaPipe face mesh iris indices: 468-473 (right), 473-478 (left)
        left_iris_indices = [474, 475, 476, 477]
        right_iris_indices = [469, 470, 471, 472]
        
        # Get iris centers
        left_iris_x = np.mean([face_landmarks.landmark[i].x for i in left_iris_indices])
        left_iris_y = np.mean([face_landmarks.landmark[i].y for i in left_iris_indices])
        
        right_iris_x = np.mean([face_landmarks.landmark[i].x for i in right_iris_indices])
        right_iris_y = np.mean([face_landmarks.landmark[i].y for i in right_iris_indices])
        
        # Get eye corners to calculate relative position
        # Left eye: 33 (outer), 133 (inner)
        # Right eye: 362 (outer), 263 (inner)
        left_eye_left = face_landmarks.landmark[33].x
        left_eye_right = face_landmarks.landmark[133].x
        right_eye_left = face_landmarks.landmark[263].x
        right_eye_right = face_landmarks.landmark[362].x
        
        # Calculate relative iris position (0 = outer corner, 1 = inner corner)
        left_eye_width = abs(left_eye_right - left_eye_left)
        right_eye_width = abs(right_eye_right - right_eye_left)
        
        left_iris_relative = (left_iris_x - left_eye_left) / left_eye_width if left_eye_width > 0 else 0.5
        right_iris_relative = (right_iris_x - right_eye_right) / right_eye_width if right_eye_width > 0 else 0.5
        
        # Eyes should be centered (around 0.5) when looking at camera
        # Allow some threshold for natural variation
        left_centered = self.LEFT_EYE_MIN < left_iris_relative < self.LEFT_EYE_MAX
        right_centered = self.RIGHT_EYE_MIN < right_iris_relative < self.RIGHT_EYE_MAX
        
        is_looking = left_centered and right_centered
        
        # Track duration of looking away
        current_time = time.time()
        
        if not is_looking:
            if self.looking_away_start_time is None:
                self.looking_away_start_time = current_time
            duration_away = current_time - self.looking_away_start_time
        else:
            self.looking_away_start_time = None
            duration_away = 0.0
        
        return is_looking, duration_away, left_iris_relative, right_iris_relative
    
    def analyze_posture(self, pose_landmarks, hand_landmarks_list, face_landmarks, 
                       image_shape) -> Optional[PostureMetrics]:
        """Analyze posture from MediaPipe landmarks"""
        if not pose_landmarks:
            return None
        
        h, w = image_shape[:2]
        
        # Extract key landmarks
        nose = pose_landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        
        # Convert to pixel coordinates
        nose_coords = (nose.x * w, nose.y * h)
        left_shoulder_coords = (left_shoulder.x * w, left_shoulder.y * h)
        right_shoulder_coords = (right_shoulder.x * w, right_shoulder.y * h)
        left_ear_coords = (left_ear.x * w, left_ear.y * h)
        right_ear_coords = (right_ear.x * w, right_ear.y * h)
        
        # Calculate midpoints
        shoulder_midpoint = (
            (left_shoulder_coords[0] + right_shoulder_coords[0]) / 2,
            (left_shoulder_coords[1] + right_shoulder_coords[1]) / 2
        )
        
        # Original posture metrics
        shoulder_angle_raw = self.calculate_slope_angle(left_shoulder_coords, right_shoulder_coords)
        shoulder_angle = self.smooth_metric('shoulder_angle', shoulder_angle_raw)
        
        head_tilt_raw = self.calculate_slope_angle(left_ear_coords, right_ear_coords)
        head_tilt = self.smooth_metric('head_tilt', head_tilt_raw)
        
        vertical_distance = abs(nose_coords[1] - shoulder_midpoint[1])
        horizontal_offset = abs(nose_coords[0] - shoulder_midpoint[0])
        forward_lean_raw = horizontal_offset / max(vertical_distance, 1)
        forward_lean = self.smooth_metric('forward_lean', forward_lean_raw)
        
        # New metrics
        head_motion = self.calculate_head_motion(nose_coords)
        hand_motion = self.calculate_hand_motion(hand_landmarks_list)
        is_looking, time_looking_away, left_iris_rel, right_iris_rel = self.check_eye_contact(face_landmarks, image_shape)
        
        # Store iris values for debugging display
        self.last_left_iris = left_iris_rel
        self.last_right_iris = right_iris_rel
        
        # Detect issues
        issues = []
        is_slouching = False
        is_tilted = False
        is_leaning = False
        is_head_moving = False
        is_hand_fidgeting = False
        eye_contact_maintained = True
        
        if shoulder_angle > self.SHOULDER_ANGLE_MAX or shoulder_angle < self.SHOULDER_ANGLE_MIN:
            is_slouching = True
            issues.append("Shoulders Not Level")
        
        if head_tilt > self.HEAD_TILT_MAX or head_tilt < self.HEAD_TILT_MIN:
            is_tilted = True
            issues.append("Head Tilted")
        
        if forward_lean > self.FORWARD_LEAN_THRESHOLD:
            is_leaning = True
            issues.append("Leaning Forward")
        
        if head_motion > self.HEAD_MOTION_THRESHOLD:
            is_head_moving = True
            issues.append("Excessive Head Movement")
        
        if hand_motion > self.HAND_MOTION_THRESHOLD:
            is_hand_fidgeting = True
            issues.append("Hand Fidgeting")
        
        if time_looking_away > self.EYE_CONTACT_DURATION_THRESHOLD:
            eye_contact_maintained = False
            issues.append("Missing Eye Contact")
        
        return PostureMetrics(
            shoulder_angle=shoulder_angle,
            head_tilt=head_tilt,
            forward_lean=forward_lean,
            is_slouching=is_slouching,
            is_tilted=is_tilted,
            is_leaning=is_leaning,
            head_motion_score=head_motion,
            hand_motion_score=hand_motion,
            eye_contact_maintained=eye_contact_maintained,
            eye_contact_duration=time_looking_away,
            is_head_moving=is_head_moving,
            is_hand_fidgeting=is_hand_fidgeting,
            timestamp=time.time(),
            issues=issues
        )
    
    def draw_posture_info(self, image, metrics: PostureMetrics):
        """Draw posture information on the image"""
        if not metrics:
            return image
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = 30
        
        # Determine primary issue for status message
        if not metrics.issues:
            status = "Good Posture"
            status_color = (0, 255, 0)
        else:
            # Prioritize issues
            if "Missing Eye Contact" in metrics.issues:
                status = "Missing Eye Contact"
            elif "Excessive Head Movement" in metrics.issues:
                status = "Head Movement"
            elif "Hand Fidgeting" in metrics.issues:
                status = "Hand Fidgeting"
            elif "Leaning Forward" in metrics.issues:
                status = "Leaning"
            elif "Shoulders Not Level" in metrics.issues:
                status = "Slouching"
            elif "Head Tilted" in metrics.issues:
                status = "Head Tilted"
            else:
                status = metrics.issues[0]
            status_color = (0, 0, 255)
        
        # Draw background rectangle
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (450, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Draw status
        cv2.putText(image, status, (20, y_offset), font, font_scale, status_color, thickness)
        
        # Draw metrics
        y_offset += 30
        cv2.putText(image, f"Shoulder: {metrics.shoulder_angle:.1f}° | Head Tilt: {metrics.head_tilt:.1f}°", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(image, f"Forward Lean: {metrics.forward_lean:.2f}", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        head_motion_color = (0, 255, 0) if not metrics.is_head_moving else (0, 165, 255)
        cv2.putText(image, f"Head Motion: {metrics.head_motion_score:.1f} px/frame", 
                   (20, y_offset), font, 0.5, head_motion_color, 1)
        
        y_offset += 25
        hand_motion_color = (0, 255, 0) if not metrics.is_hand_fidgeting else (0, 165, 255)
        cv2.putText(image, f"Hand Motion: {metrics.hand_motion_score:.1f} px/frame", 
                   (20, y_offset), font, 0.5, hand_motion_color, 1)
        
        y_offset += 25
        eye_color = (0, 255, 0) if metrics.eye_contact_maintained else (0, 0, 255)
        eye_status = "Maintained" if metrics.eye_contact_maintained else f"Away {metrics.eye_contact_duration:.1f}s"
        cv2.putText(image, f"Eye Contact: {eye_status}", 
                   (20, y_offset), font, 0.5, eye_color, 1)
        
        # Draw iris position debug info
        y_offset += 25
        if self.last_left_iris is not None and self.last_right_iris is not None:
            cv2.putText(image, f"L Iris: {self.last_left_iris:.3f} | R Iris: {self.last_right_iris:.3f}", 
                       (20, y_offset), font, 0.45, (255, 200, 0), 1)
            y_offset += 20
            threshold_info = f"Left: {self.LEFT_EYE_MIN:.2f} to {self.LEFT_EYE_MAX:.2f} \n Right: {self.RIGHT_EYE_MIN:.2f} to {self.RIGHT_EYE_MAX:.2f}"
            cv2.putText(image, threshold_info, 
                       (20, y_offset), font, 0.4, (200, 200, 200), 1)
        
        # Draw additional issues
        if len(metrics.issues) > 1:
            y_offset += 30
            cv2.putText(image, f"Other Issues: {len(metrics.issues) - 1}", 
                       (20, y_offset), font, 0.4, (255, 255, 0), 1)
        
        return image
    
    def process_frame(self, frame) -> tuple:
        """Process a single frame and return annotated frame with metrics"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        
        # Process hands
        hands_results = self.hands.process(rgb_frame)
        
        # Process face mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Analyze all metrics
        metrics = None
        if pose_results.pose_landmarks:
            metrics = self.analyze_posture(
                pose_results.pose_landmarks.landmark,
                hands_results.multi_hand_landmarks if hands_results.multi_hand_landmarks else [],
                face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None,
                frame.shape
            )
            
            frame = self.draw_posture_info(frame, metrics)
        
        return frame, metrics
    
    def release(self):
        """Clean up resources"""
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()


def main():
    """Main function to run the posture detection system"""
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = PostureDetector()
    
    print("Starting interview posture detector...")
    print("Press 'q' to quit")
    print("Press 's' to save current metrics")
    
    session_metrics = []
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            annotated_frame, metrics = detector.process_frame(frame)
            
            if metrics:
                session_metrics.append(metrics)
            
            cv2.imshow('Interview Posture Detector', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and metrics:
                print(f"\n--- Current Metrics ---")
                print(f"Shoulder Angle: {metrics.shoulder_angle:.2f}°")
                print(f"Head Tilt: {metrics.head_tilt:.2f}°")
                print(f"Forward Lean: {metrics.forward_lean:.2f}")
                print(f"Head Motion: {metrics.head_motion_score:.2f} px/frame")
                print(f"Hand Motion: {metrics.hand_motion_score:.2f} px/frame")
                print(f"Eye Contact: {'Maintained' if metrics.eye_contact_maintained else f'Away for {metrics.eye_contact_duration:.1f}s'}")
                print(f"Issues: {', '.join(metrics.issues) if metrics.issues else 'None'}")
                print("----------------------\n")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        if session_metrics:
            print("\n=== Session Summary ===")
            print(f"Total frames analyzed: {len(session_metrics)}")
            
            issues_detected = sum(1 for m in session_metrics if m.issues)
            print(f"Frames with issues: {issues_detected} ({issues_detected/len(session_metrics)*100:.1f}%)")
            
            head_moving = sum(1 for m in session_metrics if m.is_head_moving)
            hand_fidgeting = sum(1 for m in session_metrics if m.is_hand_fidgeting)
            eye_contact_lost = sum(1 for m in session_metrics if not m.eye_contact_maintained)
            
            print(f"\nIssue Breakdown:")
            print(f"  Head Movement: {head_moving} frames ({head_moving/len(session_metrics)*100:.1f}%)")
            print(f"  Hand Fidgeting: {hand_fidgeting} frames ({hand_fidgeting/len(session_metrics)*100:.1f}%)")
            print(f"  Eye Contact Lost: {eye_contact_lost} frames ({eye_contact_lost/len(session_metrics)*100:.1f}%)")
            
            print(f"\nAverage Metrics:")
            print(f"  Shoulder Angle: {np.mean([m.shoulder_angle for m in session_metrics]):.2f}°")
            print(f"  Head Tilt: {np.mean([m.head_tilt for m in session_metrics]):.2f}°")
            print(f"  Forward Lean: {np.mean([m.forward_lean for m in session_metrics]):.2f}")
            print(f"  Head Motion: {np.mean([m.head_motion_score for m in session_metrics]):.2f} px/frame")
            print(f"  Hand Motion: {np.mean([m.hand_motion_score for m in session_metrics]):.2f} px/frame")
            print("======================\n")


if __name__ == "__main__":
    main()