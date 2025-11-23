import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SpeakerDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LIP_LANDMARKS = [61, 291, 39, 181, 0, 17]

    def detect_speaker(self, frame: np.ndarray, audio_detected: bool = False) -> str:
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided for speaker detection")
            return "Not Speaking (No face detected or invalid frame)"
        
        logger.debug(f"Frame shape: {frame.shape}")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Already BGR from app.py
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return "Not Speaking (No face detected)"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        upper_lip_idx = 13
        lower_lip_idx = 14
        left_face_idx = 234
        right_face_idx = 454

        upper_lip = landmarks[upper_lip_idx].y * h
        lower_lip = landmarks[lower_lip_idx].y * h
        left_face = landmarks[left_face_idx].x * w
        right_face = landmarks[right_face_idx].x * w

        vertical_dist = abs(upper_lip - lower_lip)
        horizontal_ref = abs(left_face - right_face)
        mouth_openness = vertical_dist / horizontal_ref if horizontal_ref > 0 else 0

        is_speaking_lip = mouth_openness > 0.05
        is_speaking = is_speaking_lip or audio_detected

        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        speaker_info = f"Speaking: {'Yes' if is_speaking else 'No'}"
        if is_speaking_lip:
            speaker_info += " (Lip Movement)"
        if audio_detected:
            speaker_info += " (Audio)"
        speaker_info += f" | Face BBox: {bbox}"

        return speaker_info