"""
Live Webcam Indian Sign Language (ISL) Emergency Gesture Recognition.
Features:
- Dual-hand skeletal tracking (Left & Right hands, 21 landmarks each, 126 features)
- High-accuracy Bidirectional LSTM model with hand-swapping sagittal symmetry
- Left-Hand & Right-Hand Independence: works seamlessly regardless of signing hand
- Multi-Tier Unknown Gesture Rejection Engine:
    1. Negative 'unknown' class filtering
    2. Softmax confidence thresholding (>= 0.70)
    3. Prediction margin thresholding (>= 0.25, ambiguous gesture rejection)
    4. Latent class prototype distance gating (Out-Of-Distribution detection)
    5. Motion velocity gating (distinguishes static resting hands from active signing)
    6. Temporal prediction consensus (stabilizes continuous stream predictions)
- Prominent output display BELOW the interface in a dedicated dashboard canvas
- Synchronized terminal output with timestamps, status, and candidate breakdown
- Dual recognition modes:
    1. Spacebar Trigger / Record Mode (press SPACE to record an exact 2-second gesture window)
    2. Continuous Live Streaming (with dynamic consensus & OOD rejection)
- Interactive hotkeys: [SPACE] Record sign, [F] Invert hands toggle, [C] Clear result, [Q] Quit

Run with:
    python live.py
"""

import os
import sys
import time
import json
import urllib.request
from collections import deque
from pathlib import Path

# Configure backend seamlessly
if "KERAS_BACKEND" not in os.environ:
    try:
        import tensorflow
        os.environ["KERAS_BACKEND"] = "tensorflow"
    except ImportError:
        os.environ["KERAS_BACKEND"] = "torch"

import keras
import cv2
import numpy as np
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parent
SEQUENCE_LENGTH = 30
FEATURE_DIM = 126

# Standard MediaPipe Hand Landmark Connections (21 points)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base & knuckles
    (5, 9), (9, 13), (13, 17)
]


class LiveISLRecognizer:
    def __init__(
        self,
        model_path: Path = None,
        label_map_path: Path = None,
        prototypes_path: Path = None,
        task_model_path: Path = None,
        confidence_threshold: float = 0.70,
        margin_threshold: float = 0.25,
        smoothing_window: int = 6
    ):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.smoothing_window = smoothing_window
        self.invert_hands = False  # Toggle with [F] if camera driver already pre-mirrors
        self.bottom_panel_height = 165  # Height of dedicated bottom output interface

        # Terminal logging state
        self.last_terminal_word = None
        self.last_terminal_time = 0.0
        self.terminal_interval = 1.6  # seconds heartbeat for continuous logging

        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "isl_lstm_model.keras"
        if label_map_path is None:
            label_map_path = PROJECT_ROOT / "models" / "label_map.json"
        if prototypes_path is None:
            prototypes_path = PROJECT_ROOT / "models" / "class_prototypes.json"
        if task_model_path is None:
            task_model_path = PROJECT_ROOT / "hand_landmarker.task"

        # 1. Load trained Model
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path.resolve()}. Please train the model first.")
        print(f"Loading trained model from: {model_path.name}...")
        self.model = keras.models.load_model(str(model_path))

        # 2. Load Label Map
        if not label_map_path.is_file():
            raise FileNotFoundError(f"Label map not found: {label_map_path.resolve()}")
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
            if "id2label" in raw_labels:
                self.labels = {int(k): v for k, v in raw_labels["id2label"].items()}
            else:
                self.labels = {int(v): k for k, v in raw_labels.items()}
        self.class_names = [self.labels[i] for i in range(len(self.labels))]
        print(f"Loaded {len(self.class_names)} gesture classes: {self.class_names}")

        # 3. Load Latent Class Prototypes & OOD Rejection Thresholds
        self.prototypes = {}
        self.feature_extractor = None
        if prototypes_path.is_file():
            try:
                with open(prototypes_path, "r", encoding="utf-8") as f:
                    self.prototypes = json.load(f)
                dense_layer = self.model.get_layer("dense_64")
                self.feature_extractor = keras.Model(inputs=self.model.input, outputs=dense_layer.output)
                print(f"Loaded {len(self.prototypes)} class latent prototypes for Out-of-Distribution rejection.")
            except Exception as e:
                print(f"[WARNING] Could not initialize prototype feature extractor: {e}")
                self.prototypes = {}

        # 4. Initialize MediaPipe Hands
        self.use_legacy = hasattr(mp, "solutions")
        if self.use_legacy:
            mp_hands = mp.solutions.hands
            self.detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Hands initialized using legacy solutions API.")
        else:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not task_model_path.exists():
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                print(f"Downloading hand_landmarker.task from {url}...")
                urllib.request.urlretrieve(url, str(task_model_path))

            base_opts = python.BaseOptions(model_asset_path=str(task_model_path))
            opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(opts)
            print("MediaPipe HandLandmarker initialized using modern Tasks API.")

        # Buffers for sequence window & smoothing (continuous mode: 60 frames rolling buffer)
        self.rolling_buffer = deque(maxlen=60)
        self.prediction_buffer = deque(maxlen=smoothing_window)
        self.decision_buffer = deque(maxlen=5)  # Temporal consensus buffer
        self.hands_history = deque(maxlen=smoothing_window)

        # Trigger / Recording Mode State Machine
        # States: "IDLE", "COUNTDOWN", "RECORDING", "RESULT"
        self.mode_state = "IDLE"
        self.recorded_frames = []
        self.countdown_start = 0.0
        self.record_start = 0.0
        self.record_duration = 2.0  # 2.0 seconds matches sign duration in training dataset
        self.countdown_duration = 1.0  # 1.0s quick countdown
        self.last_recorded_prediction = None
        self.last_recorded_conf = 0.0
        self.last_recorded_probs = np.zeros(len(self.class_names), dtype=np.float32)
        self.last_recorded_is_rejected = False
        self.last_rejection_reason = ""

    def extract_landmarks(self, raw_frame: np.ndarray):
        """
        Extracts 21 3D landmarks for both hands from the UNFLIPPED camera frame.
        In unflipped camera perspective:
          - Signer's Left hand (on image right) is labeled 'Left' by MediaPipe
          - Signer's Right hand (on image left) is labeled 'Right' by MediaPipe
        This precisely preserves the coordinate system and feature indices of the training dataset.
        """
        image_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        has_left = False
        has_right = False
        detected_hands = []

        if self.use_legacy:
            results = self.detector.process(image_rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    score = handedness.classification[0].score
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                    detected_hands.append((label, coords, score))
        else:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.detector.detect(mp_img)
            if results.hand_landmarks and results.handedness:
                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    label = handedness[0].category_name
                    score = handedness[0].score
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                    detected_hands.append((label, coords, score))

        for label, coords, score in detected_hands:
            flat_coords = coords.flatten()

            # Handle optional hand inversion toggle
            effective_label = label
            if self.invert_hands:
                effective_label = "Right" if label == "Left" else "Left"

            if effective_label == "Left":
                left_hand = flat_coords
                has_left = True
            else:
                right_hand = flat_coords
                has_right = True

        landmarks_126 = np.concatenate([left_hand, right_hand])
        return landmarks_126, has_left, has_right, detected_hands

    @staticmethod
    def normalize_frame(frame_landmarks: np.ndarray) -> np.ndarray:
        """
        Translates landmarks relative to active wrist anchor.
        Crucial bugfix: Only translates landmarks for PRESENT hands.
        Absent hands remain strictly 0.0 to prevent phantom landmark inversion.
        """
        normalized = frame_landmarks.copy()
        has_l = np.any(normalized[:63] != 0)
        has_r = np.any(normalized[63:] != 0)

        if has_r:
            anchor = normalized[63:66].copy()
        elif has_l:
            anchor = normalized[0:3].copy()
        else:
            return normalized

        if has_l:
            for j in range(0, 63, 3):
                normalized[j:j+3] -= anchor
        if has_r:
            for j in range(63, 126, 3):
                normalized[j:j+3] -= anchor

        return normalized

    @staticmethod
    def resample_sequence(sequence: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
        """Standardizes a sequence to exactly target_length frames matching dataset preprocessing."""
        current_len = len(sequence)
        if current_len == target_length:
            return sequence
        elif current_len > target_length:
            indices = np.linspace(0, current_len - 1, target_length).astype(int)
            return sequence[indices]
        else:
            pad_width = target_length - current_len
            padding = np.zeros((pad_width, sequence.shape[1]), dtype=np.float32)
            return np.vstack([sequence, padding])

    def classify_and_verify_sequence(self, seq_30: np.ndarray):
        """
        Multi-Tier Recognition & Unknown Gesture Rejection Engine:
        Evaluates a normalized 30-frame sequence through 5 rejection gates:
          Gate 1: Motion velocity check (filters static resting hands)
          Gate 2: Model inference & negative 'unknown' class check
          Gate 3: Confidence thresholding (requires P >= 0.70)
          Gate 4: Margin thresholding (requires P_1 - P_2 >= 0.25 to reject ambiguous signs)
          Gate 5: Latent prototype distance gating (Out-Of-Distribution rejection)

        Returns:
            word (str): Recognized class name OR "Unknown Gesture" OR status
            conf (float): Top confidence score
            probs (np.ndarray): Probability vector
            is_valid (bool): True if recognized as one of the 8 target gestures
            reason (str): Diagnostic description of decision
        """
        # Gate 1: Motion Velocity Check
        motion_diff = np.abs(seq_30[1:] - seq_30[:-1])
        velocity = float(np.mean(motion_diff))
        if velocity < 0.0025:
            dummy_probs = np.zeros(len(self.class_names), dtype=np.float32)
            return "Hands Ready - Perform Sign", 0.0, dummy_probs, False, "Static hands"

        # Model forward pass
        seq_input = np.expand_dims(seq_30, axis=0)
        probs = self.model.predict(seq_input, verbose=0)[0]

        sorted_indices = np.argsort(probs)[::-1]
        best_idx = int(sorted_indices[0])
        best_conf = float(probs[best_idx])
        second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = best_conf - second_conf

        best_class = self.class_names[best_idx]

        # Gate 2: Negative 'unknown' class check
        if best_class.lower() == "unknown":
            return "Unknown Gesture", best_conf, probs, False, "Classified as non-emergency/unknown motion"

        # Gate 3: Confidence Threshold
        if best_conf < self.confidence_threshold:
            return "Unknown Gesture", best_conf, probs, False, f"Low confidence ({best_conf*100:.1f}% < {self.confidence_threshold*100:.0f}%)"

        # Gate 4: Confidence Margin (Ambiguous Gesture Rejection)
        if margin < self.margin_threshold:
            second_class = self.class_names[sorted_indices[1]]
            return "Unknown Gesture", best_conf, probs, False, f"Ambiguous between '{best_class}' and '{second_class}'"

        # Gate 5: Latent Class Prototype Distance Gating (Out-Of-Distribution Detection)
        if self.feature_extractor is not None and best_class in self.prototypes:
            proto = self.prototypes[best_class]
            embedding = self.feature_extractor.predict(seq_input, verbose=0)[0]
            centroid = np.array(proto["centroid"], dtype=np.float32)
            dist = float(np.linalg.norm(embedding - centroid))
            threshold = float(proto.get("threshold", 999.0))

            if dist > threshold:
                return "Unknown Gesture", best_conf, probs, False, f"OOD gesture pattern (distance {dist:.2f} > {threshold:.2f})"

        # All gates passed: genuine recognized sign!
        return best_class.upper(), best_conf, probs, True, "Confirmed genuine sign"

    def draw_skeleton_on_mirror(self, display_frame: np.ndarray, detected_hands: list):
        """
        Draws landmark joints, connection bones, and labels on the MIRRORED display frame.
        Since display_frame is horizontally flipped from raw_frame, a point at (x, y) on raw_frame
        is drawn at ((1 - x) * w, y * h) so it perfectly aligns with the user's hands on screen!
        """
        h, w, _ = display_frame.shape

        for label, coords, score in detected_hands:
            effective_label = label if not self.invert_hands else ("Right" if label == "Left" else "Left")

            if effective_label == "Left":
                line_color = (255, 215, 0)     # Sky Blue / Cyan (BGR)
                joint_color = (255, 60, 0)     # Deep Blue
                label_color = (255, 230, 100)
                hand_title = "LEFT HAND"
            else:
                line_color = (0, 255, 128)     # Emerald Green (BGR)
                joint_color = (0, 140, 255)    # Bright Orange
                label_color = (100, 255, 150)
                hand_title = "RIGHT HAND"

            # Compute mirrored screen pixel coordinates: x_screen = (1 - x) * w
            screen_pts = []
            for pt in coords:
                sx = int(np.clip((1.0 - pt[0]) * w, 0, w - 1))
                sy = int(np.clip(pt[1] * h, 0, h - 1))
                screen_pts.append((sx, sy))

            # 1. Draw connection lines (bones)
            for (start_idx, end_idx) in HAND_CONNECTIONS:
                pt1 = screen_pts[start_idx]
                pt2 = screen_pts[end_idx]
                cv2.line(display_frame, pt1, pt2, line_color, 2, cv2.LINE_AA)

            # 2. Draw landmark joints
            for i, (px, py) in enumerate(screen_pts):
                radius = 5 if i in [0, 4, 8, 12, 16, 20] else 3
                cv2.circle(display_frame, (px, py), radius, joint_color, -1, cv2.LINE_AA)
                cv2.circle(display_frame, (px, py), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Draw hand pill badge above wrist
            wrist_x, wrist_y = screen_pts[0]
            tag = f"{hand_title} ({int(score * 100)}%)"
            badge_x1 = max(10, wrist_x - 70)
            badge_y1 = max(10, wrist_y - 25)
            cv2.rectangle(display_frame, (badge_x1, badge_y1), (badge_x1 + 145, badge_y1 + 20), (25, 25, 25), -1)
            cv2.rectangle(display_frame, (badge_x1, badge_y1), (badge_x1 + 145, badge_y1 + 20), line_color, 1)
            cv2.putText(display_frame, tag, (badge_x1 + 6, badge_y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, label_color, 1, cv2.LINE_AA)

    def predict_frame(self, raw_frame: np.ndarray):
        """
        Processes one camera frame, updates internal buffers, and returns:
        (predicted_word, confidence, probabilities, has_left, has_right)
        """
        raw_landmarks, has_left, has_right, detected_hands = self.extract_landmarks(raw_frame)
        self.rolling_buffer.append(raw_landmarks)
        self.hands_history.append(has_left or has_right)

        word, conf, probs, is_valid, reason = self.process_continuous_prediction()
        if not (has_left or has_right):
            word = "No Hands in View"
            conf = 0.0

        return word, conf, probs, has_left, has_right

    def process_continuous_prediction(self):
        """Processes continuous rolling buffer with temporal consensus & rejection."""
        if len(self.rolling_buffer) < 20 or not any(self.hands_history):
            return "Waiting for gesture...", 0.0, np.zeros(len(self.class_names), dtype=np.float32), False, "Waiting for frames"

        buffer_array = np.array(list(self.rolling_buffer), dtype=np.float32)
        norm_seq = np.zeros_like(buffer_array)
        for i in range(len(buffer_array)):
            norm_seq[i] = self.normalize_frame(buffer_array[i])

        seq_30 = self.resample_sequence(norm_seq, target_length=SEQUENCE_LENGTH)

        # Run multi-tier verification
        word, conf, probs, is_valid, reason = self.classify_and_verify_sequence(seq_30)

        # Prediction smoothing
        self.prediction_buffer.append(probs)
        smoothed_probs = np.mean(self.prediction_buffer, axis=0)

        # Temporal consensus: track recent valid decisions
        self.decision_buffer.append(word)
        if is_valid:
            # Require at least 3 of last 5 decisions to agree on this word
            agree_count = list(self.decision_buffer).count(word)
            if agree_count >= 3:
                return word, conf, smoothed_probs, True, reason
            else:
                return "Signing...", conf, smoothed_probs, False, "Awaiting temporal consensus"
        else:
            return word, conf, smoothed_probs, False, reason

    def update_state_machine(self, raw_landmarks: np.ndarray, has_left: bool, has_right: bool):
        """Manages transitions between IDLE, COUNTDOWN, RECORDING, and RESULT modes."""
        now = time.time()

        if self.mode_state == "COUNTDOWN":
            elapsed = now - self.countdown_start
            if elapsed >= self.countdown_duration:
                self.mode_state = "RECORDING"
                self.record_start = time.time()
                self.recorded_frames = [raw_landmarks.copy()]
                print(f"[{time.strftime('%H:%M:%S')}] [RECORDING] Capturing 2.0-second gesture trajectory from webcam...")

        elif self.mode_state == "RECORDING":
            self.recorded_frames.append(raw_landmarks.copy())
            elapsed = now - self.record_start
            if elapsed >= self.record_duration:
                if len(self.recorded_frames) >= 10:
                    raw_seq = np.array(self.recorded_frames, dtype=np.float32)
                    norm_seq = np.zeros_like(raw_seq)
                    for i in range(len(raw_seq)):
                        norm_seq[i] = self.normalize_frame(raw_seq[i])
                    seq_30 = self.resample_sequence(norm_seq, target_length=SEQUENCE_LENGTH)

                    # Run multi-tier rejection verification on the complete recording
                    word, conf, probs, is_valid, reason = self.classify_and_verify_sequence(seq_30)

                    self.last_recorded_prediction = word
                    self.last_recorded_conf = conf
                    self.last_recorded_probs = probs
                    self.last_recorded_is_rejected = not is_valid
                    self.last_rejection_reason = reason
                    self.mode_state = "RESULT"

                    # Immediately log final output to terminal
                    self.log_to_terminal(
                        word=self.last_recorded_prediction,
                        conf=self.last_recorded_conf,
                        probs=self.last_recorded_probs,
                        has_left=has_left,
                        has_right=has_right,
                        mode="SPACEBAR RECORD (2.0s)",
                        is_rejected=not is_valid,
                        reason=reason,
                        force=True
                    )
                else:
                    self.mode_state = "IDLE"

    def log_to_terminal(
        self,
        word: str,
        conf: float,
        probs: np.ndarray,
        has_left: bool,
        has_right: bool,
        mode: str,
        is_rejected: bool = False,
        reason: str = "",
        force: bool = False
    ):
        """Prints formatted final prediction output to the terminal with timestamps and candidate breakdown."""
        now = time.time()
        is_confident = (conf >= self.confidence_threshold) or is_rejected
        word_changed = (word != self.last_terminal_word)
        time_elapsed = (now - self.last_terminal_time) >= self.terminal_interval

        if not is_confident and not force:
            if self.last_terminal_word not in [None, "No Hands in View", "Waiting for gesture...", "Hands Ready - Perform Sign"]:
                print(f"[{time.strftime('%H:%M:%S')}] [STATUS] Gesture ended / Waiting for next sign...")
                self.last_terminal_word = None
            return

        if force or word_changed or time_elapsed:
            self.last_terminal_word = word
            self.last_terminal_time = now
            timestamp = time.strftime("%H:%M:%S")

            # Determine top-3 predictions (excluding 'unknown' if desired)
            top_indices = np.argsort(probs)[::-1][:3]
            top_breakdown = ", ".join([f"{self.class_names[idx].upper()} ({probs[idx] * 100:.1f}%)" for idx in top_indices if probs[idx] > 0.005])
            if not top_breakdown:
                top_breakdown = f"{word.upper()} ({conf * 100:.1f}%)"

            hands_str = "Both Hands Active" if (has_left and has_right) else ("Left Hand Only" if has_left else ("Right Hand Only" if has_right else "No Hands"))

            banner = "=" * 76
            print(f"\n{banner}")
            if is_rejected or word == "Unknown Gesture":
                print(f"  >>> FINAL RECOGNITION OUTPUT: [ UNKNOWN GESTURE ] (REJECTED) <<<")
                if reason:
                    print(f"  Rejection Reason : {reason}")
            else:
                print(f"  >>> FINAL RECOGNITION OUTPUT: [ {word.upper()} ] ({conf * 100:.1f}% Confidence) <<<")
            print(f"  Timestamp : {timestamp}  |  Mode: {mode}")
            print(f"  Sensors   : {hands_str}  |  Tracking 2x21 Hand Skeletons")
            print(f"  Top Signs : {top_breakdown}")
            print(f"{banner}\n")

    def render_hud(
        self,
        display_frame: np.ndarray,
        continuous_word: str,
        continuous_conf: float,
        continuous_probs: np.ndarray,
        has_left: bool,
        has_right: bool,
        fps: float
    ):
        """
        Renders HUD overlay on the camera frame AND constructs a dedicated
        bottom dashboard canvas where the final output is prominently displayed
        below the camera interface.
        """
        h, w, _ = display_frame.shape

        # Determine active display word, confidence, and mode
        if self.mode_state == "RESULT" and self.last_recorded_prediction:
            active_word = self.last_recorded_prediction
            active_conf = self.last_recorded_conf
            active_probs = self.last_recorded_probs
            is_rejected = self.last_recorded_is_rejected
            mode_badge = "[RECORDED RESULT]"
            mode_color = (0, 165, 255) if is_rejected else (0, 255, 120)
        elif self.mode_state == "RECORDING":
            active_word = "RECORDING SIGN..."
            active_conf = 0.0
            active_probs = continuous_probs
            is_rejected = False
            mode_badge = "[RECORDING IN PROGRESS]"
            mode_color = (0, 69, 255)
        elif self.mode_state == "COUNTDOWN":
            active_word = "GET READY TO SIGN..."
            active_conf = 0.0
            active_probs = continuous_probs
            is_rejected = False
            mode_badge = "[COUNTDOWN]"
            mode_color = (0, 215, 255)
        else:
            active_word = continuous_word
            active_conf = continuous_conf
            active_probs = continuous_probs
            is_rejected = (continuous_word == "Unknown Gesture")
            mode_badge = "[CONTINUOUS LIVE]"
            mode_color = (200, 200, 200)

        # Word color based on rejection, confidence, and emergency level
        if is_rejected or active_word == "Unknown Gesture":
            status_color = (0, 140, 255)  # Amber / Warning Orange
        elif active_conf >= self.confidence_threshold:
            status_color = (0, 69, 255) if active_word in ["ACCIDENT", "HELP", "DOCTOR", "CALL"] else (0, 230, 115)
        else:
            status_color = (180, 180, 180)

        # -------------------------------------------------------------
        # 1. Overlay on the Camera Feed (Top Section)
        # -------------------------------------------------------------
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

        # Header title
        cv2.putText(display_frame, "ISL EMERGENCY SIGN RECOGNITION (HAND-INDEPENDENT)", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

        # Status & FPS
        inv_text = " | [F] Inverted Hands" if self.invert_hands else ""
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Buffer: {len(self.rolling_buffer)}/60{inv_text}", (w - 380, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # Animation Overlay (Countdown & Recording)
        if self.mode_state == "COUNTDOWN":
            rem = max(0.0, self.countdown_duration - (time.time() - self.countdown_start))
            count_num = int(np.ceil(rem * 3))
            count_num = max(1, min(3, count_num))
            cv2.circle(display_frame, (w // 2, h // 2), 85, (0, 215, 255), -1)
            cv2.circle(display_frame, (w // 2, h // 2), 90, (255, 255, 255), 3)
            cv2.putText(display_frame, str(count_num), (w // 2 - 25, h // 2 + 28), cv2.FONT_HERSHEY_DUPLEX, 3.0, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.putText(display_frame, "PREPARE TO SIGN", (w // 2 - 125, h // 2 + 120), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 2, cv2.LINE_AA)

        elif self.mode_state == "RECORDING":
            prog = min(1.0, (time.time() - self.record_start) / self.record_duration)
            pulse = int(abs(np.sin(time.time() * 6)) * 255)
            cv2.circle(display_frame, (w // 2 - 145, 95), 9, (0, 0, pulse), -1)
            cv2.putText(display_frame, f"RECORDING SIGN... ({prog * 100:.0f}%)", (w // 2 - 125, 102), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 69, 255), 2, cv2.LINE_AA)
            pb_x, pb_y, pb_w, pb_h = w // 2 - 170, 115, 340, 14
            cv2.rectangle(display_frame, (pb_x, pb_y), (pb_x + pb_w, pb_y + pb_h), (40, 40, 40), -1)
            cv2.rectangle(display_frame, (pb_x, pb_y), (pb_x + int(pb_w * prog), pb_y + pb_h), (0, 69, 255), -1)

        # -------------------------------------------------------------
        # 2. Dedicated Bottom Dashboard Canvas (Displayed Below Interface)
        # -------------------------------------------------------------
        panel_h = self.bottom_panel_height
        bottom_panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
        bottom_panel[:] = (22, 22, 26)  # Dark charcoal theme

        # Top border accent line separating camera from output dashboard
        accent_color = status_color if (active_conf >= self.confidence_threshold or is_rejected) else (65, 65, 75)
        cv2.line(bottom_panel, (0, 0), (w, 0), accent_color, 3)

        # Dynamic layout coordinates based on width
        col1_w = int(w * 0.40)
        col2_w = int(w * 0.30)
        col1_x = 20
        col2_x = col1_x + col1_w + 15
        col3_x = col2_x + col2_w + 15

        # Divider lines between columns
        cv2.line(bottom_panel, (col2_x - 10, 15), (col2_x - 10, panel_h - 15), (45, 45, 52), 1)
        cv2.line(bottom_panel, (col3_x - 10, 15), (col3_x - 10, panel_h - 15), (45, 45, 52), 1)

        # --- COLUMN 1: FINAL RECOGNITION OUTPUT ---
        cv2.putText(bottom_panel, "FINAL OUTPUT / RECOGNIZED SIGN", (col1_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 180, 200), 1, cv2.LINE_AA)

        display_word_text = active_word if len(active_word) <= 15 else active_word[:14] + ".."
        cv2.putText(bottom_panel, f"[ {display_word_text} ]", (col1_x, 68), cv2.FONT_HERSHEY_DUPLEX, 1.25, status_color, 2, cv2.LINE_AA)

        conf_pct_str = f"Confidence: {active_conf * 100:.1f}%" if active_conf > 0 else "Confidence: --"
        cv2.putText(bottom_panel, conf_pct_str, (col1_x, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

        bar_x, bar_y, bar_w, bar_h = col1_x, 114, min(220, col1_w - 40), 12
        cv2.rectangle(bottom_panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 55), 1)
        fill_w = int(bar_w * min(1.0, active_conf))
        cv2.rectangle(bottom_panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)

        # Status pill
        if is_rejected or active_word == "Unknown Gesture":
            pill_tag, pill_col = "REJECTED (UNKNOWN)", (0, 140, 255)
        elif active_conf >= 0.85:
            pill_tag, pill_col = "CONFIRMED (HIGH)", (0, 230, 115)
        elif active_conf >= self.confidence_threshold:
            pill_tag, pill_col = "DETECTED (MODERATE)", (0, 215, 255)
        else:
            pill_tag, pill_col = "SEARCHING...", (140, 140, 140)
        cv2.putText(bottom_panel, pill_tag, (bar_x + bar_w + 12, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pill_col, 1, cv2.LINE_AA)

        # --- COLUMN 2: SENSOR & REJECTION STATUS ---
        cv2.putText(bottom_panel, "SENSOR & HAND TRACKING", (col2_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 180, 200), 1, cv2.LINE_AA)

        left_str = "ACTIVE" if has_left else "MISSING"
        left_col = (255, 215, 0) if has_left else (100, 100, 150)
        cv2.putText(bottom_panel, f"Left Hand  : {left_str}", (col2_x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, left_col, 1, cv2.LINE_AA)

        right_str = "ACTIVE" if has_right else "MISSING"
        right_col = (0, 255, 128) if has_right else (100, 100, 150)
        cv2.putText(bottom_panel, f"Right Hand : {right_str}", (col2_x, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.46, right_col, 1, cv2.LINE_AA)

        if has_left and has_right:
            guide_str, guide_col = "[OK] Both Hands Engaged", (0, 230, 115)
        elif has_left or has_right:
            guide_str, guide_col = "[OK] Single Hand (Hand-Independent)", (0, 215, 255)
        else:
            guide_str, guide_col = "Raise hands in front of camera", (140, 140, 150)
        cv2.putText(bottom_panel, guide_str, (col2_x, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.44, guide_col, 1, cv2.LINE_AA)

        cv2.putText(bottom_panel, f"Mode: {mode_badge}", (col2_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1, cv2.LINE_AA)

        # --- COLUMN 3: TOP CANDIDATES & SHORTCUTS ---
        cv2.putText(bottom_panel, "TOP CANDIDATES", (col3_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 180, 200), 1, cv2.LINE_AA)

        top_indices = np.argsort(active_probs)[::-1][:3]
        for idx_rank, c_idx in enumerate(top_indices):
            cname = self.class_names[c_idx]
            pval = active_probs[c_idx]
            row_y = 52 + idx_rank * 24
            is_active_sign = (cname.upper() == active_word and active_conf >= self.confidence_threshold)
            row_col = (0, 255, 120) if is_active_sign else (210, 210, 210)

            cv2.putText(bottom_panel, f"#{idx_rank + 1} {cname[:8]:8s}", (col3_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, row_col, 1, cv2.LINE_AA)

            m_bar_x = col3_x + 85
            m_bar_w = 60
            cv2.rectangle(bottom_panel, (m_bar_x, row_y - 8), (m_bar_x + m_bar_w, row_y - 1), (45, 45, 50), 1)
            m_fill = int(m_bar_w * min(1.0, float(pval)))
            cv2.rectangle(bottom_panel, (m_bar_x, row_y - 8), (m_bar_x + m_fill, row_y - 1), row_col, -1)

            cv2.putText(bottom_panel, f"{pval * 100:4.1f}%", (m_bar_x + m_bar_w + 8, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, row_col, 1, cv2.LINE_AA)

        # Bottom Hotkeys Strip
        shortcuts = "[SPACE] Record Sign (2s)   |   [F] Toggle Inverted Hands   |   [C] Clear Output   |   [Q] Exit"
        cv2.putText(bottom_panel, shortcuts, (20, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 190), 1, cv2.LINE_AA)

        full_canvas = np.vstack([display_frame, bottom_panel])
        return full_canvas

    def run(self, camera_index: int = 0):
        """Starts live webcam recognition loop."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam at index {camera_index}.")
            print("Please ensure a camera is connected and not currently used by another program.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=" * 76)
        print("LIVE ISL EMERGENCY GESTURE RECOGNITION ACTIVE")
        print("=" * 76)
        print("Features: Left/Right Hand Independent + Unknown Gesture Rejection")
        print("Final output is displayed BELOW the interface and in this terminal.")
        print("Hotkeys:")
        print("  - [SPACE] : Trigger 2.0-second gesture recording (recommended)")
        print("  - [F]     : Toggle hand inversion (if webcam driver pre-mirrors)")
        print("  - [C]     : Clear current result")
        print("  - [Q]/ESC : Exit program")
        print("=" * 76)
        print(f"[{time.strftime('%H:%M:%S')}] System initialized. Waiting for hand gestures...\n")

        prev_time = time.time()
        fps = 30.0

        try:
            while cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret:
                    print("[WARNING] Failed to capture video frame from camera.")
                    break

                curr_time = time.time()
                dt = curr_time - prev_time
                fps = 1.0 / dt if dt > 0 else fps
                prev_time = curr_time

                # 1. EXTRACT LANDMARKS FROM UNMIRRORED RAW FRAME
                raw_landmarks, has_left, has_right, detected_hands = self.extract_landmarks(raw_frame)

                # Update rolling buffers for continuous mode
                self.rolling_buffer.append(raw_landmarks)
                self.hands_history.append(has_left or has_right)

                # Update State Machine for Spacebar Record Mode
                self.update_state_machine(raw_landmarks, has_left, has_right)

                # Run continuous prediction
                cont_word, cont_conf, cont_probs, is_valid, reason = self.process_continuous_prediction()

                # If in continuous mode and not recording, log confident predictions to terminal
                if self.mode_state == "IDLE":
                    self.log_to_terminal(
                        word=cont_word,
                        conf=cont_conf,
                        probs=cont_probs,
                        has_left=has_left,
                        has_right=has_right,
                        mode="CONTINUOUS STREAM",
                        is_rejected=not is_valid and cont_word == "Unknown Gesture",
                        reason=reason,
                        force=False
                    )

                # 2. CREATE MIRRORED FRAME FOR USER DISPLAY
                display_frame = cv2.flip(raw_frame, 1)

                # 3. DRAW DUAL HAND SKELETONS ALIGNED ON MIRRORED DISPLAY
                self.draw_skeleton_on_mirror(display_frame, detected_hands)

                # 4. RENDER FULL INTERFACE (WITH FINAL OUTPUT BELOW INTERFACE)
                composite_interface = self.render_hud(display_frame, cont_word, cont_conf, cont_probs, has_left, has_right, fps)

                # 5. DISPLAY WINDOW & PROCESS KEY EVENTS
                cv2.imshow("ISL Emergency Gesture Recognition (Both Hands)", composite_interface)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    print(f"[{time.strftime('%H:%M:%S')}] Exiting live recognition.")
                    break
                elif key == 32:  # SPACEBAR
                    if self.mode_state in ["IDLE", "RESULT"]:
                        print(f"\n[{time.strftime('%H:%M:%S')}] [TRIGGER] Spacebar pressed! Starting 1.0s countdown...")
                        print(f"[{time.strftime('%H:%M:%S')}] [COUNTDOWN] 3.. 2.. 1.. SIGN!")
                        self.mode_state = "COUNTDOWN"
                        self.countdown_start = time.time()
                elif key == ord("f") or key == ord("F"):
                    self.invert_hands = not self.invert_hands
                    status = "ON (Swapped)" if self.invert_hands else "OFF (Normal)"
                    print(f"\n[{time.strftime('%H:%M:%S')}] [TOGGLE] Hand inversion set to: {status}\n")
                elif key == ord("c") or key == ord("C"):
                    self.mode_state = "IDLE"
                    self.last_recorded_prediction = None
                    self.last_recorded_conf = 0.0
                    self.rolling_buffer.clear()
                    self.prediction_buffer.clear()
                    self.decision_buffer.clear()
                    print(f"\n[{time.strftime('%H:%M:%S')}] [CLEAR] Final output display and buffers reset.\n")

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = LiveISLRecognizer(confidence_threshold=0.70)
    recognizer.run(camera_index=0)
