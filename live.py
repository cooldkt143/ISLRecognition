"""
Live Webcam Indian Sign Language (ISL) Emergency Gesture Recognition.
Features:
- Dual-hand skeletal tracking (Left & Right hands, 21 landmarks each, 126 features)
- High-accuracy Bidirectional LSTM model with temporal pooling
- Flawless mirror display with correct unmirrored landmark coordinate extraction
- Dual recognition modes:
    1. Continuous Live Streaming (with motion energy detection and dynamic resampling)
    2. Spacebar Trigger / Record Mode (press SPACE to record an exact 2-second gesture window)
- Real-time on-screen probability panel, confidence meter, and dual-hand status indicators
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
        task_model_path: Path = None,
        confidence_threshold: float = 0.65,
        smoothing_window: int = 6
    ):
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.invert_hands = False  # Toggle with [F] if camera driver already pre-mirrors

        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "isl_lstm_model.keras"
        if label_map_path is None:
            label_map_path = PROJECT_ROOT / "models" / "label_map.json"
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
        print(f"Loaded {len(self.class_names)} emergency gesture classes: {self.class_names}")

        # 3. Initialize MediaPipe Hands
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
        """Translates landmarks relative to wrist anchor matching preprocessing logic."""
        normalized = frame_landmarks.copy()
        if np.any(normalized[63:66]):
            anchor = normalized[63:66].copy()
        elif np.any(normalized[0:3]):
            anchor = normalized[0:3].copy()
        else:
            return normalized

        for j in range(0, 126, 3):
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

    def draw_skeleton_on_mirror(self, display_frame: np.ndarray, detected_hands: list):
        """
        Draws landmark joints, connection bones, and labels on the MIRRORED display frame.
        Since display_frame is horizontally flipped from raw_frame, a point at (x, y) on raw_frame
        is drawn at ((1 - x) * w, y * h) so it perfectly aligns with the user's hands on screen!
        """
        h, w, _ = display_frame.shape

        for label, coords, score in detected_hands:
            # Determine visual colors:
            # In mirrored view, Left Hand is on the user's left side
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

        word, conf, probs = self.process_continuous_prediction()
        if not (has_left or has_right):
            word = "No Hands in View"
            conf = 0.0

        return word, conf, probs, has_left, has_right

    def process_continuous_prediction(self):
        """Processes continuous rolling buffer and returns prediction metadata."""
        if len(self.rolling_buffer) < 20 or not any(self.hands_history):
            return "Waiting for gesture...", 0.0, np.zeros(len(self.class_names), dtype=np.float32)

        # Resample rolling buffer to 30 frames matching model input shape
        buffer_array = np.array(list(self.rolling_buffer), dtype=np.float32)
        norm_seq = np.zeros_like(buffer_array)
        for i in range(len(buffer_array)):
            norm_seq[i] = self.normalize_frame(buffer_array[i])

        seq_30 = self.resample_sequence(norm_seq, target_length=SEQUENCE_LENGTH)

        # Calculate motion velocity to ensure active signing rather than resting pose
        motion_diff = np.abs(seq_30[1:] - seq_30[:-1])
        velocity = np.mean(motion_diff)

        # Run inference if hands are active
        seq_input = np.expand_dims(seq_30, axis=0)
        raw_probs = self.model.predict(seq_input, verbose=0)[0]

        self.prediction_buffer.append(raw_probs)
        smoothed_probs = np.mean(self.prediction_buffer, axis=0)

        best_idx = int(np.argmax(smoothed_probs))
        best_conf = float(smoothed_probs[best_idx])

        if velocity < 0.003 and best_conf < self.confidence_threshold:
            return "Hands Ready - Perform Sign", 0.0, smoothed_probs

        if best_conf >= self.confidence_threshold:
            word = self.class_names[best_idx].upper()
            return word, best_conf, smoothed_probs
        else:
            return "Signing...", best_conf, smoothed_probs

    def update_state_machine(self, raw_landmarks: np.ndarray, has_left: bool, has_right: bool):
        """Manages transitions between IDLE, COUNTDOWN, RECORDING, and RESULT modes."""
        now = time.time()
        both_hands = has_left and has_right

        if self.mode_state == "COUNTDOWN":
            elapsed = now - self.countdown_start
            if elapsed >= self.countdown_duration:
                self.mode_state = "RECORDING"
                self.record_start = time.time()
                self.recorded_frames = [raw_landmarks.copy()]

        elif self.mode_state == "RECORDING":
            self.recorded_frames.append(raw_landmarks.copy())
            elapsed = now - self.record_start
            if elapsed >= self.record_duration:
                # Finished recording: resample and run full sequence prediction
                if len(self.recorded_frames) >= 10:
                    raw_seq = np.array(self.recorded_frames, dtype=np.float32)
                    norm_seq = np.zeros_like(raw_seq)
                    for i in range(len(raw_seq)):
                        norm_seq[i] = self.normalize_frame(raw_seq[i])
                    seq_30 = self.resample_sequence(norm_seq, target_length=SEQUENCE_LENGTH)

                    seq_input = np.expand_dims(seq_30, axis=0)
                    probs = self.model.predict(seq_input, verbose=0)[0]

                    best_idx = int(np.argmax(probs))
                    self.last_recorded_prediction = self.class_names[best_idx].upper()
                    self.last_recorded_conf = float(probs[best_idx])
                    self.last_recorded_probs = probs
                    self.mode_state = "RESULT"
                else:
                    self.mode_state = "IDLE"

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
        """Renders comprehensive, modern HUD with dynamic cards, meters, probability bars, and hotkey hints."""
        h, w, _ = display_frame.shape
        overlay = display_frame.copy()

        # 1. Top Header Banner
        cv2.rectangle(overlay, (0, 0), (w, 105), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)

        # Determine display word and confidence based on mode
        if self.mode_state == "RESULT" and self.last_recorded_prediction:
            active_word = self.last_recorded_prediction
            active_conf = self.last_recorded_conf
            active_probs = self.last_recorded_probs
            mode_badge = "[RECORDED RESULT]"
            mode_color = (0, 255, 120)
        elif self.mode_state == "RECORDING":
            active_word = "RECORDING SIGN..."
            active_conf = 0.0
            active_probs = continuous_probs
            mode_badge = "[RECORDING IN PROGRESS]"
            mode_color = (0, 69, 255)
        elif self.mode_state == "COUNTDOWN":
            active_word = "GET READY TO SIGN..."
            active_conf = 0.0
            active_probs = continuous_probs
            mode_badge = "[COUNTDOWN]"
            mode_color = (0, 215, 255)
        else:
            active_word = continuous_word
            active_conf = continuous_conf
            active_probs = continuous_probs
            mode_badge = "[CONTINUOUS LIVE]"
            mode_color = (200, 200, 200)

        # Word color based on confidence and sign type
        if active_conf >= self.confidence_threshold:
            status_color = (0, 69, 255) if active_word in ["ACCIDENT", "HELP", "DOCTOR", "CALL"] else (0, 255, 120)
            glow = 3
        else:
            status_color = (190, 190, 190)
            glow = 2

        # Display Sign Name
        cv2.putText(display_frame, f"SIGN: {active_word}", (20, 52), cv2.FONT_HERSHEY_DUPLEX, 1.3, status_color, glow, cv2.LINE_AA)

        # Mode Badge & Invert Indicator
        inv_text = " [Hands Inverted (F)]" if self.invert_hands else ""
        cv2.putText(display_frame, f"{mode_badge}{inv_text}", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 1, cv2.LINE_AA)

        # Confidence percentage & Progress Bar
        conf_str = f"Confidence: {active_conf * 100:.1f}%" if active_conf > 0 else "Confidence: --"
        cv2.putText(display_frame, conf_str, (240, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        bar_x, bar_y, bar_w, bar_h = 420, 75, 160, 14
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), 1)
        fill_w = int(bar_w * min(1.0, active_conf))
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)

        # 2. Hands Detection Status Indicator (Right Header)
        if has_left and has_right:
            hands_status = "BOTH HANDS ACTIVE"
            hands_col = (0, 255, 120)
        elif has_left:
            hands_status = "LEFT HAND ONLY (Need Both!)"
            hands_col = (255, 215, 0)
        elif has_right:
            hands_status = "RIGHT HAND ONLY (Need Both!)"
            hands_col = (0, 255, 128)
        else:
            hands_status = "NO HANDS IN VIEW"
            hands_col = (100, 100, 255)

        cv2.putText(display_frame, hands_status, (w - 420, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.62, hands_col, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Buffer: {len(self.rolling_buffer)}/60", (w - 420, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Mode Animation Overlay (Countdown & Recording)
        if self.mode_state == "COUNTDOWN":
            rem = max(0.0, self.countdown_duration - (time.time() - self.countdown_start))
            count_num = int(np.ceil(rem * 3))
            count_num = max(1, min(3, count_num))
            cv2.circle(display_frame, (w // 2, h // 2), 90, (0, 215, 255), -1)
            cv2.circle(display_frame, (w // 2, h // 2), 95, (255, 255, 255), 3)
            cv2.putText(display_frame, str(count_num), (w // 2 - 25, h // 2 + 30), cv2.FONT_HERSHEY_DUPLEX, 3.2, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.putText(display_frame, "PREPARE TO SIGN", (w // 2 - 130, h // 2 + 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)

        elif self.mode_state == "RECORDING":
            prog = min(1.0, (time.time() - self.record_start) / self.record_duration)
            # Red recording indicator circle
            pulse = int(abs(np.sin(time.time() * 6)) * 255)
            cv2.circle(display_frame, (w // 2 - 150, 135), 10, (0, 0, pulse), -1)
            cv2.putText(display_frame, f"RECORDING SIGN... ({prog * 100:.0f}%)", (w // 2 - 130, 142), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 69, 255), 2, cv2.LINE_AA)
            # Big center progress bar
            pb_x, pb_y, pb_w, pb_h = w // 2 - 180, 155, 360, 16
            cv2.rectangle(display_frame, (pb_x, pb_y), (pb_x + pb_w, pb_y + pb_h), (40, 40, 40), -1)
            cv2.rectangle(display_frame, (pb_x, pb_y), (pb_x + int(pb_w * prog), pb_y + pb_h), (0, 69, 255), -1)

        # 4. Right Probability Dashboard
        panel_w = 220
        panel_h = len(self.class_names) * 27 + 24
        panel_x = w - panel_w - 15
        panel_y = 120

        overlay_panel = display_frame.copy()
        cv2.rectangle(overlay_panel, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_panel, 0.80, display_frame, 0.20, 0, display_frame)

        cv2.putText(display_frame, "GESTURE PROBABILITIES", (panel_x + 10, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 255), 1, cv2.LINE_AA)

        for idx, (cname, pval) in enumerate(zip(self.class_names, active_probs)):
            row_y = panel_y + 44 + idx * 26
            is_winner = (cname.upper() == active_word and active_conf >= self.confidence_threshold)
            row_col = (0, 255, 120) if is_winner else (210, 210, 210)
            txt_weight = 2 if is_winner else 1

            # Class text
            cv2.putText(display_frame, f"{cname[:8]:8s}", (panel_x + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, row_col, txt_weight, cv2.LINE_AA)

            # Mini bar
            mini_bar_x = panel_x + 85
            mini_bar_w = 75
            cv2.rectangle(display_frame, (mini_bar_x, row_y - 10), (mini_bar_x + mini_bar_w, row_y - 2), (50, 50, 50), 1)
            fill_len = int(mini_bar_w * min(1.0, float(pval)))
            cv2.rectangle(display_frame, (mini_bar_x, row_y - 10), (mini_bar_x + fill_len, row_y - 2), row_col, -1)

            # Percent
            cv2.putText(display_frame, f"{pval * 100:4.1f}%", (panel_x + 168, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, row_col, 1, cv2.LINE_AA)

        # 5. Bottom Navigation & Hotkeys Hint Bar
        bar_bottom_y = h - 35
        cv2.rectangle(overlay, (0, bar_bottom_y), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)

        hints = "[SPACE] Record Sign (2s)   |   [F] Toggle Inverted Hands   |   [C] Clear Result   |   [Q] Quit"
        cv2.putText(display_frame, hints, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    def run(self, camera_index: int = 0):
        """Starts live webcam recognition loop."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam at index {camera_index}.")
            print("Please ensure a camera is connected and not currently used by another program.")
            return

        # Attempt to set reasonable resolution (1280x720 or 640x480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=" * 68)
        print("LIVE ISL EMERGENCY GESTURE RECOGNITION (DUAL-HAND LSTM) ACTIVE")
        print("=" * 68)
        print("Tracking BOTH Left and Right hands with 21 skeletal joints each.")
        print("Hotkeys:")
        print("  - [SPACE] : Trigger 2.0-second gesture recording (recommended)")
        print("  - [F]     : Toggle hand inversion (if webcam driver pre-mirrors)")
        print("  - [C]     : Clear current result")
        print("  - [Q]/ESC : Exit program")
        print("=" * 68)

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
                # This guarantees coordinate alignment with the training dataset
                raw_landmarks, has_left, has_right, detected_hands = self.extract_landmarks(raw_frame)

                # Update rolling buffers for continuous mode
                self.rolling_buffer.append(raw_landmarks)
                self.hands_history.append(has_left or has_right)

                # Update State Machine for Spacebar Record Mode
                self.update_state_machine(raw_landmarks, has_left, has_right)

                # Run continuous prediction
                cont_word, cont_conf, cont_probs = self.process_continuous_prediction()

                # 2. CREATE MIRRORED FRAME FOR INTUITIVE USER EXPERIENCE
                display_frame = cv2.flip(raw_frame, 1)

                # 3. DRAW DUAL HAND SKELETONS ALIGNED ON MIRRORED DISPLAY
                self.draw_skeleton_on_mirror(display_frame, detected_hands)

                # 4. RENDER COMPREHENSIVE HUD
                self.render_hud(display_frame, cont_word, cont_conf, cont_probs, has_left, has_right, fps)

                # 5. DISPLAY WINDOW & PROCESS KEY EVENTS
                cv2.imshow("ISL Emergency Gesture Recognition (Both Hands)", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    print("Exiting live recognition.")
                    break
                elif key == 32:  # SPACEBAR
                    if self.mode_state in ["IDLE", "RESULT"]:
                        print("[TRIGGER] Starting gesture recording...")
                        self.mode_state = "COUNTDOWN"
                        self.countdown_start = time.time()
                elif key == ord("f") or key == ord("F"):
                    self.invert_hands = not self.invert_hands
                    status = "ON (Swapped)" if self.invert_hands else "OFF (Normal)"
                    print(f"[TOGGLE] Hand inversion set to: {status}")
                elif key == ord("c") or key == ord("C"):
                    self.mode_state = "IDLE"
                    self.last_recorded_prediction = None
                    self.last_recorded_conf = 0.0
                    self.rolling_buffer.clear()
                    self.prediction_buffer.clear()
                    print("[CLEAR] Buffer and results reset.")

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = LiveISLRecognizer(confidence_threshold=0.65)
    recognizer.run(camera_index=0)
