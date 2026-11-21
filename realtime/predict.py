"""
Real-Time ISL Emergency Gesture Recognition via Webcam.
Captures live frames, extracts MediaPipe hand landmarks, maintains a 30-frame rolling buffer,
normalizes with wrist anchoring (preserving absent hand zeros), and predicts gestures in real time
with hand-independence and multi-gate unknown gesture rejection.
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

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEQUENCE_LENGTH = 30
FEATURE_DIM = 126


class RealTimeGestureRecognizer:
    def __init__(
        self,
        model_path: Path = None,
        label_map_path: Path = None,
        prototypes_path: Path = None,
        task_model_path: Path = None,
        confidence_threshold: float = 0.70,
        margin_threshold: float = 0.25,
        smoothing_window: int = 8
    ):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.smoothing_window = smoothing_window

        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "isl_lstm_model.keras"
        if label_map_path is None:
            label_map_path = PROJECT_ROOT / "models" / "label_map.json"
        if prototypes_path is None:
            prototypes_path = PROJECT_ROOT / "models" / "class_prototypes.json"
        if task_model_path is None:
            task_model_path = PROJECT_ROOT / "hand_landmarker.task"

        # 1. Load trained LSTM model
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found at: {model_path.resolve()}. Please train the model first.")
        print(f"Loading trained Keras model from: {model_path.name}...")
        self.model = keras.models.load_model(str(model_path))

        # 2. Load class labels
        if not label_map_path.is_file():
            raise FileNotFoundError(f"Label map not found at: {label_map_path.resolve()}")
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
            if "id2label" in raw_labels:
                self.labels = {int(k): v for k, v in raw_labels["id2label"].items()}
            else:
                self.labels = {int(v): k for k, v in raw_labels.items()}
        self.class_names = [self.labels[i] for i in range(len(self.labels))]
        print(f"Loaded {len(self.class_names)} gesture classes: {self.class_names}")

        # 3. Load Latent Class Prototypes for OOD Distance Gating
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

        # 4. Initialize MediaPipe
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

        # Buffers
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=smoothing_window)
        self.hand_detected_buffer = deque(maxlen=smoothing_window)
        self.decision_buffer = deque(maxlen=5)

    def extract_landmarks(self, frame: np.ndarray):
        """Extracts 21 3D coordinates for Left and Right hands (126 features)."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)
        hands_found = 0

        if self.use_legacy:
            results = self.detector.process(image_rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                hands_found = len(results.multi_hand_landmarks)
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()
                    label = handedness.classification[0].label
                    if label == "Left":
                        left_hand = coords
                    else:
                        right_hand = coords
        else:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.detector.detect(mp_img)
            if results.hand_landmarks and results.handedness:
                hands_found = len(results.hand_landmarks)
                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32).flatten()
                    label = handedness[0].category_name
                    if label == "Left":
                        left_hand = coords
                    else:
                        right_hand = coords

        landmarks_126 = np.concatenate([left_hand, right_hand])
        return landmarks_126, hands_found

    @staticmethod
    def normalize_frame(frame_landmarks: np.ndarray) -> np.ndarray:
        """
        Translates landmarks relative to active wrist anchor.
        Only translates present hands; absent hands remain strictly 0.0.
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

    def predict_frame(self, frame: np.ndarray):
        """Processes one video frame and returns prediction metadata with unknown rejection."""
        raw_landmarks, hands_found = self.extract_landmarks(frame)
        norm_landmarks = self.normalize_frame(raw_landmarks)

        self.frame_buffer.append(norm_landmarks)
        self.hand_detected_buffer.append(hands_found > 0)

        predicted_label = "Waiting for Gesture..."
        confidence = 0.0
        probabilities = np.zeros(len(self.class_names), dtype=np.float32)

        if len(self.frame_buffer) == SEQUENCE_LENGTH and any(self.hand_detected_buffer):
            seq_array = np.array(self.frame_buffer, dtype=np.float32)

            # Motion velocity check
            diff = np.abs(seq_array[1:] - seq_array[:-1])
            velocity = float(np.mean(diff))

            if velocity < 0.0025:
                return "Hands Ready - Perform Sign", 0.0, probabilities, hands_found

            seq_input = np.expand_dims(seq_array, axis=0)
            raw_probs = self.model.predict(seq_input, verbose=0)[0]

            self.prediction_buffer.append(raw_probs)
            smoothed_probs = np.mean(self.prediction_buffer, axis=0)
            probabilities = smoothed_probs

            sorted_indices = np.argsort(smoothed_probs)[::-1]
            best_idx = int(sorted_indices[0])
            best_conf = float(smoothed_probs[best_idx])
            second_conf = float(smoothed_probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = best_conf - second_conf

            best_class = self.class_names[best_idx]

            # Multi-tier rejection checks
            is_rejected = False
            if best_class.lower() == "unknown":
                is_rejected = True
            elif best_conf < self.confidence_threshold:
                is_rejected = True
            elif margin < self.margin_threshold:
                is_rejected = True
            elif self.feature_extractor is not None and best_class in self.prototypes:
                proto = self.prototypes[best_class]
                embedding = self.feature_extractor.predict(seq_input, verbose=0)[0]
                dist = float(np.linalg.norm(embedding - np.array(proto["centroid"])))
                if dist > float(proto.get("threshold", 999.0)):
                    is_rejected = True

            self.decision_buffer.append("REJECTED" if is_rejected else best_class.upper())

            if is_rejected:
                predicted_label = "Unknown Gesture"
                confidence = best_conf
            else:
                # Temporal consensus
                agree = list(self.decision_buffer).count(best_class.upper())
                if agree >= 3:
                    predicted_label = best_class.upper()
                    confidence = best_conf
                else:
                    predicted_label = "Signing..."
                    confidence = best_conf

        elif not any(self.hand_detected_buffer):
            self.prediction_buffer.clear()
            self.decision_buffer.clear()
            predicted_label = "No Hands Detected"

        return predicted_label, confidence, probabilities, hands_found

    def render_overlay(self, frame: np.ndarray, label: str, conf: float, probs: np.ndarray, hands_count: int, fps: float):
        """Draws visual HUD, prediction banners, and confidence bars on the video frame."""
        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if label == "Unknown Gesture":
            status_color = (0, 140, 255)  # Amber
        elif conf >= self.confidence_threshold and label not in ["Waiting for Gesture...", "Signing...", "Hands Ready - Perform Sign"]:
            status_color = (0, 215, 255) if label in ["HELP", "ACCIDENT", "CALL", "DOCTOR"] else (0, 255, 0)
        else:
            status_color = (180, 180, 180)

        cv2.putText(frame, f"SIGN: {label}", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.1, status_color, 2, cv2.LINE_AA)

        conf_str = f"Confidence: {conf * 100:.1f}%" if conf > 0 else "Confidence: --"
        cv2.putText(frame, conf_str, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

        buffer_info = f"Buffer: {len(self.frame_buffer)}/{SEQUENCE_LENGTH} | Hands: {hands_count} | FPS: {fps:.1f}"
        cv2.putText(frame, buffer_info, (w - 380, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 'Q' to Exit", (w - 180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 255), 2, cv2.LINE_AA)

        bar_x, bar_y, bar_w, bar_h = 240, 68, 180, 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
        fill_w = int(bar_w * min(1.0, conf))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)

        panel_w = 210
        cv2.rectangle(overlay, (w - panel_w - 10, 110), (w - 10, 110 + len(self.class_names) * 24 + 10), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        for idx, (cname, pval) in enumerate(zip(self.class_names, probs)):
            y_pos = 132 + idx * 24
            col = (0, 255, 120) if (cname.upper() == label and conf >= self.confidence_threshold) else (220, 220, 220)
            cv2.putText(frame, f"{cname[:8]:8s} : {pval * 100:4.1f}%", (w - panel_w, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

    def run_live(self, camera_index: int = 0):
        """Starts live webcam recognition loop."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam at index {camera_index}.")
            return

        print("=" * 65)
        print("REAL-TIME ISL EMERGENCY GESTURE RECOGNITION ACTIVE")
        print("=" * 65)
        print("Features: Hand-Independent + Unknown Gesture Rejection")
        print("Press 'Q' in the video window to quit.")
        print("=" * 65)

        prev_time = time.time()
        fps = 0.0

        try:
            while cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret:
                    print("[WARNING] Frame capture failed.")
                    break

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else fps
                prev_time = curr_time

                label, conf, probs, hands_count = self.predict_frame(raw_frame)

                if conf >= self.confidence_threshold:
                    now = time.time()
                    if label != getattr(self, "_last_term_label", None) or (now - getattr(self, "_last_term_time", 0.0)) > 1.8:
                        self._last_term_label = label
                        self._last_term_time = now
                        print(f"[{time.strftime('%H:%M:%S')}] >>> FINAL OUTPUT: [ {label} ] (Confidence: {conf * 100:.1f}%) | Hands: {hands_count}")

                display_frame = cv2.flip(raw_frame, 1)
                self.render_overlay(display_frame, label, conf, probs, hands_count, fps)

                cv2.imshow("ISL Emergency Gesture Recognizer (LSTM)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("Exiting real-time recognition loop.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = RealTimeGestureRecognizer()
    recognizer.run_live(camera_index=0)
