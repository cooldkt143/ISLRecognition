"""
Offline Video Gesture Recognizer for ISL Emergency Signs.
Processes a recorded video file (.mp4, .avi, etc.), predicts the gesture,
prints confidence scores, and optionally saves an annotated output video.
"""

import os
import sys
import argparse
import json
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

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.predict import RealTimeGestureRecognizer
from training.dataset_loader import EXPECTED_SHAPE


def predict_video_file(
    video_path: str,
    output_video_path: str = None,
    confidence_threshold: float = 0.70
):
    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"Video file not found at: {video_file.resolve()}")

    recognizer = RealTimeGestureRecognizer(confidence_threshold=confidence_threshold)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_file.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_video_path:
        out_path = Path(output_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print("=" * 65)
    print(f"Processing Video: {video_file.name}")
    print(f"Resolution      : {width}x{height} @ {fps:.1f} FPS")
    print(f"Total Frames    : {total_frames}")
    print("=" * 65)

    frame_idx = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        label, conf, probs, hands_count = recognizer.predict_frame(frame)
        predictions.append((label, conf, probs.copy()))

        if writer:
            annotated_frame = frame.copy()
            recognizer.render_overlay(annotated_frame, label, conf, probs, hands_count, fps)
            writer.write(annotated_frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to: {Path(output_video_path).resolve()}")

    # Determine final video-level prediction
    # Aggregate probabilities across all frames where hands were detected
    valid_probs = [p[2] for p in predictions if np.max(p[2]) > 0]
    if valid_probs:
        avg_probs = np.mean(valid_probs, axis=0)
        best_class_idx = int(np.argmax(avg_probs))
        best_class = recognizer.class_names[best_class_idx]
        best_conf = float(avg_probs[best_class_idx])

        print("\n" + "=" * 65)
        print("VIDEO PREDICTION RESULT")
        print("=" * 65)
        print(f"Predicted Sign Gesture : {best_class.upper()}")
        print(f"Aggregated Confidence  : {best_conf * 100:.2f}%")
        print("-" * 65)
        print("Per-class probabilities:")
        for idx, (cname, prob) in enumerate(zip(recognizer.class_names, avg_probs)):
            print(f"  {cname:10s} : {prob * 100:5.2f}%")
        print("=" * 65)
        return best_class, best_conf, avg_probs
    else:
        print("[WARNING] No hands detected in video sequence.")
        return "UNKNOWN", 0.0, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ISL emergency sign from video file.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file (.mp4, .avi)")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save annotated output video")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    args = parser.parse_args()

    predict_video_file(args.video_path, args.output_path, args.threshold)
