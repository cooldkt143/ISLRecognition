"""
Comprehensive Unit & Integration Test Suite for ISL Emergency Gesture Recognition.
Tests:
1. Dataset file integrity and shape consistency
2. Label mapping completeness
3. Keras model loading, output shapes, and inference latency
4. Wrist-anchored coordinate normalization invariance
5. Live recognizer engine and HUD overlay rendering
6. Offline video predictor functionality
"""

import os
import sys
import time
import json
import unittest
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_CLASSES = ["accident", "call", "doctor", "help", "hot", "lose", "pain", "thief"]
EXPECTED_SHAPE = (30, 126)


class TestISLPipeline(unittest.TestCase):

    def test_01_dataset_files_and_shapes(self):
        """Verifies that all 8 class directories exist and contain valid (30, 126) sequences."""
        data_dir = PROJECT_ROOT / "data"
        self.assertTrue(data_dir.is_dir(), f"data/ directory does not exist at {data_dir}")

        total_files = 0
        for class_name in EXPECTED_CLASSES:
            class_folder = data_dir / class_name
            self.assertTrue(class_folder.is_dir(), f"Class folder missing: {class_name}")

            npy_files = list(class_folder.glob("*.npy"))
            self.assertGreaterEqual(len(npy_files), 100, f"Expected at least 100 files for {class_name}, got {len(npy_files)}")
            total_files += len(npy_files)

            # Check a sample of 10 files per class
            for fpath in npy_files[:10]:
                arr = np.load(fpath)
                self.assertEqual(arr.shape, EXPECTED_SHAPE, f"Wrong shape in {fpath.name}: {arr.shape}")
                self.assertTrue(np.isfinite(arr).all(), f"NaN or Inf in {fpath.name}")

        self.assertGreaterEqual(total_files, 800, f"Expected >= 800 total files, found {total_files}")
        print(f"  [PASS] test_01_dataset_files_and_shapes: {total_files} valid .npy files verified.")

    def test_02_label_mapping(self):
        """Verifies label_map.json exists and maps all 8 words correctly."""
        map_path = PROJECT_ROOT / "models" / "label_map.json"
        self.assertTrue(map_path.is_file(), f"label_map.json missing at {map_path}")

        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("label2id", data)
        self.assertIn("id2label", data)
        self.assertEqual(len(data["label2id"]), len(EXPECTED_CLASSES))

        for cname in EXPECTED_CLASSES:
            self.assertIn(cname, data["label2id"])
            cid = data["label2id"][cname]
            self.assertEqual(data["id2label"][str(cid)], cname)

        print("  [PASS] test_02_label_mapping: Label map correctly maps 8 classes.")

    def test_03_model_loading_and_latency(self):
        """Verifies model loading, forward pass shape, and real-time inference latency."""
        model_path = PROJECT_ROOT / "models" / "isl_lstm_model.keras"
        self.assertTrue(model_path.is_file(), f"Trained model missing at {model_path}")

        model = keras.models.load_model(str(model_path))
        self.assertEqual(model.input_shape, (None, 30, 126))
        self.assertEqual(model.output_shape, (None, 8))

        # Benchmark forward pass latency
        dummy_seq = np.random.randn(1, 30, 126).astype(np.float32)
        # Warmup
        _ = model.predict(dummy_seq, verbose=0)

        latencies = []
        for _ in range(20):
            t0 = time.time()
            probs = model.predict(dummy_seq, verbose=0)[0]
            latencies.append((time.time() - t0) * 1000)

        avg_latency = np.mean(latencies)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        self.assertLess(avg_latency, 50.0, f"Inference latency too high: {avg_latency:.2f} ms")
        print(f"  [PASS] test_03_model_loading_and_latency: Output shape (1, 8), avg latency = {avg_latency:.2f} ms.")

    def test_04_wrist_normalization_invariance(self):
        """Tests that relative wrist normalization ensures translation invariance."""
        from live import LiveISLRecognizer

        # Synthetic landmark sequence
        seq = np.zeros(126, dtype=np.float32)
        # Right wrist at (0.5, 0.5, 0.1)
        seq[63] = 0.5
        seq[64] = 0.5
        seq[65] = 0.1
        # Right index tip at (0.6, 0.7, 0.2)
        seq[63 + 8 * 3] = 0.6
        seq[64 + 8 * 3] = 0.7
        seq[65 + 8 * 3] = 0.2

        # Apply offset to simulate person sitting elsewhere in camera view
        offset = np.array([0.2, -0.3, 0.05], dtype=np.float32)
        seq_shifted = seq.copy()
        for j in range(63, 126, 3):
            seq_shifted[j:j+3] += offset

        norm1 = LiveISLRecognizer.normalize_frame(seq)
        norm2 = LiveISLRecognizer.normalize_frame(seq_shifted)

        # Right wrist should be at (0, 0, 0)
        np.testing.assert_allclose(norm1[63:66], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(norm2[63:66], [0.0, 0.0, 0.0], atol=1e-6)
        # Shifted index finger should match original index finger relative position
        np.testing.assert_allclose(norm1[63 + 8 * 3 : 66 + 8 * 3], norm2[63 + 8 * 3 : 66 + 8 * 3], atol=1e-6)
        print("  [PASS] test_04_wrist_normalization_invariance: Relative anchoring verified.")

    def test_05_live_recognizer_pipeline(self):
        """Verifies LiveISLRecognizer instantiation, frame buffering, and HUD rendering."""
        from live import LiveISLRecognizer

        recognizer = LiveISLRecognizer(confidence_threshold=0.70)
        self.assertEqual(len(recognizer.class_names), 8)

        # Feed 30 blank frames (no hands detected)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        word, conf, probs, has_left, has_right = recognizer.predict_frame(dummy_frame)

        self.assertEqual(word, "No Hands in View")
        self.assertEqual(conf, 0.0)
        self.assertFalse(has_left)
        self.assertFalse(has_right)

        # Test HUD rendering
        recognizer.render_hud(dummy_frame, word, conf, probs, has_left, has_right, 30.0)
        self.assertEqual(dummy_frame.shape, (480, 640, 3))
        print("  [PASS] test_05_live_recognizer_pipeline: Live recognizer and HUD overlay verified.")

    def test_06_offline_video_prediction(self):
        """Verifies offline video prediction on an actual dataset clip."""
        from realtime.predict_video import predict_video_file

        video_path = PROJECT_ROOT / "ISL_Emergency_Sign_dataset" / "accident_Cropped" / "accident_Crop  (1).mp4"
        if video_path.is_file():
            predicted_word, confidence, probs = predict_video_file(str(video_path), confidence_threshold=0.70)
            self.assertEqual(predicted_word, "accident")
            self.assertGreater(confidence, 0.85)
            print(f"  [PASS] test_06_offline_video_prediction: Correctly predicted '{predicted_word}' ({confidence*100:.1f}%).")
        else:
            print("  [SKIP] test_06_offline_video_prediction: Sample video not found.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING ISL EMERGENCY GESTURE RECOGNITION TEST SUITE")
    print("=" * 65)
    unittest.main(verbosity=2)
