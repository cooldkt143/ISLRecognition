"""
Comprehensive Unit & Integration Test Suite for ISL Emergency Gesture Recognition.
Tests:
1. Dataset file integrity and shape consistency
2. Label mapping completeness
3. Keras model loading, output shapes, and inference latency
4. Wrist-anchored coordinate normalization invariance and zero-preservation
5. Live recognizer engine and bottom HUD canvas rendering
6. Left-Hand & Right-Hand Sagittal Mirror Invariance
7. Multi-tier Unknown Gesture Rejection Engine
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
        """Verifies label_map.json exists and maps all 8 emergency words correctly."""
        map_path = PROJECT_ROOT / "models" / "label_map.json"
        self.assertTrue(map_path.is_file(), f"label_map.json missing at {map_path}")

        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("label2id", data)
        self.assertIn("id2label", data)
        self.assertGreaterEqual(len(data["label2id"]), len(EXPECTED_CLASSES))

        for cname in EXPECTED_CLASSES:
            self.assertIn(cname, data["label2id"])
            cid = data["label2id"][cname]
            self.assertEqual(data["id2label"][str(cid)], cname)

        print(f"  [PASS] test_02_label_mapping: Label map correctly maps {len(data['label2id'])} classes.")

    def test_03_model_loading_and_latency(self):
        """Verifies model loading, forward pass shape, and real-time inference latency."""
        model_path = PROJECT_ROOT / "models" / "isl_lstm_model.keras"
        self.assertTrue(model_path.is_file(), f"Trained model missing at {model_path}")

        model = keras.models.load_model(str(model_path))
        self.assertEqual(model.input_shape, (None, 30, 126))
        self.assertIn(model.output_shape, [(None, 8), (None, 9)])

        # Benchmark forward pass latency
        dummy_seq = np.random.randn(1, 30, 126).astype(np.float32)
        _ = model.predict(dummy_seq, verbose=0)

        latencies = []
        for _ in range(20):
            t0 = time.time()
            probs = model.predict(dummy_seq, verbose=0)[0]
            latencies.append((time.time() - t0) * 1000)

        avg_latency = np.mean(latencies)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        self.assertLess(avg_latency, 50.0, f"Inference latency too high: {avg_latency:.2f} ms")
        print(f"  [PASS] test_03_model_loading_and_latency: Output shape {model.output_shape}, avg latency = {avg_latency:.2f} ms.")

    def test_04_wrist_normalization_invariance(self):
        """Tests that relative wrist normalization ensures translation invariance and preserves absent hands."""
        from live import LiveISLRecognizer

        # Synthetic landmark sequence with only Right hand present
        seq = np.zeros(126, dtype=np.float32)
        # Right wrist at (0.5, 0.5, 0.1)
        seq[63] = 0.5
        seq[64] = 0.5
        seq[65] = 0.1
        # Right index tip at (0.6, 0.7, 0.2)
        seq[63 + 8 * 3] = 0.6
        seq[64 + 8 * 3] = 0.7
        seq[65 + 8 * 3] = 0.2

        # Left hand is absent: should stay STRICTLY 0.0
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
        # Absent left hand must remain strictly zero (NO phantom landmark translation)
        self.assertTrue(np.all(norm1[:63] == 0.0))
        self.assertTrue(np.all(norm2[:63] == 0.0))
        print("  [PASS] test_04_wrist_normalization_invariance: Relative anchoring & zero preservation verified.")

    def test_05_live_recognizer_pipeline(self):
        """Verifies LiveISLRecognizer instantiation, frame buffering, and HUD rendering."""
        from live import LiveISLRecognizer

        recognizer = LiveISLRecognizer(confidence_threshold=0.70)
        self.assertGreaterEqual(len(recognizer.class_names), 8)

        # Feed blank frame (no hands detected)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        word, conf, probs, has_left, has_right = recognizer.predict_frame(dummy_frame)

        self.assertEqual(word, "No Hands in View")
        self.assertEqual(conf, 0.0)
        self.assertFalse(has_left)
        self.assertFalse(has_right)

        # Test HUD canvas assembly (camera + bottom panel)
        full_canvas = recognizer.render_hud(dummy_frame, word, conf, probs, has_left, has_right, 30.0)
        self.assertEqual(full_canvas.shape, (480 + recognizer.bottom_panel_height, 640, 3))
        print("  [PASS] test_05_live_recognizer_pipeline: Live recognizer and bottom HUD canvas verified.")

    def test_06_hand_swapping_invariance(self):
        """Verifies that Left-Hand and Right-Hand versions of a target sign are both recognized."""
        from training.train import mirror_sequence
        from live import LiveISLRecognizer

        recognizer = LiveISLRecognizer(confidence_threshold=0.60)
        sample_path = PROJECT_ROOT / "data" / "accident" / "accident_Crop  (1).npy"
        if not sample_path.is_file():
            # Grab any existing sample in accident folder
            sample_path = list((PROJECT_ROOT / "data" / "accident").glob("*.npy"))[0]

        seq = np.load(sample_path)
        # Canonical sequence
        word_canon, conf_canon, _, is_valid_c, _ = recognizer.classify_and_verify_sequence(seq)

        # Sagittal-mirrored sequence (opposite hand execution)
        seq_mirr = mirror_sequence(np.expand_dims(seq, 0))[0]
        word_mirr, conf_mirr, _, is_valid_m, _ = recognizer.classify_and_verify_sequence(seq_mirr)

        self.assertEqual(word_canon, "ACCIDENT")
        self.assertEqual(word_mirr, "ACCIDENT")
        self.assertTrue(is_valid_c)
        self.assertTrue(is_valid_m)
        print(f"  [PASS] test_06_hand_swapping_invariance: Canonical '{word_canon}' ({conf_canon*100:.1f}%) and Mirrored '{word_mirr}' ({conf_mirr*100:.1f}%) both recognized.")

    def test_07_unknown_gesture_rejection(self):
        """Verifies that non-emergency, random, and resting hand gestures are rejected as 'Unknown Gesture'."""
        from training.dataset_loader import generate_negative_samples
        from live import LiveISLRecognizer

        recognizer = LiveISLRecognizer(confidence_threshold=0.70)
        # Generate 20 diverse negative sequences (resting, waving, random walk)
        neg_samples = generate_negative_samples(n_samples=20, random_state=1234)

        rejected_count = 0
        for i in range(len(neg_samples)):
            seq = neg_samples[i]
            word, conf, probs, is_valid, reason = recognizer.classify_and_verify_sequence(seq)
            if not is_valid or word in ["Unknown Gesture", "Hands Ready - Perform Sign"]:
                rejected_count += 1

        rejection_rate = (rejected_count / len(neg_samples)) * 100.0
        self.assertGreaterEqual(rejection_rate, 85.0, f"Unknown rejection rate too low: {rejection_rate:.1f}%")
        print(f"  [PASS] test_07_unknown_gesture_rejection: Rejection rate = {rejection_rate:.1f}% ({rejected_count}/20 rejected).")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING ISL EMERGENCY GESTURE RECOGNITION TEST SUITE")
    print("=" * 65)
    unittest.main(verbosity=2)
