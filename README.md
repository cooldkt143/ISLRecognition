# 🤟 Hand-Independent ISL Emergency Gesture Recognition with Unknown Rejection

An end-to-end deep learning framework for recognizing **8 Indian Sign Language (ISL) emergency gestures** in real time from video and live webcam feeds. Built following the enhanced methodology of:

> **"LSTM-Based Recognition of Sign Language" (IC3 2024)**

Enhanced with:
1. **Left-Hand & Right-Hand Independence**: True bilateral sign language recognition via sagittal reflection augmentation.
2. **Multi-Tier Unknown Gesture Rejection**: Rejects random movements, unfamiliar poses, and out-of-distribution gestures instead of forced classification.
3. **Dedicated Bottom Output Dashboard**: Displays the final recognized sign, confidence meter, sensor status, and top candidates in a dedicated console panel below the camera feed.
4. **Synchronized Terminal Logging**: Formatted terminal output with timestamps, prediction confidence, sensor diagnostics, and candidate breakdown.

---

## ⚡ Quick Start: Live Webcam Recognition

To launch the real-time recognition interface:
```bash
python live.py
```

### Hotkeys:
- `[SPACE]` : Trigger 2.0-second gesture recording window (recommended for precise signs)
- `[F]`     : Toggle hand inversion (swaps Left/Right slots if webcam driver pre-mirrors)
- `[C]`     : Clear current result and reset buffer
- `[Q]` / `ESC` : Exit program

---

## 📌 Recognized Emergency Sign Vocabulary (8 Words):
- `accident` 🚨
- `call` 📞
- `doctor` 🩺
- `help` 🆘
- `hot` 🔥
- `lose` 📉
- `pain` 🤕
- `thief` 🦹

Plus an explicit rejection state:
- `Unknown Gesture` / `No Recognized Sign`

---

## 📊 Comprehensive Model Performance & Evaluation Metrics

Evaluated on independent, unseen test datasets:

| Metric | Score |
| :--- | :---: |
| **Standard Test Accuracy (Right-Handed)** | **97.26%** |
| **Mirrored Test Accuracy (Left-Handed)** | **97.54%** |
| **Unknown Gesture Rejection Rate** | **100.00%** (120/120 rejected) |
| **Weighted Precision** | **0.9750** |
| **Weighted Recall** | **0.9726** |
| **Weighted F1-Score** | **0.9725** |
| **Macro F1-Score** | **0.9707** |
| **Average Inference Latency** | **37.6 ms** |

### Per-Class Performance:
| Gesture Word | Precision | Recall | F1-Score | Right-Hand Acc | Left-Hand Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **accident** | 0.9412 | **1.0000** | **0.9697** | **100.0%** | **100.0%** |
| **call** | **1.0000** | 0.8667 | **0.9286** | 86.7% | 86.7% |
| **doctor** | **1.0000** | 0.9333 | **0.9655** | 93.3% | **100.0%** |
| **help** | 0.9412 | **1.0000** | **0.9697** | **100.0%** | **100.0%** |
| **hot** | **1.0000** | 0.9333 | **0.9655** | 93.3% | 93.3% |
| **lose** | **1.0000** | **1.0000** | **1.0000** | **100.0%** | **100.0%** |
| **pain** | 0.8824 | **1.0000** | **0.9375** | **100.0%** | **100.0%** |
| **thief** | **1.0000** | **1.0000** | **1.0000** | **100.0%** | **100.0%** |
| **unknown** | **1.0000** | **1.0000** | **1.0000** | **100.0%** | N/A |

---

## 🛡️ Multi-Tier Unknown Gesture Rejection Engine

1. **Motion Velocity Gating**: Filters static resting hands ($\Delta_{vel} < 0.0025$) to prevent resting hands from triggering false predictions.
2. **Negative Unknown Class**: Trained on synthetic negative samples (idle resting hands, circular waving, thumbs up, random walk).
3. **Softmax Confidence Threshold**: Requires $P_{(1)} \ge 0.70$.
4. **Confidence Margin Threshold**: Requires $P_{(1)} - P_{(2)} \ge 0.25$ to reject ambiguous hand movements between two classes.
5. **Latent Class Prototype Distance Gating**: Extracts embedding from `dense_64` and verifies $d(z, c_k) \le \tau_k$. If out-of-distribution, rejects as `Unknown Gesture`.
6. **Temporal Consensus Filter**: Requires $\ge 3$ agreeing consecutive predictions before committing in continuous mode.

---

## 📁 Repository Layout

```text
ISL-Recognition/
├── live.py                               # Live webcam recognizer with dual-hand overlay & bottom dashboard
├── data/                                 # 808 preprocessed .npy gesture sequences
│   ├── accident/                         # 102 sequences (30, 126)
│   ├── call/                             # 102 sequences (30, 126)
│   ├── doctor/                           # 102 sequences (30, 126)
│   ├── help/                             # 102 sequences (30, 126)
│   ├── hot/                              # 100 sequences (30, 126)
│   ├── lose/                             # 100 sequences (30, 126)
│   ├── pain/                             # 100 sequences (30, 126)
│   └── thief/                            # 100 sequences (30, 126)
├── ISL_Emergency_Sign_dataset/           # Raw video recordings
├── hand_landmarker.task                  # MediaPipe HandLandmarker asset
├── training/
│   ├── dataset_loader.py                 # Dataset loading, negative sample generator, and splitting
│   ├── train.py                          # Hand-independent BiLSTM training script
│   └── evaluate.py                       # Comprehensive 3-part evaluation script
├── realtime/
│   ├── predict.py                        # Real-time webcam recognizer module
│   └── predict_video.py                  # Offline single-video prediction and annotation
├── models/
│   ├── isl_lstm_model.keras              # Best trained BiLSTM model checkpoint
│   ├── label_map.json                    # 9-class label mapping dictionary
│   ├── class_prototypes.json             # Latent class prototypes & OOD thresholds
│   └── dataset_splits.npz                # Stratified train/val/test split indices
├── results/
│   ├── accuracy_plot.png                 # Accuracy progression curve
│   ├── loss_plot.png                     # Loss progression curve
│   ├── training_curves.png               # Side-by-side training curves
│   ├── confusion_matrix.png              # Test set confusion matrix heatmap
│   ├── classification_report.txt         # Comprehensive classification report
│   ├── evaluation_metrics.json           # Evaluation metrics JSON
│   └── training_history.json             # Epoch-by-epoch training history
├── tests/
│   └── test_pipeline.py                  # 7 unit and integration tests (100% pass)
└── requirements.txt                      # Project dependency specification
```

---

## 🛠️ Step-by-Step Terminal Execution

### 1. Run Live Webcam Sign Recognition
```bash
python live.py
```

### 2. Retrain Model
```bash
python training/train.py --batch_size 32 --epochs 65 --patience 16
```

### 3. Evaluate Model on Test Dataset
```bash
python training/evaluate.py
```

### 4. Run Comprehensive Test Suite
```bash
python tests/test_pipeline.py
```

### 5. Test on a Recorded Video File
```bash
python realtime/predict_video.py --video_path "ISL_Emergency_Sign_dataset/accident_Cropped/accident_Crop  (1).mp4"
```
