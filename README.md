# 🤟 LSTM-Based Indian Sign Language (ISL) Emergency Gesture Recognition

An end-to-end deep learning framework for recognizing **8 Indian Sign Language (ISL) emergency gestures** in real time from video and live webcam feeds. Built following the enhanced methodology of:

> **"LSTM-Based Recognition of Sign Language" (IC3 2024)**

---

## ⚡ Quick Start: Live Webcam Recognition

To launch the real-time recognition interface with dual-hand skeletal visualization:
```bash
python live.py
```
### Live Interface Highlights:
- **Dual Hands Tracked & Rendered**: Tracks **both Left and Right hands** simultaneously (21 skeletal joints each, 126 features total).
- **Correct Sensor Coordinates**: MediaPipe landmark detection is run on the unmirrored raw camera feed, ensuring 100% mathematical alignment with the training dataset coordinate frame.
- **Natural Mirror Display**: The video window is intuitively mirrored for the user, with hand skeletons and joint labels mapped pixel-perfectly onto the user's hands.
- **Dual Recognition Modes**:
  1. **Spacebar Gesture Recording Mode (`[SPACE]`)**: Press `SPACE` to start a 1-second countdown ("3.. 2.. 1.. SIGN!"), record an exact 2.0-second gesture (matching the training video duration), and display the predicted word in a prominent green victory card!
  2. **Continuous Live Streaming Mode**: Rolling buffer with dynamic motion energy filtering so resting hands don't trigger false predictions.
- **Interactive Hotkeys**:
  - `[SPACE]` : Trigger 2.0-second gesture recording window
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

---

## 📊 Performance & Evaluation Metrics

Evaluated on an independent, unseen test dataset ($N = 122$ samples):

| Metric | Score |
| :--- | :---: |
| **Overall Test Accuracy** | **95.90%** |
| **Weighted Precision** | **0.9606** |
| **Weighted Recall** | **0.9590** |
| **Weighted F1-Score** | **0.9587** |
| **Macro F1-Score** | **0.9585** |

### Per-Class Test Set Performance:
| Gesture Word | Precision | Recall | F1-Score | Test Samples |
| :--- | :---: | :---: | :---: | :---: |
| **accident** | **1.0000** | **1.0000** | **1.0000** | 16 |
| **call** | **1.0000** | 0.8667 | **0.9286** | 15 |
| **doctor** | 0.9375 | **1.0000** | **0.9677** | 15 |
| **help** | 0.9375 | 0.9375 | **0.9375** | 16 |
| **hot** | 0.9333 | 0.9333 | **0.9333** | 15 |
| **lose** | 0.9375 | **1.0000** | **0.9677** | 15 |
| **pain** | 0.9375 | **1.0000** | **0.9677** | 15 |
| **thief** | **1.0000** | 0.9333 | **0.9655** | 15 |

---

## 🏗️ Deep Learning Architecture

```text
Input Sequence: (30 frames, 126 coordinates: 2 hands × 21 landmarks × 3 coords)
                       │
                       ▼
Bidirectional LSTM (128 units, return_sequences=True) ➔ Output: (30, 256)
                       │
                       ▼
LayerNormalization + Dropout (0.30)
                       │
                       ▼
Bidirectional LSTM (64 units, return_sequences=True)  ➔ Output: (30, 128)
                       │
                       ▼
LayerNormalization + Dropout (0.30)
                       │
                       ▼
GlobalAveragePooling1D (Temporal Aggregation)         ➔ Output: (128,)
                       │
                       ▼
Dense (128 units, ReLU) + BatchNormalization + Dropout (0.30)
                       │
                       ▼
Dense (64 units, ReLU) + Dropout (0.20)
                       │
                       ▼
Dense (8 units, Softmax)                              ➔ Output: (8 probabilities)
```

---

## 📁 Repository Layout

```text
ISL-Recognition/
├── live.py                               # One-command live webcam recognizer with dual-hand overlay
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
├── ISL_Recognition_preprocessing.ipynb   # Interactive video preprocessing notebook
├── ISL_Recognition_training.ipynb        # Interactive model training notebook
├── training/
│   ├── dataset_loader.py                 # Dataset loading, validation, and stratified splitting
│   ├── train.py                          # High-accuracy BiLSTM training script
│   └── evaluate.py                       # Test set evaluation and metric generation
├── realtime/
│   ├── predict.py                        # Standalone webcam recognizer module
│   └── predict_video.py                  # Offline single-video prediction and annotation
├── models/
│   ├── isl_lstm_model.keras              # Best trained BiLSTM model checkpoint
│   ├── label_map.json                    # Gesture label mapping dictionary
│   └── dataset_splits.npz                # Stratified train/val/test split indices
├── results/
│   ├── accuracy_plot.png                 # Accuracy progression curve
│   ├── loss_plot.png                     # Loss progression curve
│   ├── training_curves.png               # Side-by-side training curves
│   ├── confusion_matrix.png              # Test set confusion matrix heatmap
│   ├── classification_report.txt         # Precision/Recall text report
│   ├── evaluation_metrics.json           # Evaluation metrics JSON
│   └── training_history.json             # Epoch-by-epoch training history
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
python training/train.py --batch_size 32 --epochs 60 --lr 0.001 --patience 15
```

### 3. Evaluate Model on Test Dataset
```bash
python training/evaluate.py
```

### 4. Test on a Recorded Video File
```bash
python realtime/predict_video.py --video_path "path/to/video.mp4"
```
