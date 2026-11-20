"""
Model Training Script for ISL Emergency Gesture Recognition.
Trains a high-accuracy Bidirectional LSTM model with Feature Projection, LayerNormalization,
Dual Temporal Pooling (Average + Max), Dropout, and multi-factor Data Augmentation.
Saves the best model checkpoint to models/isl_lstm_model.keras and history to results/.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Configure backend seamlessly
if "KERAS_BACKEND" not in os.environ:
    try:
        import tensorflow
        os.environ["KERAS_BACKEND"] = "tensorflow"
    except ImportError:
        os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models, callbacks, optimizers
import numpy as np

# Ensure training package is in path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_loader import (
    load_dataset,
    split_dataset,
    save_label_mapping,
    EXPECTED_SHAPE
)


def augment_training_data(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Applies multi-factor data augmentation to improve real-world generalization:
    1. Sensor landmark jitter (Gaussian noise on active coordinates)
    2. Hand size scaling (0.92 - 1.08)
    3. Temporal shifting (+/- 1-2 frames)
    4. Speed warping (faster / slower signs resampled to 30 frames)
    5. Frame dropout simulation (robustness against webcam detector flicker)
    """
    np.random.seed(random_state)
    N, T, D = X_train.shape

    # 1. Landmark Jitter (Gaussian noise on non-zero landmarks)
    noise = np.random.normal(0, 0.005, X_train.shape).astype(np.float32)
    mask = (X_train != 0).astype(np.float32)
    X_jitter = X_train + (noise * mask)

    # 2. Hand Scale Scaling (0.92 to 1.08)
    scales = np.random.uniform(0.92, 1.08, (N, 1, 1)).astype(np.float32)
    X_scale = X_train * scales

    # 3. Temporal Shifting (+/- 1-2 frames)
    X_shift = np.zeros_like(X_train)
    for i in range(N):
        shift = np.random.randint(-2, 3)
        if shift > 0:
            X_shift[i, shift:] = X_train[i, :-shift]
            X_shift[i, :shift] = X_train[i, 0]
        elif shift < 0:
            X_shift[i, :shift] = X_train[i, -shift:]
            X_shift[i, shift:] = X_train[i, -1]
        else:
            X_shift[i] = X_train[i]

    # 4. Speed Warping (Resampling random sub-sequences or stretched sequences)
    X_warp = np.zeros_like(X_train)
    for i in range(N):
        # Stretch or compress between 25 and 35 frames
        warp_len = np.random.randint(25, 36)
        src_indices = np.linspace(0, T - 1, warp_len)
        resampled_indices = np.linspace(0, warp_len - 1, T).astype(int)
        interp_seq = np.zeros((warp_len, D), dtype=np.float32)
        for d in range(D):
            interp_seq[:, d] = np.interp(src_indices, np.arange(T), X_train[i, :, d])
        X_warp[i] = interp_seq[resampled_indices]

    # Combine all augmentations (5x expansion)
    X_aug = np.concatenate([X_train, X_jitter, X_scale, X_shift, X_warp], axis=0)
    y_aug = np.concatenate([y_train, y_train, y_train, y_train, y_train], axis=0)

    # 5. Intermittent detector flicker dropout on 10% of augmented samples
    flicker_mask = np.random.rand(len(X_aug)) < 0.10
    for idx in np.where(flicker_mask)[0]:
        drop_frame = np.random.randint(0, T)
        X_aug[idx, drop_frame, :] = 0.0

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_aug))
    return X_aug[shuffle_idx], y_aug[shuffle_idx]


def build_bilstm_model(
    sequence_length: int = 30,
    feature_dim: int = 126,
    num_classes: int = 8,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Builds an optimized Deep BiLSTM model with:
    - Spatial Joint Projection: Dense(128) + LayerNorm to model inter-finger spatial relations
    - BiLSTM Layer 1: 128 units (256 per step) + LayerNorm + Dropout
    - BiLSTM Layer 2: 96 units (192 per step) + LayerNorm + Dropout
    - Dual Temporal Pooling: Concatenated GlobalAveragePooling1D + GlobalMaxPooling1D (384 units)
    - Dense Classification Head: 128 (ReLU) + BatchNorm + Dropout(0.35) -> 64 (ReLU) -> Softmax(8)
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="input_sequence")

    # 1. Spatial feature projection
    x = layers.Dense(128, activation="relu", name="spatial_dense_projection")(inputs)
    x = layers.LayerNormalization(name="spatial_layer_norm")(x)
    x = layers.Dropout(0.2, name="spatial_dropout")(x)

    # 2. Recurrent temporal layers (Stacked BiLSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="bilstm_1")(x)
    x = layers.LayerNormalization(name="layer_norm_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True), name="bilstm_2")(x)
    x = layers.LayerNormalization(name="layer_norm_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # 3. Dual Temporal Pooling (captures both overall gesture flow and peak contact points)
    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    pooled = layers.Concatenate(name="dual_temporal_pooling")([avg_pool, max_pool])

    # 4. Dense Classification Head
    d1 = layers.Dense(128, activation="relu", name="dense_128")(pooled)
    d1 = layers.BatchNormalization(name="batch_norm_128")(d1)
    d1 = layers.Dropout(0.35, name="dropout_128")(d1)

    d2 = layers.Dense(64, activation="relu", name="dense_64")(d1)
    d2 = layers.Dropout(0.20, name="dropout_64")(d2)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_classes")(d2)

    model = models.Model(inputs=inputs, outputs=outputs, name="ISL_BiLSTM_Fortified")
    return model


def train_pipeline(
    batch_size: int = 32,
    max_epochs: int = 70,
    learning_rate: float = 0.001,
    patience: int = 18,
    random_state: int = 42
):
    print("=" * 65)
    print("ISL Emergency Gesture Recognition - Complete High-Accuracy Training")
    print("=" * 65)
    print(f"Keras Version   : {keras.__version__}")
    print(f"Keras Backend   : {keras.backend.backend()}")

    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and split dataset
    X, y, class_names, label_map, _ = load_dataset(data_dir, verbose=True)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), splits = split_dataset(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        random_state=random_state, verbose=True
    )

    # Save label map and test splits
    save_label_mapping(label_map, models_dir / "label_map.json")
    np.savez(
        models_dir / "dataset_splits.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print(f"Saved dataset split cache to: {models_dir / 'dataset_splits.npz'}")

    # 2. Apply Data Augmentation to Training Set
    print("\nApplying Multi-Factor Sequence Augmentation...")
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train, random_state=random_state)
    print(f"  Base Training Samples     : {len(X_train)}")
    print(f"  Augmented Training Samples: {len(X_train_aug)} (5x expansion)")

    # 3. Build model architecture
    sequence_length = X.shape[1]
    feature_dim = X.shape[2]
    num_classes = len(class_names)

    model = build_bilstm_model(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout_rate=0.3
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel Architecture Summary:")
    model.summary()

    # 4. Configure callbacks
    checkpoint_path = models_dir / "isl_lstm_model.keras"
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 5. Train model
    print(f"\nStarting complete training (Batch Size: {batch_size}, Max Epochs: {max_epochs}, LR: {learning_rate})...")
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=training_callbacks,
        verbose=1
    )

    # 6. Save training history
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    history_file = results_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)
    print(f"\nSaved training history to: {history_file.resolve()}")
    print(f"Best model successfully saved to: {checkpoint_path.resolve()}")

    return model, history_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train high-accuracy BiLSTM model for ISL emergency gestures.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=70, help="Maximum epochs (default: 70)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=18, help="Early stopping patience (default: 18)")
    args = parser.parse_args()

    train_pipeline(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience
    )
