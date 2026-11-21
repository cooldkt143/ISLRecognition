"""
Model Training Script for ISL Emergency Gesture Recognition.
Trains a high-accuracy Bidirectional LSTM model with:
- Feature Projection + LayerNormalization + Recurrent Dropout
- Stacked Bidirectional LSTMs + Dual Temporal Pooling (Average + Max)
- Hand-Swapping (Sagittal Reflection) Augmentation for true Left-Hand and Right-Hand Independence
- Explicit Negative Class ('unknown') for rejecting random/unrelated hand gestures
- Multi-factor spatial and temporal data augmentation
- Class Latent Prototypes extraction for Out-Of-Distribution Distance Gating
Saves the best model checkpoint to models/isl_lstm_model.keras, label map, and prototypes.
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


def mirror_sequence(X: np.ndarray) -> np.ndarray:
    """
    Applies sagittal reflection:
    1. Swaps Left Hand (indices 0..62) and Right Hand (indices 63..125).
    2. Inverts horizontal X coordinates (x -> -x).
    Preserves exact sign meaning for left-handed signers.
    """
    X_mirr = np.zeros_like(X)
    X_mirr[:, :, :63] = X[:, :, 63:]
    X_mirr[:, :, 63:] = X[:, :, :63]
    for j in range(0, 126, 3):
        X_mirr[:, :, j] = -X_mirr[:, :, j]
    return X_mirr


def augment_training_data(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Applies multi-factor data augmentation to achieve:
    1. Left-Hand and Right-Hand Independence (Sagittal Mirroring on target classes)
    2. Sensor landmark jitter (Gaussian noise on active coordinates)
    3. Hand size scaling (0.92 - 1.08)
    4. Temporal shifting (+/- 1-2 frames)
    5. Speed warping (faster / slower signs resampled to 30 frames)
    6. Intermittent detector flicker dropout
    """
    np.random.seed(random_state)
    N, T, D = X_train.shape

    # 1. Hand-Swapped / Sagittal Reflection (guarantees Left/Right hand independence)
    X_mirr = mirror_sequence(X_train)
    y_mirr = y_train.copy()

    # Combine original and mirrored base data (2x)
    X_base = np.concatenate([X_train, X_mirr], axis=0)
    y_base = np.concatenate([y_train, y_mirr], axis=0)
    M = len(X_base)

    # 2. Landmark Jitter (Gaussian noise on active landmarks)
    noise = np.random.normal(0, 0.005, X_base.shape).astype(np.float32)
    mask = (X_base != 0).astype(np.float32)
    X_jitter = X_base + (noise * mask)

    # 3. Hand Scale Scaling (0.92 to 1.08)
    scales = np.random.uniform(0.92, 1.08, (M, 1, 1)).astype(np.float32)
    X_scale = X_base * scales

    # 4. Temporal Shifting (+/- 1-2 frames)
    X_shift = np.zeros_like(X_base)
    for i in range(M):
        shift = np.random.randint(-2, 3)
        if shift > 0:
            X_shift[i, shift:] = X_base[i, :-shift]
            X_shift[i, :shift] = X_base[i, 0]
        elif shift < 0:
            X_shift[i, :shift] = X_base[i, -shift:]
            X_shift[i, shift:] = X_base[i, -1]
        else:
            X_shift[i] = X_base[i]

    # 5. Speed Warping (Resampling random sub-sequences)
    X_warp = np.zeros_like(X_base)
    for i in range(M):
        warp_len = np.random.randint(25, 36)
        src_indices = np.linspace(0, T - 1, warp_len)
        resampled_indices = np.linspace(0, warp_len - 1, T).astype(int)
        interp_seq = np.zeros((warp_len, D), dtype=np.float32)
        for d in range(D):
            interp_seq[:, d] = np.interp(src_indices, np.arange(T), X_base[i, :, d])
        X_warp[i] = interp_seq[resampled_indices]

    # Combine all augmentations
    X_aug = np.concatenate([X_base, X_jitter, X_scale, X_shift, X_warp], axis=0)
    y_aug = np.concatenate([y_base, y_base, y_base, y_base, y_base], axis=0)

    # 6. Single-hand drop simulation on 8% of samples
    for idx in range(len(X_aug)):
        if np.random.rand() < 0.08:
            drop_slot = np.random.choice([0, 1])
            if drop_slot == 0:
                X_aug[idx, :, :63] = 0.0  # Drop left hand
            else:
                X_aug[idx, :, 63:] = 0.0  # Drop right hand

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_aug))
    return X_aug[shuffle_idx], y_aug[shuffle_idx]


def build_bilstm_model(
    sequence_length: int = 30,
    feature_dim: int = 126,
    num_classes: int = 9,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Builds the fortified BiLSTM architecture:
    - Spatial Joint Projection: Dense(128) + LayerNorm + Dropout(0.2)
    - Stacked BiLSTMs: 128 units + 96 units with LayerNorm and Dropout
    - Dual Temporal Pooling: Concatenated GlobalAveragePooling1D + GlobalMaxPooling1D (384 units)
    - Dense Classification Head: 128 (ReLU) -> 64 (ReLU, latent embedding) -> Softmax(num_classes)
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

    # 3. Dual Temporal Pooling
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

    model = models.Model(inputs=inputs, outputs=outputs, name="ISL_BiLSTM_HandIndependent_Rejection")
    return model


def compute_and_save_class_prototypes(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_names: list,
    save_path: Path
):
    """
    Extracts the latent embeddings from layer 'dense_64' for all training samples,
    computes class centroid vectors, radius distributions, and rejection thresholds,
    and saves them to JSON for out-of-distribution (unknown gesture) rejection.
    """
    feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer("dense_64").output)
    embeddings = feature_extractor.predict(X_train, verbose=0)

    prototypes = {}
    for c_idx, c_name in enumerate(class_names):
        c_feats = embeddings[y_train == c_idx]
        if len(c_feats) == 0:
            continue
        centroid = np.mean(c_feats, axis=0)
        dists = np.linalg.norm(c_feats - centroid, axis=1)
        r95 = float(np.percentile(dists, 95))
        r_max = float(np.max(dists))
        threshold = float(max(r95 * 1.30, r_max * 1.10))

        prototypes[c_name] = {
            "class_id": int(c_idx),
            "centroid": [float(v) for v in centroid],
            "norm": float(np.linalg.norm(centroid)),
            "r95": r95,
            "threshold": threshold
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, indent=4)
    print(f"Saved class latent prototypes and OOD rejection thresholds to: {save_path.resolve()}")


def train_pipeline(
    batch_size: int = 32,
    max_epochs: int = 65,
    learning_rate: float = 0.001,
    patience: int = 16,
    random_state: int = 42
):
    print("=" * 68)
    print("ISL Emergency Gesture Recognition - Hand-Independent & Rejection Training")
    print("=" * 68)
    print(f"Keras Version   : {keras.__version__}")
    print(f"Keras Backend   : {keras.backend.backend()}")

    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset with negative unknown samples
    X, y, class_names, label_map, _ = load_dataset(data_dir, include_unknown=True, unknown_samples_count=160, verbose=True)
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

    # 2. Apply Hand-Swapping (Sagittal Mirror) and Multi-Factor Augmentation
    print("\nApplying Hand-Swapping (Left/Right Hand Independence) & Sequence Augmentation...")
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train, random_state=random_state)
    print(f"  Base Training Samples     : {len(X_train)}")
    print(f"  Augmented Training Samples: {len(X_train_aug)} (Hand-swapped + Jitter + Scale + Shift + Warp)")

    # Also augment validation set with mirrored samples for robust validation monitoring
    X_val_mirr = mirror_sequence(X_val)
    X_val_mon = np.concatenate([X_val, X_val_mirr], axis=0)
    y_val_mon = np.concatenate([y_val, y_val], axis=0)

    # 3. Build model architecture (9 classes: 8 emergency signs + 1 unknown rejection class)
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
        validation_data=(X_val_mon, y_val_mon),
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

    # 7. Compute and save Class Latent Prototypes for Out-of-Distribution Rejection
    prototypes_file = models_dir / "class_prototypes.json"
    compute_and_save_class_prototypes(
        model=model,
        X_train=X_train_aug,
        y_train=y_train_aug,
        class_names=class_names,
        save_path=prototypes_file
    )

    return model, history_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hand-independent BiLSTM model with unknown rejection for ISL.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=65, help="Maximum epochs (default: 65)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=16, help="Early stopping patience (default: 16)")
    args = parser.parse_args()

    train_pipeline(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience
    )
