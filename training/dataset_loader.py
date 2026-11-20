"""
Dataset Loader for ISL Emergency Gesture Recognition.
Inspects, validates, loads, and splits preprocessed (30, 126) NumPy sequences.
"""

import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_WORDS = ["accident", "call", "doctor", "help", "hot", "lose", "pain", "thief"]
EXPECTED_SHAPE = (30, 126)


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def discover_classes(data_dir: Path) -> list:
    """Discovers and validates gesture class directories."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found at: {data_dir.resolve()}")

    subdirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(f"No class folders found in {data_dir.resolve()}")

    return subdirs


def load_dataset(
    data_dir: Path = None,
    expected_shape: tuple = EXPECTED_SHAPE,
    verbose: bool = True
):
    """
    Loads all .npy sequences from the class subdirectories in data_dir.

    Returns:
        X (np.ndarray): Shape (N, 30, 126) float32 features.
        y (np.ndarray): Shape (N,) int64 class labels.
        class_names (list): List of class names.
        label_map (dict): Dict with 'label2id' and 'id2label'.
        file_paths (list): List of paths corresponding to each loaded sequence.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"

    class_names = discover_classes(data_dir)
    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {str(idx): name for idx, name in enumerate(class_names)}
    label_map = {"label2id": label2id, "id2label": id2label}

    sequences = []
    labels = []
    file_paths = []
    corrupted_count = 0
    class_counts = Counter()

    if verbose:
        print("=" * 65)
        print("Loading Preprocessed ISL Dataset")
        print("=" * 65)
        print(f"Data directory   : {data_dir.resolve()}")
        print(f"Classes ({len(class_names)}): {class_names}")
        print(f"Expected shape   : {expected_shape} (30 frames x 126 landmarks)")
        print("-" * 65)

    for class_name in class_names:
        class_folder = data_dir / class_name
        npy_files = sorted(class_folder.glob("*.npy"))
        valid_in_class = 0

        for fpath in npy_files:
            try:
                arr = np.load(fpath, allow_pickle=False)
                if arr.shape != expected_shape:
                    corrupted_count += 1
                    if verbose:
                        print(f"  [INVALID SHAPE] {fpath.name}: {arr.shape} != {expected_shape}")
                    continue

                if not np.isfinite(arr).all():
                    corrupted_count += 1
                    if verbose:
                        print(f"  [NON-FINITE VALUES] {fpath.name}")
                    continue

                sequences.append(arr.astype(np.float32))
                labels.append(label2id[class_name])
                file_paths.append(str(fpath))
                valid_in_class += 1
                class_counts[class_name] += 1

            except Exception as exc:
                corrupted_count += 1
                if verbose:
                    print(f"  [ERROR] {fpath.name}: {exc}")

        if verbose:
            print(f"  Class '{class_name:10s}': {valid_in_class:3d} valid sequences loaded")

    if not sequences:
        raise ValueError(f"No valid sequences found in {data_dir.resolve()}")

    X = np.stack(sequences).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    if verbose:
        print("-" * 65)
        print(f"Total Valid Samples : {len(X)}")
        print(f"Total Invalid/Skip  : {corrupted_count}")
        print(f"X Feature Shape     : {X.shape} (dtype: {X.dtype})")
        print(f"y Label Shape       : {y.shape} (dtype: {y.dtype})")
        print("=" * 65)

    return X, y, class_names, label_map, file_paths


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    verbose: bool = True
):
    """
    Performs stratified 70/15/15 split of the dataset.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), (train_idx, val_idx, test_idx)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    indices = np.arange(len(y))
    temp_ratio = val_ratio + test_ratio

    # First split: Train vs (Val + Test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y
    )

    # Second split: Val vs Test
    val_relative_ratio = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_relative_ratio),
        random_state=random_state,
        stratify=y[temp_idx]
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if verbose:
        print("\nDataset Split Summary (Stratified 70/15/15):")
        print(f"  Training Set   : {len(X_train):4d} samples ({len(X_train)*100/len(X):.1f}%) | Shape: {X_train.shape}")
        print(f"  Validation Set : {len(X_val):4d} samples ({len(X_val)*100/len(X):.1f}%) | Shape: {X_val.shape}")
        print(f"  Testing Set    : {len(X_test):4d} samples ({len(X_test)*100/len(X):.1f}%) | Shape: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (train_idx, val_idx, test_idx)


def save_label_mapping(label_map: dict, save_path: Path):
    """Saves label mapping dictionary to a JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)
    print(f"Saved class label mapping to: {save_path.resolve()}")


def load_label_mapping(load_path: Path) -> dict:
    """Loads label mapping dictionary from a JSON file."""
    if not load_path.is_file():
        raise FileNotFoundError(f"Label map not found at {load_path.resolve()}")
    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    root = get_project_root()
    data_directory = root / "data"
    models_directory = root / "models"

    X, y, classes, label_mapping, paths = load_dataset(data_directory, verbose=True)
    splits = split_dataset(X, y, verbose=True)
    save_label_mapping(label_mapping, models_directory / "label_map.json")
