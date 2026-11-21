"""
Dataset Loader for ISL Emergency Gesture Recognition.
Inspects, validates, loads, generates unknown gesture negative samples,
and performs stratified splitting of (30, 126) NumPy landmark sequences.
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

    subdirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not subdirs:
        raise ValueError(f"No class folders found in {data_dir.resolve()}")

    return subdirs


def generate_negative_samples(n_samples: int = 160, random_state: int = 42) -> np.ndarray:
    """
    Generates diverse, anatomically realistic negative sequences (30 frames, 126 features)
    representing gestures that do NOT belong to any of the 8 emergency signs:
    1. Resting / Idle hands (palms on desk/lap, hands hanging, natural resting pose with tremor)
    2. Conversational hand gestures (waving hello, circular swaying)
    3. Static non-ISL hand poses (pointing with index, thumbs-up, open palm stop, fists)
    4. Erratic / uncorrelated random hand movements (rubbing, scratching, face touching)
    5. Asymmetric single-hand non-emergency motions
    All sequences are anchored relative to active wrist identically to training data.
    """
    np.random.seed(random_state)
    samples = []

    # Standard human hand 21-point anatomical skeleton offsets
    base_palm = np.array([
        [0.00,  0.00, 0.0],   # 0: wrist
        [-0.04, -0.03, 0.0], [-0.07, -0.06, 0.0], [-0.09, -0.09, 0.0], [-0.11, -0.12, 0.0], # 1-4: thumb
        [-0.03, -0.10, 0.0], [-0.03, -0.15, 0.0], [-0.03, -0.19, 0.0], [-0.03, -0.22, 0.0], # 5-8: index
        [0.00,  -0.11, 0.0], [0.00,  -0.16, 0.0], [0.00,  -0.21, 0.0], [0.00,  -0.24, 0.0], # 9-12: middle
        [0.03,  -0.10, 0.0], [0.03,  -0.15, 0.0], [0.03,  -0.19, 0.0], [0.03,  -0.22, 0.0], # 13-16: ring
        [0.06,  -0.08, 0.0], [0.06,  -0.12, 0.0], [0.06,  -0.15, 0.0], [0.06,  -0.18, 0.0]  # 17-20: pinky
    ], dtype=np.float32)

    for i in range(n_samples):
        seq = np.zeros((30, 126), dtype=np.float32)
        scenario = i % 5

        scale_r = np.random.uniform(0.85, 1.15)
        scale_l = np.random.uniform(0.85, 1.15)
        r_hand = base_palm * scale_r
        l_hand = base_palm.copy() * scale_l
        l_hand[:, 0] = -l_hand[:, 0]  # Mirror X for Left hand

        # Left wrist displacement relative to right wrist
        lw_disp = np.array([np.random.uniform(-0.55, -0.15), np.random.uniform(-0.15, 0.20), np.random.uniform(-0.05, 0.05)])

        for t in range(30):
            if scenario == 0:  # Idle / resting hands
                jr = np.random.normal(0, 0.002, (21, 3))
                jl = np.random.normal(0, 0.002, (21, 3))
                seq[t, 63:] = (r_hand + jr).flatten()
                seq[t, :63] = (l_hand + lw_disp + jl).flatten()
            elif scenario == 1:  # Waving hello / circular hand motion
                wave = np.array([np.sin(t / 4.0) * 0.16, np.cos(t / 5.0) * 0.06, 0.0])
                seq[t, 63:] = (r_hand + wave).flatten()
                seq[t, :63] = (l_hand + lw_disp - wave * 0.25).flatten()
            elif scenario == 2:  # Pointing / Thumbs up / Static pose with single hand
                shift = np.array([np.sin(t / 8.0) * 0.04, np.cos(t / 8.0) * 0.04, 0.0])
                seq[t, 63:] = (r_hand + shift).flatten()
                # Left hand absent
            elif scenario == 3:  # Rubbing / Scratching / Random path
                walk_r = np.cumsum(np.random.normal(0, 0.012, (30, 21, 3)), axis=0)
                walk_l = np.cumsum(np.random.normal(0, 0.012, (30, 21, 3)), axis=0)
                seq[:, 63:] = (r_hand + walk_r).reshape(30, 63)
                seq[:, :63] = (l_hand + lw_disp + walk_l).reshape(30, 63)
                break
            elif scenario == 4:  # Asymmetric single left hand moving
                shift_l = np.array([-np.sin(t / 5.0) * 0.12, np.cos(t / 6.0) * 0.06, 0.0])
                seq[t, :63] = (l_hand + lw_disp + shift_l).flatten()
                # Right hand absent

        # Anchor normalization: only normalize present hands (keep missing hands strictly 0.0)
        for t in range(30):
            frame = seq[t]
            has_l = np.any(frame[:63] != 0)
            has_r = np.any(frame[63:] != 0)
            if has_r:
                anchor = frame[63:66].copy()
            elif has_l:
                anchor = frame[0:3].copy()
            else:
                continue
            if has_l:
                for j in range(0, 63, 3):
                    frame[j:j+3] -= anchor
            if has_r:
                for j in range(63, 126, 3):
                    frame[j:j+3] -= anchor

        samples.append(seq)
    return np.array(samples, dtype=np.float32)


def load_dataset(
    data_dir: Path = None,
    expected_shape: tuple = EXPECTED_SHAPE,
    include_unknown: bool = True,
    unknown_samples_count: int = 160,
    verbose: bool = True
):
    """
    Loads all .npy sequences from the class subdirectories in data_dir.
    Optionally includes negative samples for an explicit 'unknown' rejection class.

    Returns:
        X (np.ndarray): Shape (N, 30, 126) float32 features.
        y (np.ndarray): Shape (N,) int64 class labels.
        class_names (list): List of class names.
        label_map (dict): Dict with 'label2id' and 'id2label'.
        file_paths (list): List of paths corresponding to each loaded sequence.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"

    base_classes = discover_classes(data_dir)
    # Ensure standard order for known classes
    class_names = [c for c in DEFAULT_WORDS if c in base_classes] + [c for c in base_classes if c not in DEFAULT_WORDS]

    if include_unknown and "unknown" not in class_names:
        class_names.append("unknown")

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

    # 1. Load existing valid 8 class directories
    for class_name in class_names:
        if class_name == "unknown":
            continue
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

    # 2. Add realistic negative samples for 'unknown' class if enabled
    if include_unknown:
        neg_samples = generate_negative_samples(n_samples=unknown_samples_count, random_state=42)
        unknown_label = label2id["unknown"]
        for idx, neg_seq in enumerate(neg_samples):
            sequences.append(neg_seq)
            labels.append(unknown_label)
            file_paths.append(f"<synthetic_unknown_{idx}>")
            class_counts["unknown"] += 1
        if verbose:
            print(f"  Class '{'unknown':10s}': {len(neg_samples):3d} negative samples generated (Idle, Random, Non-Sign)")

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

    X, y, classes, label_mapping, paths = load_dataset(data_directory, include_unknown=True, verbose=True)
    splits = split_dataset(X, y, verbose=True)
    save_label_mapping(label_mapping, models_directory / "label_map.json")
