"""
Evaluation Script for ISL Emergency Gesture Recognition.
Evaluates the trained LSTM model on:
1. Standard Independent Test Dataset (Right-Hand / Canonical)
2. Left-Hand Sagittal-Mirrored Test Dataset (Hand-Independence Verification)
3. Synthetic Out-of-Distribution Negative Gestures (Unknown Rejection Verification)

Generates:
- Training and validation accuracy graph (results/accuracy_plot.png)
- Training and validation loss graph (results/loss_plot.png)
- Combined learning curves (results/training_curves.png)
- Confusion Matrix heatmap (results/confusion_matrix.png)
- Comprehensive Performance Metrics & Classification Report (results/classification_report.txt & evaluation_metrics.json)
"""

import os
import sys
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Project paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_loader import (
    load_dataset,
    split_dataset,
    load_label_mapping,
    generate_negative_samples
)
from training.train import mirror_sequence


def plot_accuracy(history: dict, save_path: Path):
    """Generates and saves the training and validation accuracy plot."""
    epochs = range(1, len(history["accuracy"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], "b-o", label="Training Accuracy", linewidth=2, markersize=4)
    plt.plot(epochs, history["val_accuracy"], "r--s", label="Validation Accuracy", linewidth=2, markersize=4)
    plt.title("Model Accuracy Progression vs Epochs", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved accuracy plot to: {save_path.resolve()}")


def plot_loss(history: dict, save_path: Path):
    """Generates and saves the training and validation loss plot."""
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], "b-o", label="Training Loss", linewidth=2, markersize=4)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2, markersize=4)
    plt.title("Model Loss Progression vs Epochs", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (Crossentropy)", fontsize=11)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss plot to: {save_path.resolve()}")


def plot_training_curves(history: dict, save_path: Path):
    """Generates a side-by-side accuracy and loss figure."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["accuracy"], "b-", label="Train Acc", linewidth=2)
    ax1.plot(epochs, history["val_accuracy"], "r--", label="Val Acc", linewidth=2)
    ax1.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, history["loss"], "b-", label="Train Loss", linewidth=2)
    ax2.plot(epochs, history["val_loss"], "r--", label="Val Loss", linewidth=2)
    ax2.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("BiLSTM Hand-Independent ISL Model Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined curves to: {save_path.resolve()}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """Generates and saves the confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title("Confusion Matrix - Test Dataset", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Gesture", fontsize=11, fontweight="bold")
    plt.ylabel("True Gesture", fontsize=11, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {save_path.resolve()}")


def evaluate_model():
    print("=" * 70)
    print("ISL Emergency Gesture Recognition - Comprehensive Model Evaluation")
    print("=" * 70)

    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "isl_lstm_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at: {model_path.resolve()}. Please run train.py first.")

    label_map_path = models_dir / "label_map.json"
    label_map = load_label_mapping(label_map_path)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    class_names = [id2label[i] for i in range(len(id2label))]

    # 1. Load test split
    splits_cache = models_dir / "dataset_splits.npz"
    if splits_cache.is_file():
        data = np.load(splits_cache)
        X_test, y_test = data["X_test"], data["y_test"]
        print(f"Loaded test split ({len(X_test)} samples) from: {splits_cache.name}")
    else:
        print("Re-computing stratified dataset split...")
        X, y, _, _, _ = load_dataset(PROJECT_ROOT / "data", include_unknown=True, verbose=False)
        _, _, (X_test, y_test), _ = split_dataset(X, y, verbose=False)

    # 2. Load trained model
    print(f"Loading trained model from: {model_path.name}...")
    model = keras.models.load_model(str(model_path))

    # Optional feature extractor for latent prototype distance verification
    has_prototypes = False
    prototypes = {}
    prototypes_path = models_dir / "class_prototypes.json"
    if prototypes_path.is_file():
        with open(prototypes_path, "r", encoding="utf-8") as f:
            prototypes = json.load(f)
        has_prototypes = True
        print(f"Loaded latent class prototypes ({len(prototypes)} classes) from: {prototypes_path.name}")

    # =========================================================================
    # PART A: Standard Independent Test Set Evaluation (Right-Hand / Canonical)
    # =========================================================================
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc_standard = accuracy_score(y_test, y_pred)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

    report_str = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # =========================================================================
    # PART B: Left-Hand Sagittal-Mirrored Test Evaluation (Hand Independence)
    # =========================================================================
    # Filter only target emergency classes (excluding 'unknown' for hand mirroring)
    target_mask = np.array([id2label[yi] != "unknown" for yi in y_test])
    X_test_targets = X_test[target_mask]
    y_test_targets = y_test[target_mask]

    X_test_mirr = mirror_sequence(X_test_targets)
    y_pred_mirr_probs = model.predict(X_test_mirr, verbose=0)
    y_pred_mirr = np.argmax(y_pred_mirr_probs, axis=1)

    acc_mirrored = accuracy_score(y_test_targets, y_pred_mirr)
    per_class_mirr = {}
    for c_idx, c_name in id2label.items():
        if c_name == "unknown":
            continue
        c_mask = (y_test_targets == c_idx)
        if np.sum(c_mask) > 0:
            c_acc = accuracy_score(y_test_targets[c_mask], y_pred_mirr[c_mask])
            per_class_mirr[c_name] = float(c_acc)

    # =========================================================================
    # PART C: Unknown / Out-of-Distribution Gesture Rejection Evaluation
    # =========================================================================
    # Generate 120 independent out-of-distribution sequences (random seed 9999)
    n_unknown_eval = 120
    X_ood = generate_negative_samples(n_samples=n_unknown_eval, random_state=9999)
    ood_probs = model.predict(X_ood, verbose=0)

    # Count how many are rejected by the multi-tier rejection logic:
    # 1. Direct classification into 'unknown' class
    # 2. Confidence below 0.70 threshold
    # 3. Softmax margin below 0.25 threshold
    rejected_count = 0
    unknown_class_id = label_map["label2id"].get("unknown", -1)

    for i in range(n_unknown_eval):
        probs_i = ood_probs[i]
        sorted_p = np.sort(probs_i)[::-1]
        top_idx = int(np.argmax(probs_i))
        top_conf = sorted_p[0]
        margin = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else top_conf

        is_rejected = False
        if top_idx == unknown_class_id:
            is_rejected = True
        elif top_conf < 0.70:
            is_rejected = True
        elif margin < 0.25:
            is_rejected = True

        if is_rejected:
            rejected_count += 1

    unknown_rejection_rate = (rejected_count / n_unknown_eval) * 100.0

    # Print summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SET PERFORMANCE METRICS")
    print("=" * 70)
    print(f"1. Standard Test Accuracy (Right-Handed): {acc_standard * 100:.2f}%")
    print(f"   - Weighted Precision                 : {prec_weighted:.4f}")
    print(f"   - Weighted Recall                    : {rec_weighted:.4f}")
    print(f"   - Weighted F1-Score                  : {f1_weighted:.4f}")
    print(f"   - Macro F1-Score                     : {f1_macro:.4f}")
    print(f"2. Mirrored Test Accuracy (Left-Handed) : {acc_mirrored * 100:.2f}%")
    for c_name, c_acc in per_class_mirr.items():
        print(f"     - Class '{c_name:8s}' Left-Hand Acc: {c_acc * 100:.1f}%")
    print(f"3. Unknown Gesture Rejection Rate       : {unknown_rejection_rate:.2f}% ({rejected_count}/{n_unknown_eval} rejected)")
    print("\nClassification Report (Standard Test Set):\n")
    print(report_str)
    print("=" * 70)

    # Save classification report
    report_txt_file = results_dir / "classification_report.txt"
    with open(report_txt_file, "w", encoding="utf-8") as f:
        f.write("ISL Emergency Gesture Recognition - Independent Test Set Evaluation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Standard Test Accuracy (Right-Handed): {acc_standard * 100:.2f}%\n")
        f.write(f"Mirrored Test Accuracy (Left-Handed) : {acc_mirrored * 100:.2f}%\n")
        f.write(f"Unknown Gesture Rejection Rate       : {unknown_rejection_rate:.2f}%\n")
        f.write(f"Weighted Precision: {prec_weighted:.4f}\n")
        f.write(f"Weighted Recall: {rec_weighted:.4f}\n")
        f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
        f.write(f"Macro F1-Score: {f1_macro:.4f}\n\n")
        f.write("Left-Hand Accuracies per Class:\n")
        for c_name, c_acc in per_class_mirr.items():
            f.write(f"  - {c_name:8s}: {c_acc * 100:.2f}%\n")
        f.write("\nClassification Report:\n")
        f.write(report_str)
    print(f"Saved classification report text to: {report_txt_file.resolve()}")

    metrics_json_file = results_dir / "evaluation_metrics.json"
    metrics_data = {
        "standard_test_accuracy": float(acc_standard),
        "mirrored_left_hand_accuracy": float(acc_mirrored),
        "unknown_rejection_rate_percent": float(unknown_rejection_rate),
        "unknown_rejection_count": int(rejected_count),
        "unknown_evaluation_total": int(n_unknown_eval),
        "per_class_left_hand_accuracy": per_class_mirr,
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "macro_f1": float(f1_macro),
        "classification_report": report_dict
    }
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved metrics JSON to: {metrics_json_file.resolve()}")

    # Generate Confusion Matrix
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path)

    # Plot training curves from history
    history_file = results_dir / "training_history.json"
    if history_file.is_file():
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_accuracy(history, results_dir / "accuracy_plot.png")
        plot_loss(history, results_dir / "loss_plot.png")
        plot_training_curves(history, results_dir / "training_curves.png")

    print("\nEvaluation successfully completed. All results saved in results/.")


if __name__ == "__main__":
    evaluate_model()
