"""
Evaluation Script for ISL Emergency Gesture Recognition.
Evaluates the trained LSTM model on the independent test dataset.
Generates:
- Training and validation accuracy graph (results/accuracy_plot.png)
- Training and validation loss graph (results/loss_plot.png)
- Combined learning curves (results/training_curves.png)
- Confusion Matrix heatmap (results/confusion_matrix.png)
- Classification Report (results/classification_report.txt and .json)
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

from training.dataset_loader import load_dataset, split_dataset, load_label_mapping


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

    plt.suptitle("LSTM Model Training History (IC3 2024)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined curves to: {save_path.resolve()}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """Generates and saves the confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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
    print("=" * 65)
    print("ISL Emergency Gesture Recognition - Model Evaluation")
    print("=" * 65)

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

    # Load test split
    splits_cache = models_dir / "dataset_splits.npz"
    if splits_cache.is_file():
        data = np.load(splits_cache)
        X_test, y_test = data["X_test"], data["y_test"]
        print(f"Loaded cached test split ({len(X_test)} samples) from: {splits_cache.name}")
    else:
        print("Re-computing stratified dataset split...")
        X, y, _, _, _ = load_dataset(PROJECT_ROOT / "data", verbose=False)
        _, _, (X_test, y_test), _ = split_dataset(X, y, verbose=False)

    # Load trained model
    print(f"Loading trained model from: {model_path.name}...")
    model = keras.models.load_model(str(model_path))

    # Evaluate on test set
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

    report_str = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print("\n" + "=" * 65)
    print("INDEPENDENT TEST SET PERFORMANCE METRICS")
    print("=" * 65)
    print(f"Overall Test Accuracy : {acc * 100:.2f}%")
    print(f"Weighted Precision    : {prec_weighted:.4f}")
    print(f"Weighted Recall       : {rec_weighted:.4f}")
    print(f"Weighted F1-Score     : {f1_weighted:.4f}")
    print(f"Macro F1-Score        : {f1_macro:.4f}")
    print("\nClassification Report:\n")
    print(report_str)
    print("=" * 65)

    # Save classification report
    report_txt_file = results_dir / "classification_report.txt"
    with open(report_txt_file, "w", encoding="utf-8") as f:
        f.write("ISL Emergency Gesture Recognition - Independent Test Set Evaluation\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Weighted Precision: {prec_weighted:.4f}\n")
        f.write(f"Weighted Recall: {rec_weighted:.4f}\n")
        f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n\n")
        f.write(report_str)
    print(f"Saved classification report text to: {report_txt_file.resolve()}")

    metrics_json_file = results_dir / "evaluation_metrics.json"
    metrics_data = {
        "overall_accuracy": float(acc),
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
