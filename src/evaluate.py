import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


def calculate_metrics(y_true, y_pred, y_proba, model_name: str):

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }
    return metrics


def plot_conf_matrix(y_true, y_pred, model_name: str, output_dir: str):

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")

    file_path = os.path.join(output_dir, f"confusion_matrix_{model_name.lower()}.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def get_roc_curve_data(y_true, y_proba):

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return fpr, tpr, auc


def plot_all_roc_curves(roc_data: dict, output_dir: str):

    plt.figure(figsize=(8, 6))

    for model_name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()

    file_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def plot_mlp_history(history, output_dir: str):

    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training History")
    plt.legend()

    file_path = os.path.join(output_dir, "mlp_training_history.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def save_metrics_table(metrics_list, output_dir: str):

    df_metrics = pd.DataFrame(metrics_list)
    file_path = os.path.join(output_dir, "metrics_table.csv")
    df_metrics.to_csv(file_path, index=False)
    return df_metrics