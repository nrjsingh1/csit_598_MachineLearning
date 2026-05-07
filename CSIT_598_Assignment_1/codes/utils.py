from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def load_mnist_openml(as_frame: bool = False):
    """Load MNIST from OpenML with deterministic parser settings."""
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=as_frame,
        parser="auto",
    )
    return X, y


def prepare_mnist_splits(
    test_size: int = 10000,
    val_size: int = 10000,
    random_state: int = RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """
    Create train/val/test splits from MNIST.

    Output sizes default to 50k train / 10k val / 10k test.
    """
    X, y = load_mnist_openml(as_frame=False)

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_normalized = X / 255.0

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_normalized,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    val_ratio = val_size / X_train_val.shape[0]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_std": X_train_std,
        "X_val_std": X_val_std,
        "X_test_std": X_test_std,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def evaluate_classifier(model, X_train, y_train, X_eval, y_eval) -> Dict[str, Any]:
    """Train a classifier and return core timing + classification metrics."""
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_pred = model.predict(X_eval)
    pred_time_sec = time.perf_counter() - start_pred

    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision_macro": float(precision_score(y_eval, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_eval, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_eval, y_pred, average="macro", zero_division=0)),
        "train_time_sec": float(train_time_sec),
        "pred_time_sec": float(pred_time_sec),
        "classification_report": classification_report(y_eval, y_pred, zero_division=0),
        "y_pred": y_pred,
    }
    return metrics


def plot_confusion(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
