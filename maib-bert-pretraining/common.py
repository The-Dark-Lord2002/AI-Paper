"""
Small helpers used by several steps: reading the data split and the corpus, computing metrics, saving JSON.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import config


def load_split():
    """Return the train, val and test tables (columns: text, label) and the list of class names."""
    data_dir = Path(config.DATA_DIR)
    if not (data_dir / "train.csv").exists():
        raise SystemExit(f"{data_dir}/train.csv not found. Run step1_prepare_data.py first.")
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    labels = json.loads((data_dir / "labels.json").read_text())
    return train, val, test, labels


def read_lines(path):
    """One text per line (the corpus format of step 2)."""
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(texts, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(texts) + "\n", encoding="utf-8")


def set_seed(seed):
    """Same seed -> same initial weights, same dropout, same batch order, same masks -> same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred, labels):
    """
    Every number we report (as in the paper). y_true and y_pred are class indices (0 .. len(labels) - 1).
    macro_f1 is the main metric: every class counts the same, so ignoring the rare classes is punished.
    """
    class_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            labels[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                        "f1": float(f1[i]), "support": int(support[i])}
            for i in class_ids
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
    }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
