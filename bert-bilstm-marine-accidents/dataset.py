"""Loading and preprocessing for the baker-street/maib-incident-reports-5K dataset.

Each record is a short marine-incident narrative (`text`) with a single
incident-type label (`label`), e.g. "Grounding / Stranding", "Fire / Explosion",
"Collision", "Accident to person(s)", "Damage / Loss Of Equipment".
"""
import re

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

DATASET_NAME = "baker-street/maib-incident-reports-5K"


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_maib_splits(seed: int = 42, val_size: float = 0.1, test_size: float = 0.1):
    """Fetch the dataset from the Hub and produce stratified train/val/test splits.

    The dataset only ships a single `train` split, so the splits below are
    carved out of it (stratified on label to keep class balance across splits).
    """
    raw = load_dataset(DATASET_NAME, split="train")
    texts = [clean_text(t) for t in raw["text"]]
    labels_raw = raw["label"]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_raw)

    holdout_size = val_size + test_size
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=holdout_size, random_state=seed, stratify=labels
    )
    relative_test_size = test_size / holdout_size
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels,
    )

    splits = {
        "train": (train_texts, train_labels),
        "val": (val_texts, val_labels),
        "test": (test_texts, test_labels),
    }
    return splits, label_encoder


class MAIBTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
