"""
بارگذاری، پاک‌سازی و تقسیم داده‌ی MAIB.

نکته‌ی روش‌شناختی: تقسیم داده یک‌بار با seed ثابت ساخته و در splits/ ذخیره می‌شود.
همه‌ی مدل‌ها (baseline و BERT با هر loss و هر seed) روی همین تقسیم ارزیابی می‌شوند؛
seed آموزش جدا از seed تقسیم است. بدون این، مقایسه‌ی مدل‌ها منصفانه نیست.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HF_DATASET = "baker-street/maib-incident-reports-5K"


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()


def _canonicalize_labels(labels: pd.Series) -> pd.Series:
    """برچسب‌هایی که فقط در حروف بزرگ/کوچک فرق دارند ("Loss Of Control" / "Loss of Control")
    یکی می‌شوند؛ پرتکرارترین نگارش به‌عنوان نام نهایی نگه داشته می‌شود."""
    labels = labels.map(_normalize_text)
    key = labels.str.lower()
    canonical = labels.groupby(key).agg(lambda s: s.value_counts().index[0])
    return key.map(canonical)


def load_raw(path: str | None = None) -> pd.DataFrame:
    """از فایل محلی (jsonl / csv / parquet) یا مستقیم از Hugging Face می‌خواند."""
    if path:
        p = Path(path)
        if p.suffix == ".jsonl":
            df = pd.read_json(p, lines=True)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")
    else:
        from datasets import load_dataset
        df = load_dataset(HF_DATASET, split="train").to_pandas()
    return df[["text", "label"]]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """حذف خالی‌ها، یکسان‌سازی برچسب، حذف تکراری‌ها و تکراری‌های با برچسب متناقض."""
    stats = {"raw": len(df)}
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].map(_normalize_text)
    n_variants = df["label"].map(_normalize_text).nunique()
    df["label"] = _canonicalize_labels(df["label"])
    stats["label_spelling_variants_merged"] = n_variants - df["label"].nunique()
    df = df[df["text"].str.len() > 0]

    key = df["text"].str.lower()
    conflicting = df.groupby(key)["label"].transform("nunique") > 1
    stats["conflicting_duplicates_removed"] = int(conflicting.sum())
    df = df[~conflicting]

    before = len(df)
    df = df.loc[~df["text"].str.lower().duplicated()]
    stats["exact_duplicates_removed"] = before - len(df)
    stats["final"] = len(df)
    return df.reset_index(drop=True), stats


def make_splits(df: pd.DataFrame, out_dir: str = "splits", seed: int = 42,
                val_size: float = 0.15, test_size: float = 0.15) -> dict:
    """تقسیم stratified سه‌گانه و ذخیره روی دیسک."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df, tmp = train_test_split(df, test_size=val_size + test_size,
                                     stratify=df["label"], random_state=seed)
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(tmp, test_size=rel_test,
                                       stratify=tmp["label"], random_state=seed)
    labels = sorted(df["label"].unique())
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d.to_json(out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    (out / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    return {"train": len(train_df), "val": len(val_df), "test": len(test_df), "labels": labels}


def load_splits(split_dir: str = "splits"):
    d = Path(split_dir)
    if not (d / "train.jsonl").exists():
        raise FileNotFoundError(f"{d}/train.jsonl not found. Run: python prepare_data.py")
    labels = json.loads((d / "labels.json").read_text())
    dfs = {s: pd.read_json(d / f"{s}.jsonl", lines=True) for s in ("train", "val", "test")}
    return dfs["train"], dfs["val"], dfs["test"], labels
