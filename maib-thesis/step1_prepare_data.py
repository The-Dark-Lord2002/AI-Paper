"""
STEP 1 - Data (proposal, step 1)

  1. Load the MAIB dataset (5,768 reports, 11 classes).
  2. Clean it: tidy whitespace, merge label spelling variants, remove duplicate reports.
  3. Show the class distribution and find the rare classes.
  4. Drop classes with fewer than config.MIN_REPORTS_PER_CLASS reports.
  5. Split 70 / 15 / 15 into train / validation / test, keeping class proportions equal in all three.

Run:
  python step1_prepare_data.py                    # downloads from Hugging Face
  python step1_prepare_data.py --file maib.csv    # or use a local copy (.csv / .jsonl / .parquet)

Output:
  data/train.csv, data/val.csv, data/test.csv, data/labels.json
  reports/data_summary.json, reports/class_distribution.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from common import save_json

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="local copy of the dataset instead of downloading it")
args = parser.parse_args()

# ----------------------------------------------------------------------------- 1. load
if args.file:
    path = Path(args.file)
    readers = {".csv": pd.read_csv, ".parquet": pd.read_parquet,
               ".jsonl": lambda p: pd.read_json(p, lines=True)}
    df = readers[path.suffix](path)
else:
    from datasets import load_dataset
    df = load_dataset(config.DATASET, split="train").to_pandas()

df = df[["text", "label"]]
summary = {"raw_reports": len(df)}
print(f"Loaded {len(df)} reports")

# ----------------------------------------------------------------------------- 2. clean
df = df.dropna().copy()
df["text"] = df["text"].astype(str).str.split().str.join(" ")    # any run of spaces/newlines -> one space
df = df[df["text"] != ""]

# "Loss Of Control" and "Loss of Control" are the same class: use the most common spelling for both.
df["label"] = df["label"].astype(str).str.strip()
lower = df["label"].str.lower()
spelling = df.groupby(lower)["label"].agg(lambda s: s.value_counts().index[0])
df["label"] = lower.map(spelling)

# The same text with two different labels: we cannot know which label is right, so drop every copy.
text_key = df["text"].str.lower()
conflicting = df.groupby(text_key)["label"].transform("nunique") > 1
summary["conflicting_duplicates_removed"] = int(conflicting.sum())
df = df[~conflicting]

# The same text twice with the same label: keep one copy.
# (A report in both train and test would let the model "remember" the answer.)
duplicate = df["text"].str.lower().duplicated()
summary["exact_duplicates_removed"] = int(duplicate.sum())
df = df[~duplicate]
summary["clean_reports"] = len(df)

# ----------------------------------------------------------------------------- 3. class distribution
counts = df["label"].value_counts()
print("\nClass distribution:")
for label, n in counts.items():
    print(f"  {label:30s} {n:5d}  ({100 * n / len(df):4.1f}%)")

# ----------------------------------------------------------------------------- 4. drop tiny classes
# A class needs reports in train, val AND test. With fewer than ~20 reports the test set would hold
# 1-2 of them, and that single report would move macro-F1 by several points on its own.
assert config.MIN_REPORTS_PER_CLASS >= 10, "MIN_REPORTS_PER_CLASS must be at least 10 for a stratified split"
dropped = counts[counts < config.MIN_REPORTS_PER_CLASS]
kept = counts[counts >= config.MIN_REPORTS_PER_CLASS]
df = df[df["label"].isin(kept.index)]

print(f"\nDropped classes (< {config.MIN_REPORTS_PER_CLASS} reports): {dropped.to_dict()}")
print(f"Kept {len(kept)} classes, {len(df)} reports")
print(f"Imbalance ratio (largest / smallest kept class): {kept.max() / kept.min():.1f}")

# ----------------------------------------------------------------------------- 5. split 70 / 15 / 15
# stratify=label keeps the share of every class the same in all three parts.
train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=config.SPLIT_SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=config.SPLIT_SEED)
print(f"\nSplit: train={len(train)}  val={len(val)}  test={len(test)}")

data_dir = Path(config.DATA_DIR)
data_dir.mkdir(exist_ok=True)
train.to_csv(data_dir / "train.csv", index=False)
val.to_csv(data_dir / "val.csv", index=False)
test.to_csv(data_dir / "test.csv", index=False)
labels = sorted(kept.index)                                     # class id = position in this list
(data_dir / "labels.json").write_text(json.dumps(labels, indent=2))

summary.update({
    "class_counts": counts.to_dict(),
    "dropped_classes": dropped.to_dict(),
    "kept_classes": len(kept),
    "imbalance_ratio": round(float(kept.max() / kept.min()), 1),
    "split": {"train": len(train), "val": len(val), "test": len(test)},
    "test_reports_per_class": test["label"].value_counts().to_dict(),
    "words_per_report": df["text"].str.split().str.len().describe().round(1).to_dict(),
})
save_json(summary, Path(config.REPORTS_DIR) / "data_summary.json")

# ----------------------------------------------------------------------------- figure for the thesis
ordered = counts.sort_values()
colors = ["#bbbbbb" if label in dropped.index else
          "#d1495b" if label in config.CRITICAL_CLASSES else "#4c72b0" for label in ordered.index]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(ordered.index, ordered.values, color=colors)
for i, n in enumerate(ordered.values):
    ax.text(n, i, f" {n}", va="center", fontsize=8)
ax.set_xlabel("number of reports")
ax.set_title("MAIB class distribution (red = safety-critical, grey = dropped)")
fig.tight_layout()
fig.savefig(Path(config.REPORTS_DIR) / "class_distribution.png", dpi=150)
print(f"\nSaved data to {data_dir}/ and summary + figure to {config.REPORTS_DIR}/")
