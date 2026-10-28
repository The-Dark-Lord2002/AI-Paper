"""
STEP 1 - The labelled dataset, split exactly as in the paper.

  1. Load the MAIB dataset (5,768 reports, 11 classes).
  2. Clean it: tidy whitespace, merge label spelling variants, remove duplicate reports.
  3. Drop classes with fewer than config.MIN_REPORTS_PER_CLASS reports.
  4. Split 70 / 15 / 15 into train / validation / test with the paper's seed, keeping class proportions.
  5. Check that the test set is the paper's test set (865 reports, same reports per class).

Run:
  python step1_prepare_data.py                    # downloads from Hugging Face
  python step1_prepare_data.py --file maib.jsonl  # or use a local copy (.csv / .jsonl / .parquet)

Output: data/train.csv, data/val.csv, data/test.csv, data/labels.json, reports/data_summary.json
"""
import argparse
import hashlib
import json
from pathlib import Path

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

# ----------------------------------------------------------------------------- 2. clean (as in the paper)
df = df.dropna().copy()
df["text"] = df["text"].astype(str).str.split().str.join(" ")
df = df[df["text"] != ""]

df["label"] = df["label"].astype(str).str.strip()
lower = df["label"].str.lower()
spelling = df.groupby(lower)["label"].agg(lambda s: s.value_counts().index[0])
df["label"] = lower.map(spelling)

text_key = df["text"].str.lower()
conflicting = df.groupby(text_key)["label"].transform("nunique") > 1
summary["conflicting_duplicates_removed"] = int(conflicting.sum())
df = df[~conflicting]

duplicate = df["text"].str.lower().duplicated()
summary["exact_duplicates_removed"] = int(duplicate.sum())
df = df[~duplicate]
summary["clean_reports"] = len(df)

# ----------------------------------------------------------------------------- 3. drop tiny classes
counts = df["label"].value_counts()
dropped = counts[counts < config.MIN_REPORTS_PER_CLASS]
kept = counts[counts >= config.MIN_REPORTS_PER_CLASS]
df = df[df["label"].isin(kept.index)]
print(f"Kept {len(kept)} classes, {len(df)} reports; dropped {dropped.to_dict()}")

# ----------------------------------------------------------------------------- 4. split 70 / 15 / 15
train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=config.SPLIT_SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=config.SPLIT_SEED)
print(f"Split: train={len(train)}  val={len(val)}  test={len(test)}")

data_dir = Path(config.DATA_DIR)
data_dir.mkdir(exist_ok=True)
train.to_csv(data_dir / "train.csv", index=False)
val.to_csv(data_dir / "val.csv", index=False)
test.to_csv(data_dir / "test.csv", index=False)
labels = sorted(kept.index)
(data_dir / "labels.json").write_text(json.dumps(labels, indent=2))

# ----------------------------------------------------------------------------- 5. same test set as the paper?
test_counts = test["label"].value_counts().to_dict()
same_split = test_counts == config.EXPECTED_TEST_COUNTS
fingerprint = hashlib.sha256("\n".join(test["text"]).encode("utf-8")).hexdigest()[:16]
if same_split:
    print("[ok] the test set has the paper's 865 reports and class counts")
else:
    print("[WARNING] the test set differs from the paper's; the results cannot be compared with the paper.\n"
          f"          expected {config.EXPECTED_TEST_COUNTS}\n          got      {test_counts}")
print(f"Test-set fingerprint: {fingerprint}  (identical texts in identical order give the same fingerprint)")

summary.update({
    "class_counts": counts.to_dict(),
    "dropped_classes": dropped.to_dict(),
    "split": {"train": len(train), "val": len(val), "test": len(test)},
    "test_reports_per_class": test_counts,
    "same_test_set_as_paper": same_split,
    "test_fingerprint": fingerprint,
})
save_json(summary, Path(config.REPORTS_DIR) / "data_summary.json")
print(f"Saved data to {data_dir}/ and the summary to {config.REPORTS_DIR}/data_summary.json")
