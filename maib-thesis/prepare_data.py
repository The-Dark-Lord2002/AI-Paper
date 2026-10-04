"""
مرحله‌ی ۱: بارگذاری، پاک‌سازی، تحلیل اکتشافی و ساخت تقسیم ثابت train/val/test.

    python prepare_data.py                         # مستقیم از Hugging Face
    python prepare_data.py --data maib.jsonl       # از فایل محلی (اگر HF در دسترس نیست)
"""
import argparse
import json
from pathlib import Path

from maib.data import clean, load_raw, make_splits

p = argparse.ArgumentParser()
p.add_argument("--data", default=None, help="local .jsonl/.csv/.parquet (default: Hugging Face)")
p.add_argument("--out", default="splits")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--min_count", type=int, default=3,
               help="classes with fewer reports are dropped (min 3: one each in train/val/test)")
args = p.parse_args()

df, stats = clean(load_raw(args.data))
words = df["text"].str.split().str.len()
dist = df["label"].value_counts()

print("== Cleaning ==")
for k, v in stats.items():
    print(f"  {k:32s} {v}")
print("\n== Label distribution ==")
for lab, n in dist.items():
    print(f"  {lab:32s} {n:5d}  ({100*n/len(df):5.1f}%)")
kept = dist[dist >= max(args.min_count, 3)]          # کلاس‌هایی که بعد از --min_count باقی می‌مانند
print(f"\n  imbalance ratio (max/min, {len(kept)} kept classes): {kept.max()/kept.min():.1f}")
print(f"\n== Text length (words) ==\n{words.describe().round(1).to_string()}")

split_info = make_splits(df, args.out, args.seed, min_count=args.min_count)
if split_info["dropped_classes"]:
    print(f"\n[warn] dropped classes with < {args.min_count} reports: {split_info['dropped_classes']}")
rare = {l: int(n) for l, n in dist.items() if args.min_count <= n < 20}
if rare:
    print(f"[warn] very rare classes (test F1 will be noisy, consider --min_count 20): {rare}")
print(f"\n== Splits saved to {args.out}/ ==  train={split_info['train']} "
      f"val={split_info['val']} test={split_info['test']}")

Path(args.out, "data_stats.json").write_text(json.dumps({
    "cleaning": stats,
    "label_distribution": dist.to_dict(),
    "imbalance_ratio": float(kept.max() / kept.min()),
    "words": words.describe().round(2).to_dict(),
    "splits": {k: v for k, v in split_info.items() if k != "labels"},
    "min_count": args.min_count,
    "split_seed": args.seed,
}, indent=2, ensure_ascii=False))
