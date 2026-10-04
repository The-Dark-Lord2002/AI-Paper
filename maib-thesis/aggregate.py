"""
تجمیع نتایج همه‌ی اجراها به جدول‌های آماده‌ی پایان‌نامه/مقاله.

    python aggregate.py
    python aggregate.py --reference bert-base-uncased/cls/ce

خروجی‌ها در reports/:
  summary.csv / summary.md   : میانگین ± انحراف معیار روی seedها (معیار اصلی: macro-F1)
  per_class_f1.csv           : F1 هر کلاس برای هر پیکربندی
  per_class_recall.csv       : Recall هر کلاس (معیار اصلی پروپوزال برای رده‌های نادر)
  significance.csv           : آزمون bootstrap جفت‌شده در برابر پیکربندی مرجع
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", default="results")
ap.add_argument("--split_dir", default="splits")
ap.add_argument("--out", default="reports")
ap.add_argument("--reference", default="bert-base-uncased/bilstm/ce",
                help="model/head/loss to test every other config against (default: paper reproduction)")
ap.add_argument("--n_boot", type=int, default=2000)
args = ap.parse_args()

runs = [json.loads(p.read_text()) for p in sorted(Path(args.results_dir).glob("*.json"))]
if not runs:
    raise SystemExit(f"No results in {args.results_dir}/")
labels = runs[0]["labels"]
l2i = {l: i for i, l in enumerate(labels)}
y_test = pd.read_json(Path(args.split_dir) / "test.jsonl", lines=True)["label"].map(l2i).values

groups = defaultdict(list)
for r in runs:
    c = r["config"]
    key = f"{c['model'].split('/')[-1]}/{c.get('head', c.get('pooling'))}/{c['loss']}"
    groups[key].append(r)

# رده‌های کم‌نمونه و ایمنی‌بحرانی که پروپوزال روی آن‌ها تأکید دارد
CRITICAL = [l for l in labels if any(k in l.lower() for k in
            ("fire", "capsiz", "hull", "flooding"))]

rows, pc_rows, rc_rows = [], [], []
for key, rs in groups.items():
    def agg(metric):
        v = np.array([r["test"][metric] for r in rs])
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    row = {"config": key, "n_seeds": len(rs)}
    for m in ("macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"):
        mu, sd = agg(m)
        row[f"{m}_mean"], row[f"{m}_std"] = round(mu, 4), round(sd, 4)
    crit = np.array([np.mean([r["test"]["per_class"][l]["recall"] for l in CRITICAL]) for r in rs])
    row["critical_recall_mean"] = round(crit.mean(), 4)
    row["critical_recall_std"] = round(crit.std(ddof=1) if len(crit) > 1 else 0.0, 4)
    rows.append(row)
    rc = {"config": key}
    for lab in labels:
        rc[lab] = round(np.mean([r["test"]["per_class"][lab]["recall"] for r in rs]), 4)
    rc_rows.append(rc)
    pc = {"config": key}
    for lab in labels:
        pc[lab] = round(np.mean([r["test"]["per_class"][lab]["f1"] for r in rs]), 4)
    pc_rows.append(pc)

summary = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / "summary.csv", index=False)

support = {lab: runs[0]["test"]["per_class"][lab]["support"] for lab in labels}
per_class = pd.DataFrame(pc_rows).set_index("config").loc[summary["config"]]
per_class = per_class[sorted(labels, key=lambda l: support[l])]   # نادرترین کلاس اول
per_class.to_csv(out / "per_class_f1.csv")
per_recall = pd.DataFrame(rc_rows).set_index("config").loc[summary["config"]]
per_recall = per_recall[sorted(labels, key=lambda l: support[l])]
per_recall.to_csv(out / "per_class_recall.csv")

with open(out / "summary.md", "w") as f:
    f.write(f"Critical classes: {', '.join(CRITICAL)}\n\n")
    f.write("| Config | Seeds | Macro-F1 | Critical Recall | Weighted-F1 | Balanced Acc | Accuracy |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for _, r in summary.iterrows():
        cell = lambda m: f"{r[m+'_mean']:.4f} ± {r[m+'_std']:.4f}"
        f.write(f"| {r['config']} | {r['n_seeds']} | {cell('macro_f1')} | {cell('critical_recall')} | {cell('weighted_f1')} | "
                f"{cell('balanced_accuracy')} | {cell('accuracy')} |\n")


# ---- آزمون معناداری: bootstrap جفت‌شده روی نمونه‌های test، seed به seed ----
def paired_bootstrap(y, pa, pb, n, rng):
    """p-value یک‌طرفه برای H0: macroF1(B) <= macroF1(A)."""
    idx = np.arange(len(labels))
    obs = f1_score(y, pb, labels=idx, average="macro") - f1_score(y, pa, labels=idx, average="macro")
    worse = 0
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        d = (f1_score(y[s], pb[s], labels=idx, average="macro", zero_division=0)
             - f1_score(y[s], pa[s], labels=idx, average="macro", zero_division=0))
        worse += d <= 0
    return obs, (worse + 1) / (n + 1)


sig_rows = []
if args.reference in groups:
    rng = np.random.default_rng(0)
    ref = {r["config"]["seed"]: np.array(r["test_predictions"]) for r in groups[args.reference]}
    for key, rs in groups.items():
        if key == args.reference:
            continue
        for r in rs:
            s = r["config"]["seed"]
            pa = ref.get(s, next(iter(ref.values())))   # baselineهای قطعی فقط seed 0 دارند
            d, p = paired_bootstrap(y_test, pa, np.array(r["test_predictions"]), args.n_boot, rng)
            sig_rows.append({"config": key, "seed": s, "delta_macro_f1": round(d, 4), "p_value": round(p, 4)})
    pd.DataFrame(sig_rows).to_csv(out / "significance.csv", index=False)
else:
    print(f"[warn] reference '{args.reference}' not found; skipping significance tests")

print(open(out / "summary.md").read())
print(f"\nRecall on rarest classes (rarest first):\n{per_recall.iloc[:, :4].to_string()}")
if sig_rows:
    sig = pd.DataFrame(sig_rows).groupby("config").agg(
        delta=("delta_macro_f1", "mean"), max_p=("p_value", "max"))
    print(f"\nvs {args.reference} (mean Δ macro-F1, worst-seed p):\n{sig.round(4).to_string()}")
print(f"\nSaved to {out}/")
