"""Aggregate maib-thesis/results/*.json into data.json + figures for the report and slides."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1])            # maib-thesis
OUT = Path(sys.argv[2])             # output dir
OUT.mkdir(parents=True, exist_ok=True)
runs = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))]
summary = json.loads((ROOT / "reports" / "data_summary.json").read_text())

def name(r):
    if r["model"].startswith("tfidf"):
        return r["model"]
    n = f"{r['model']} + {r['loss']}"
    return n + (f" [{r['tag']}]" if r.get("tag") else "")

groups = defaultdict(list)
for r in runs:
    groups[name(r)].append(r)
order = ["tfidf_svm", "tfidf_svm_oversample", "bert + ce", "bert_bilstm + ce", "bert_bilstm_att + ce",
         "bert_bilstm_att + wce", "bert_bilstm_att + focal", "bert_bilstm_att + cb", "bert_bilstm_att + ce [oversample]"]
names = [n for n in order if n in groups] + sorted(n for n in groups if n not in order)
labels = runs[0]["labels"]
support = {c: runs[0]["test"]["per_class"][c]["support"] for c in labels}
critical = ["Fire / Explosion", "Capsizing / Listing", "Flooding / Foundering"]

def ms(v):
    v = np.array(v, float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]

configs = []
for n in names:
    rs = groups[n]
    configs.append({
        "name": n, "seeds": len(rs),
        "macro_f1": ms([r["test"]["macro_f1"] for r in rs]),
        "accuracy": ms([r["test"]["accuracy"] for r in rs]),
        "critical_recall": ms([np.mean([r["test"]["per_class"][c]["recall"] for c in critical]) for r in rs]),
        "recall": {c: float(np.mean([r["test"]["per_class"][c]["recall"] for r in rs])) for c in labels},
        "found": {c: float(np.mean([r["test"]["per_class"][c]["recall"] for r in rs]) * support[c]) for c in labels},
        "f1": {c: float(np.mean([r["test"]["per_class"][c]["f1"] for r in rs])) for c in labels},
    })
per_run = [{"run": r["run"], "best_epoch": r.get("best_epoch"), "minutes": r.get("train_minutes"),
            "macro_f1": r["test"]["macro_f1"], "accuracy": r["test"]["accuracy"]}
           for r in runs if r.get("history")]
best_epochs = [r["best_epoch"] for r in runs if r.get("best_epoch")]

# confusion pairs of the best BERT config (summed over seeds)
bert = [c for c in configs if not c["name"].startswith("tfidf")]
best = max(bert, key=lambda c: c["macro_f1"][0])["name"]
cm = np.sum([np.array(r["test"]["confusion_matrix"]) for r in groups[best]], axis=0)
pairs = sorted(((labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels))
                if i != j and cm[i, j] > 0), key=lambda t: -t[2])[:8]

top_words = {}
tw = ROOT / "reports" / "attention_top_words.csv"
if tw.exists():
    import csv
    for row in csv.DictReader(open(tw)):
        top_words.setdefault(row["class"], []).append(row["word"])

data = {"summary": summary, "labels": labels, "support": support, "critical": critical, "configs": configs,
        "per_run": per_run, "best_config": best, "best_cm_seeds": len(groups[best]),
        "confusion_pairs": pairs, "top_words": top_words,
        "best_epoch_counts": {str(k): best_epochs.count(k) for k in sorted(set(best_epochs))}}
(OUT / "data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))

# ------------------------------------------------------------------ figure: dot plot with seed std
SHORT = {"tfidf_svm": "TF-IDF + SVM", "tfidf_svm_oversample": "TF-IDF + SVM + oversample",
         "bert + ce": "BERT", "bert_bilstm + ce": "BERT + BiLSTM (reference)",
         "bert_bilstm_att + ce": "BERT + BiLSTM + Att", "bert_bilstm_att + wce": "  + Att + wce",
         "bert_bilstm_att + focal": "  + Att + focal", "bert_bilstm_att + cb": "  + Att + cb",
         "bert_bilstm_att + ce [oversample]": "  + Att + oversample"}
def group_of(n):
    if n.startswith("tfidf"): return 0
    if "oversample" in n: return 3
    if any(l in n for l in ("+ wce", "+ focal", "+ cb")): return 2
    return 1
COL = ["#8f8e88", "#2a78d6", "#eb6834", "#1baf7a"]          # baseline grey, architecture, loss, oversampling
GNAME = ["classic baseline", "architecture (loss = ce)", "imbalance-aware loss", "oversampling"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(configs) + 1.6), sharey=True)
ys = np.arange(len(configs))[::-1]
for ax, key, title in [(axes[0], "macro_f1", "Macro-F1 (test)"), (axes[1], "critical_recall",
                       "Recall on critical classes (fire, capsizing, flooding)")]:
    for y, c in zip(ys, configs):
        g = group_of(c["name"]); m, s = c[key]
        if s > 0:
            ax.plot([m - s, m + s], [y, y], color=COL[g], lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([m], [y], s=70, color=COL[g], edgecolor="white", linewidth=2, zorder=3)
        ax.text(m, y + 0.28, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, labelsize=8, length=0)
axes[0].set_yticks(ys, [SHORT.get(c["name"], c["name"]) for c in configs], fontsize=9, color=INK)
axes[0].set_ylim(-0.7, len(configs) - 0.2)
handles = [plt.Line2D([], [], marker="o", ls="", color=COL[g], markersize=8, label=GNAME[g])
           for g in sorted({group_of(c["name"]) for c in configs})]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.55, 0.0))
fig.text(0.99, 0.005, "dot = mean over seeds, line = ± 1 std", ha="right", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig_results.png", dpi=200, facecolor="white")

# ------------------------------------------------------------------ figure: learning curves (val macro-F1)
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, n in enumerate(["bert + ce", "bert_bilstm + ce", "bert_bilstm_att + ce", "bert_bilstm_att + wce"]):
    if n not in groups: continue
    h = np.array([[e["val_macro_f1"] for e in r["history"]] + [np.nan] * (5 - len(r["history"])) for r in groups[n]])
    ax.plot(range(1, 6), np.nanmean(h, axis=0), marker="o", lw=2, ms=6, color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][i],
            markeredgecolor="white", markeredgewidth=1.5, label=SHORT[n].strip())
ax.set_xlabel("epoch", color=MUTED, fontsize=9); ax.set_ylabel("validation macro-F1", color=MUTED, fontsize=9)
ax.set_xticks(range(1, 6)); ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8, length=0); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_learning_curves.png", dpi=200, facecolor="white")
print("configs:", [c["name"] for c in configs]); print("best:", best); print("pairs:", pairs[:5])
