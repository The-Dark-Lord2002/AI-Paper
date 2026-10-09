"""
STEP 5 - Results tables and figures (proposal, step 5)

Reads every file in results/ and produces:
  reports/main_table.csv / .md     macro-F1, recall on the critical classes, accuracy: mean +- std over seeds
  reports/per_class_recall.csv     recall of every class for every configuration (rarest class first)
  reports/per_class_recall.png     the same as a heat map
  reports/confusion_matrix.png     confusion matrix of the best BERT configuration
and prints the answers to the two research questions.

  python step5_report.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

reports = Path(config.REPORTS_DIR)
reports.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- load and group runs
runs = [json.loads(p.read_text()) for p in sorted(Path(config.RESULTS_DIR).glob("*.json"))]
if not runs:
    raise SystemExit(f"No results in {config.RESULTS_DIR}/. Run steps 2 and 4 first.")


def config_name(run):
    """'bert_bilstm_att + focal' - the same name for all seeds of one configuration."""
    if run["model"] == "tfidf_svm":
        return "tfidf_svm"
    name = f"{run['model']} + {run['loss']}"
    return name + (f" [{run['tag']}]" if run.get("tag") else "")


groups = defaultdict(list)
for run in runs:
    groups[config_name(run)].append(run)

order = ["tfidf_svm"] + [f"{m} + {l}" for m, l in config.EXPERIMENTS]
names = [n for n in order if n in groups] + sorted(n for n in groups if n not in order)

labels = runs[0]["labels"]
critical = [c for c in config.CRITICAL_CLASSES if c in labels]
missing = set(config.CRITICAL_CLASSES) - set(critical)
if missing:
    print(f"[warning] critical classes not in the data, ignored: {sorted(missing)}")


def mean_std(values):
    values = np.array(values, dtype=float)
    return values.mean(), (values.std(ddof=1) if len(values) > 1 else 0.0)


# ----------------------------------------------------------------------------- main table
stats, rows = {}, []
for name in names:
    rs = groups[name]
    macro = mean_std([r["test"]["macro_f1"] for r in rs])
    accuracy = mean_std([r["test"]["accuracy"] for r in rs])
    crit = mean_std([np.mean([r["test"]["per_class"][c]["recall"] for c in critical]) for r in rs])
    stats[name] = {"macro_f1": macro, "critical_recall": crit}
    rows.append({"configuration": name, "seeds": len(rs),
                 "macro_f1": f"{macro[0]:.3f} ± {macro[1]:.3f}",
                 "critical_recall": f"{crit[0]:.3f} ± {crit[1]:.3f}",
                 "accuracy": f"{accuracy[0]:.3f} ± {accuracy[1]:.3f}"})
table = pd.DataFrame(rows)
table.to_csv(reports / "main_table.csv", index=False)


# ----------------------------------------------------------------------------- research questions
def compare(base, other):
    """One line: how much `other` changes macro-F1 and critical recall compared with `base`."""
    if base not in stats or other not in stats:
        return f"- {other} vs {base}: not available yet"
    d_f1 = stats[other]["macro_f1"][0] - stats[base]["macro_f1"][0]
    d_crit = stats[other]["critical_recall"][0] - stats[base]["critical_recall"][0]
    noise = max(stats[base]["macro_f1"][1], stats[other]["macro_f1"][1])
    verdict = "within seed-to-seed noise" if abs(d_f1) <= noise else "larger than seed-to-seed noise"
    return (f"- {other} vs {base}: macro-F1 {d_f1:+.3f}, critical recall {d_crit:+.3f} "
            f"(seed std {noise:.3f}: {verdict})")


questions = [
    "Is BERT better than the classic baseline?",
    compare("tfidf_svm", "bert + ce"),
    compare("bert + ce", "bert_bilstm + ce"),
    "",
    "RQ1 - Does attention over the BiLSTM states help? (proposal step 3)",
    compare("bert_bilstm + ce", "bert_bilstm_att + ce"),
    "",
    "RQ2 - Which imbalance treatment helps? (proposal step 4)",
    *[compare("bert_bilstm_att + ce", f"bert_bilstm_att + {loss}") for loss in ("wce", "focal", "cb")],
]

with open(reports / "main_table.md", "w") as f:
    f.write(f"Critical classes: {', '.join(critical)}\n\n")
    f.write("| Configuration | Seeds | Macro-F1 | Critical recall | Accuracy |\n|---|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r['configuration']} | {r['seeds']} | {r['macro_f1']} | {r['critical_recall']} | {r['accuracy']} |\n")
    f.write("\n" + "\n".join(questions) + "\n")

print(table.to_string(index=False))
print("\n" + "\n".join(questions))

# ----------------------------------------------------------------------------- per-class recall
test_support = runs[0]["test"]["per_class"]
classes_rarest_first = sorted(labels, key=lambda c: test_support[c]["support"])
recall = pd.DataFrame(
    {name: [np.mean([r["test"]["per_class"][c]["recall"] for r in groups[name]]) for c in classes_rarest_first]
     for name in names},
    index=classes_rarest_first).T.round(3)
recall.to_csv(reports / "per_class_recall.csv")

fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 0.5 * len(names) + 2))
image = ax.imshow(recall.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(labels)), [f"{c}\n(n={test_support[c]['support']})" for c in recall.columns],
              rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(labels)):
        ax.text(j, i, f"{recall.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Test recall per class (mean over seeds, rarest class on the left)")
fig.colorbar(image, ax=ax, fraction=0.02)
fig.tight_layout()
fig.savefig(reports / "per_class_recall.png", dpi=150)

# ----------------------------------------------------------------------------- confusion matrix
bert_names = [n for n in names if n != "tfidf_svm"]
if bert_names:
    best = max(bert_names, key=lambda n: stats[n]["macro_f1"][0])
    counts = np.sum([np.array(r["test"]["confusion_matrix"]) for r in groups[best]], axis=0)
    share = counts / counts.sum(axis=1, keepdims=True)              # each row sums to 1 -> diagonal = recall

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j]:
                ax.text(j, i, f"{share[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if share[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"Confusion matrix: {best} ({len(groups[best])} seeds summed, rows normalised)")
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.png", dpi=150)

print(f"\nSaved tables and figures to {reports}/")
