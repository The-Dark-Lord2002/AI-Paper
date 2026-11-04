"""
Writes the result tables of the paper from maib-thesis/results/*.json, so that no number is typed by hand.

  python make_tables.py

  tables/results.tex    Table III: macro precision, recall and F1, accuracy and critical-class recall
                        (mean +- standard deviation over seeds); best value of each column in bold,
                        second best underlined
  tables/perclass.tex   Table IV: F1 of every class for the main configurations; best per row in bold,
                        second best underlined
  tables/numbers.txt    every value with four decimals, to check the numbers quoted in the text
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE.parents[1] / "maib-thesis" / "results"
OUT = HERE / "tables"
CRITICAL = ["Fire / Explosion", "Capsizing / Listing", "Flooding / Foundering"]

# (result-file group, label in the paper); blocks are separated by a rule
BLOCKS = [
    [("tfidf_svm", "TF-IDF + SVM"),
     ("tfidf_svm_oversample", "TF-IDF + SVM + oversampling"),
     ("tfidf_svm_augment", "TF-IDF + SVM + LLM augm.")],
    [("bert + ce", "BERT"),
     ("bert_bilstm + ce", "BERT + BiLSTM \\cite{zhao2025causation}"),
     ("bert_bilstm_att + ce", "BERT + BiLSTM + Att")],
    [("bert_bilstm_att + wce", "Att + wce"),
     ("bert_bilstm_att + focal", "Att + focal"),
     ("bert_bilstm_att + cb", "Att + cb"),
     ("bert_bilstm_att + ce [oversample]", "Att + oversampling"),
     ("bert_bilstm_att + ce [augment]", "Att + LLM augm."),
     ("bert_bilstm_att + wce [augment]", "Att + wce + LLM augm. (proposed)")],
    [("bert_bilstm_att + wce [h384_l1]", "Att + wce, BiLSTM 384$\\times$1"),
     ("bert_bilstm_att + wce [h128_l2]", "Att + wce, BiLSTM 128$\\times$2"),
     ("bert_bilstm_att + wce [h384_l2]", "Att + wce, BiLSTM 384$\\times$2")],
]
PER_CLASS_COLUMNS = [("tfidf_svm", "SVM"), ("bert + ce", "BERT"), ("bert_bilstm + ce", "+BiLSTM"),
                     ("bert_bilstm_att + ce", "Att"), ("bert_bilstm_att + wce", "+wce"),
                     ("bert_bilstm_att + wce [augment]", "+LLM")]
SHORT_CLASS = {"Accident to person(s)": "Accident", "Damage / Loss Of Equipment": "Damage",
               "Loss Of Control": "Loss of Control", "Grounding / Stranding": "Grounding",
               "Fire / Explosion": "Fire*", "Flooding / Foundering": "Flooding*",
               "Capsizing / Listing": "Capsizing*"}


def group_name(run):
    if run["model"].startswith("tfidf"):
        return run["model"]
    return f"{run['model']} + {run['loss']}" + (f" [{run['tag']}]" if run.get("tag") else "")


groups = defaultdict(list)
for path in sorted(RESULTS.glob("*.json")):
    run = json.loads(path.read_text())
    groups[group_name(run)].append(run)


def run_metrics(run):
    pc = run["test"]["per_class"]
    return {"precision": np.mean([m["precision"] for m in pc.values()]),     # macro average, as sklearn
            "recall": np.mean([m["recall"] for m in pc.values()]),
            "f1": run["test"]["macro_f1"],
            "accuracy": run["test"]["accuracy"],
            "critical": np.mean([pc[c]["recall"] for c in CRITICAL])}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), (values.std(ddof=1) if len(values) > 1 else None)


stats = {}
for name, runs in groups.items():
    per_run = [run_metrics(r) for r in runs]
    stats[name] = {k: mean_std([m[k] for m in per_run]) for k in per_run[0]}


def marks(values):
    """Bold for the best rounded value, underline for the second best (ties share the mark)."""
    distinct = sorted({round(v, 3) for v in values}, reverse=True)
    best, second = distinct[0], (distinct[1] if len(distinct) > 1 else None)
    return ["best" if round(v, 3) == best else "second" if round(v, 3) == second else "" for v in values]


def cell(mean, std, mark):
    text = f"{mean:.3f}"
    text = {"best": f"\\textbf{{{text}}}", "second": f"\\underline{{{text}}}"}.get(mark, text)
    # IEEE style: a zero before the decimal point, also in the standard deviation (0.009, not .009)
    return text + (f"{{\\tiny$\\pm${std:.3f}}}" if std is not None else "")


OUT.mkdir(exist_ok=True)
columns = ["precision", "recall", "f1", "accuracy", "critical"]
rows = [(name, label) for block in BLOCKS for name, label in block if name in groups]
column_marks = {k: dict(zip([n for n, _ in rows], marks([stats[n][k][0] for n, _ in rows]))) for k in columns}
lines = []
for i, block in enumerate(BLOCKS):
    if i:
        lines.append("\\midrule")
    for name, label in block:
        if name in groups:
            lines.append(label + " & " + " & ".join(cell(*stats[name][k], column_marks[k][name]) for k in columns)
                         + " \\\\")
(OUT / "results.tex").write_text("\n".join(lines) + "\n")

labels = groups["tfidf_svm"][0]["labels"]
support = {c: groups["tfidf_svm"][0]["test"]["per_class"][c]["support"] for c in labels}
lines = []
for c in sorted(labels, key=lambda c: support[c]):
    values = [np.mean([r["test"]["per_class"][c]["f1"] for r in groups[n]]) for n, _ in PER_CLASS_COLUMNS]
    lines.append(f"{SHORT_CLASS.get(c, c)} ({support[c]}) & "
                 + " & ".join(cell(v, None, m) for v, m in zip(values, marks(values))) + " \\\\")
values = [stats[n]["f1"][0] for n, _ in PER_CLASS_COLUMNS]
lines += ["\\midrule", "Macro-F1 & " + " & ".join(cell(v, None, m) for v, m in zip(values, marks(values))) + " \\\\"]
(OUT / "perclass.tex").write_text("\n".join(lines) + "\n")

with open(OUT / "numbers.txt", "w") as f:
    for name, _ in rows:
        f.write(f"{name:36s} " + "  ".join(f"{k}={stats[name][k][0]:.4f}" + (f"+-{stats[name][k][1]:.4f}" if stats[name][k][1] is not None else "")
                                         for k in columns) + "\n")
print((OUT / "numbers.txt").read_text())
