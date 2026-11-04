"""
Builds figures/confusion_matrix.pdf for the paper: the row-normalized test confusion matrix of one
configuration, summed over its seeds, in percent of each true class.

  python make_confusion_matrix.py                                   # the best configuration
  python make_confusion_matrix.py bert_bilstm_att wce augment       # model, loss, tag ("" for none)

The figure is drawn at its printed size (included without scaling in main.tex), so the font sizes below
are the sizes in the paper. IEEE asks for 8-point Times New Roman in figure labels; Liberation Serif has
the same metrics. Fonts are embedded as TrueType (pdf.fonttype 42): matplotlib's default, Type 3, is
rejected by IEEE PDF eXpress.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
})

HERE = Path(__file__).parent
RESULTS = HERE.parents[2] / "maib-thesis" / "results"
model, loss, tag = (sys.argv[1:4] + ["", "", ""])[:3] if len(sys.argv) > 1 else ("bert_bilstm_att", "wce", "augment")

runs = [r for r in (json.loads(p.read_text()) for p in RESULTS.glob(f"{model}__{loss}__seed*.json"))
        if (r.get("tag") or "") == tag]
if not runs:
    raise SystemExit(f"no results for {model} + {loss} [{tag}]")
labels = runs[0]["labels"]
counts = np.sum([np.array(r["test"]["confusion_matrix"]) for r in runs], axis=0)
share = counts / counts.sum(axis=1, keepdims=True)                    # each row sums to 1: diagonal = recall

# Full words, no abbreviations, in the axis labels (IEEE guideline); * marks safety-critical classes
NAMES = {"Accident to person(s)": "Accident", "Capsizing / Listing": "Capsizing*", "Collision": "Collision",
         "Contact": "Contact", "Damage / Loss Of Equipment": "Damage", "Fire / Explosion": "Fire*",
         "Flooding / Foundering": "Flooding*", "Grounding / Stranding": "Grounding",
         "Loss Of Control": "Loss of control"}
names = [NAMES.get(c, c) for c in labels]


def cell_text(value):
    """Percent of the true class, as an integer; '<1' for a non-zero share that rounds to 0."""
    percent = 100 * value
    return "<1" if percent < 0.5 else f"{percent:.0f}"


fig, ax = plt.subplots(figsize=(2.66, 2.2))                          # printed size in inches
ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")       # cells fill the width: room for the labels
ax.set_xticks(range(len(labels)), names, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
ax.set_yticks(range(len(labels)), names, fontsize=7)
for i in range(len(labels)):
    for j in range(len(labels)):
        if counts[i, j]:
            ax.text(j, i, cell_text(share[i, j]), ha="center", va="center", fontsize=5.5,
                    color="white" if share[i, j] > 0.5 else "black")
ax.set_xlabel("Predicted class", fontsize=8)
ax.set_ylabel("True class", fontsize=8)
ax.tick_params(length=0, pad=2)
fig.tight_layout(pad=0.2)
fig.savefig(HERE / "confusion_matrix.pdf")
print(f"{model} + {loss} [{tag}]: {len(runs)} seeds, {int(counts.sum() - np.trace(counts))} errors "
      f"of {int(counts.sum())} -> {HERE / 'confusion_matrix.pdf'}")
