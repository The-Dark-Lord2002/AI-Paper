"""
Builds figures/confusion_matrix.pdf for the paper: the row-normalized test confusion matrix of one
configuration, summed over its seeds.

  python make_confusion_matrix.py                                   # the best configuration
  python make_confusion_matrix.py bert_bilstm_att wce augment       # model, loss, tag ("" for none)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

SHORT = {"Accident to person(s)": "Accident", "Capsizing / Listing": "Capsizing*", "Collision": "Collision",
         "Contact": "Contact", "Damage / Loss Of Equipment": "Damage", "Fire / Explosion": "Fire*",
         "Flooding / Foundering": "Flooding*", "Grounding / Stranding": "Grounding",
         "Loss Of Control": "Loss of ctrl"}
names = [SHORT.get(c, c) for c in labels]

fig, ax = plt.subplots(figsize=(3.5, 3.3))
ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), names, rotation=45, ha="right", fontsize=6.5)
ax.set_yticks(range(len(labels)), names, fontsize=6.5)
for i in range(len(labels)):
    for j in range(len(labels)):
        if counts[i, j]:
            ax.text(j, i, f"{share[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5,
                    color="white" if share[i, j] > 0.5 else "black")
ax.set_xlabel("Predicted class", fontsize=7)
ax.set_ylabel("True class", fontsize=7)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(HERE / "confusion_matrix.pdf")
print(f"{model} + {loss} [{tag}]: {len(runs)} seeds, {int(counts.sum() - np.trace(counts))} errors "
      f"of {int(counts.sum())} -> {HERE / 'confusion_matrix.pdf'}")
