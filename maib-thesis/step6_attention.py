"""
STEP 6 - Interpretability: which words did the model look at? (proposal, end of step 5)

The attention model (bert_bilstm_att) gives every word of a report a weight; the weights sum to 1.
Step 3 saved these weights for every test report. This script:
  1. merges BERT word-pieces back into words ("capsiz", "##ed" -> "capsized") and adds their weights
  2. writes reports/attention_examples.html: for every class, correctly AND wrongly classified reports,
     with each word coloured by its weight (darker red = more weight)
  3. writes reports/attention_top_words.csv: the words with the most attention for each class

  python step6_attention.py                                   # uses the best attention configuration
  python step6_attention.py --run bert_bilstm_att__focal__seed1

Note for the thesis: attention weights show where the model looked, not a proof of why it decided
(Jain & Wallace, 2019). Describe them as "the words the model weighted most".
"""
import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config
from common import load_split

parser = argparse.ArgumentParser()
parser.add_argument("--run", help="result file name without .json (default: best attention configuration)")
parser.add_argument("--examples", type=int, default=2, help="correct and wrong examples shown per class")
args = parser.parse_args()

# ----------------------------------------------------------------------------- choose the run
results = Path(config.RESULTS_DIR)
if args.run:
    run = json.loads((results / f"{args.run}.json").read_text())
else:
    attention_runs = [json.loads(p.read_text()) for p in results.glob("bert_bilstm_att__*.json")]
    attention_runs = [r for r in attention_runs if r.get("test_attention")]
    if not attention_runs:
        raise SystemExit("No bert_bilstm_att results found. Run step 4 first.")
    by_loss = defaultdict(list)
    for r in attention_runs:
        by_loss[(r["loss"], r.get("tag", ""))].append(r)
    best_group = max(by_loss.values(), key=lambda rs: np.mean([r["test"]["macro_f1"] for r in rs]))
    run = min(best_group, key=lambda r: r["seed"])
print(f"Using run: {run['run']}  (test macro-F1 = {run['test']['macro_f1']:.3f})")

labels = run["labels"]
_, _, test_df, _ = load_split()
truth = test_df["label"].tolist()
predicted = [labels[i] for i in run["test_predictions"]]


def to_words(tokens, weights):
    """Join word-pieces into words and add up their weights. Skips [CLS] and [SEP]."""
    words, word_weights = [], []
    for token, weight in zip(tokens, weights):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if token.startswith("##") and words:
            words[-1] += token[2:]
            word_weights[-1] += weight
        else:
            words.append(token)
            word_weights.append(weight)
    return words, word_weights


reports = [to_words(a["tokens"], a["weights"]) for a in run["test_attention"]]

# ----------------------------------------------------------------------------- top words per class
top_words = defaultdict(Counter)
for (words, weights), gold, pred in zip(reports, truth, predicted):
    if gold != pred:
        continue                                          # use correctly classified reports only
    for word, weight in zip(words, weights):
        if word.isalpha() and len(word) > 2 and word not in ENGLISH_STOP_WORDS:
            top_words[gold][word] += weight

rows = [{"class": label, "rank": rank, "word": word, "attention": round(total, 3)}
        for label in labels for rank, (word, total) in enumerate(top_words[label].most_common(10), start=1)]
top_table = pd.DataFrame(rows)
top_table.to_csv(Path(config.REPORTS_DIR) / "attention_top_words.csv", index=False)

# ----------------------------------------------------------------------------- HTML with coloured examples
def colour(words, weights):
    biggest = max(weights) or 1.0
    return " ".join(f'<span style="background: rgba(214,40,40,{w / biggest:.2f})">{html.escape(word)}</span>'
                    for word, w in zip(words, weights))


parts = [f"<h1>Attention weights: {html.escape(run['run'])}</h1>",
         "<p>Darker red = more attention. ✓ = correct prediction, ✗ = wrong prediction.</p>"]
for label in labels:
    parts.append(f"<h2>{html.escape(label)}</h2>")
    words_line = ", ".join(word for word, _ in top_words[label].most_common(10))
    parts.append(f"<p><b>Top attention words:</b> {html.escape(words_line) or '-'}</p>")
    for correct in (True, False):
        shown = 0
        for (words, weights), gold, pred in zip(reports, truth, predicted):
            if gold != label or (gold == pred) != correct or shown >= args.examples:
                continue
            shown += 1
            mark = "✓" if correct else f"✗ predicted <b>{html.escape(pred)}</b>"
            parts.append(f"<div class='ex'>{mark}<p>{colour(words, weights)}</p></div>")

page = ("<!doctype html><meta charset='utf-8'><title>Attention examples</title>"
        "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;line-height:1.8}"
        ".ex{border-bottom:1px solid #ddd;padding:.5em 0}</style>" + "\n".join(parts))
(Path(config.REPORTS_DIR) / "attention_examples.html").write_text(page)

print(top_table[top_table["rank"] <= 5].pivot(index="rank", columns="class", values="word").to_string())
print(f"\nSaved {config.REPORTS_DIR}/attention_examples.html and {config.REPORTS_DIR}/attention_top_words.csv")
