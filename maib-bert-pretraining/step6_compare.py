"""
STEP 6 - Does the adapted BERT beat the original BERT? Tables and significance tests. No training needed.

For every configuration (config.FINETUNE_CONFIGS) and every encoder:
  macro-F1, accuracy, macro precision and recall, recall of the critical classes, F1 of each critical class
  (mean +- standard deviation over seeds)
For every adapted encoder against the original BERT ("base"), as in the paper:
  - exact McNemar test per seed
  - paired bootstrap of the seed-averaged macro-F1 difference (95% interval and two-sided p-value)
And, from step 3, the MLM perplexity on the validation reports before and after the adaptation.

Result files copied from the paper (without an "encoder" field) count as "base".

  python step6_compare.py
Output: reports/comparison.md, reports/comparison.csv, reports/significance.csv
"""
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config
from common import load_split

N_BOOT = 2000


def macro_f1(true, pred, n_classes):
    return f1_score(true, pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(p, 1.0)


def pooled_bootstrap(true, preds_a, preds_b, n_classes, rng):
    """Seed-averaged macro-F1 difference (B - A), its 95% interval and two-sided bootstrap p-value."""
    def diff(idx):
        return np.mean([macro_f1(true[idx], pb[idx], n_classes) - macro_f1(true[idx], pa[idx], n_classes)
                        for pa, pb in zip(preds_a, preds_b)])
    observed = diff(np.arange(len(true)))
    diffs = np.array([diff(rng.integers(0, len(true), len(true))) for _ in range(N_BOOT)])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return observed, lo, hi, p


def markdown(df):
    return ("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
            + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)))


def main():
    _, _, test, labels = load_split()
    true = np.array([labels.index(name) for name in test["label"]])

    # ------------------------------------------------------------------ read the result files
    runs = defaultdict(dict)                         # (encoder, model, loss) -> {seed: result}
    for path in sorted(Path(config.RESULTS_DIR).glob("*.json")):
        d = json.loads(path.read_text())
        if d.get("tag") or "test_predictions" not in d or d["model"].startswith("tfidf"):
            continue                                 # e.g. augmented or oversampled runs copied from the paper
        pred = np.array(d["test_predictions"])
        if d["labels"] != labels or len(pred) != len(true) or abs(macro_f1(true, pred, len(labels)) - d["test"]["macro_f1"]) > 1e-9:
            raise SystemExit(f"{path.name} was not made on data/test.csv (different split or class order). "
                             "Use the same data as the runs, see step 1.")
        runs[(d.get("encoder", "base"), d["model"], d["loss"])][d["seed"]] = d
    if not runs:
        raise SystemExit("No result files in results/. Run step5_run_all.py first.")

    # ------------------------------------------------------------------ table of results
    rows = []
    for (encoder, model, loss), by_seed in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0] != "base", kv[0][0])):
        def stat(fn):
            values = [fn(r) for r in by_seed.values()]
            return f"{np.mean(values):.3f}" + (f" ± {np.std(values, ddof=1):.3f}" if len(values) > 1 else "")
        pc = lambda r: r["test"]["per_class"]
        row = {"configuration": f"{model} + {loss}", "encoder": encoder, "seeds": len(by_seed),
               "macro-F1": stat(lambda r: r["test"]["macro_f1"]),
               "accuracy": stat(lambda r: r["test"]["accuracy"]),
               "macro-P": stat(lambda r: np.mean([m["precision"] for m in pc(r).values()])),
               "macro-R": stat(lambda r: np.mean([m["recall"] for m in pc(r).values()])),
               "critical recall": stat(lambda r: np.mean([pc(r)[c]["recall"] for c in config.CRITICAL_CLASSES]))}
        for c in config.CRITICAL_CLASSES:
            row[f"F1 {c.split(' /')[0]}"] = stat(lambda r, c=c: pc(r)[c]["f1"])
        rows.append(row)
    table = pd.DataFrame(rows)

    # ------------------------------------------------------------------ adapted vs base
    rng = np.random.default_rng(0)
    tests = []
    for (encoder, model, loss), by_seed in runs.items():
        base = runs.get(("base", model, loss), {})
        seeds = sorted(set(by_seed) & set(base))
        if encoder == "base" or not seeds:
            continue
        preds_a = [np.array(base[s]["test_predictions"]) for s in seeds]
        preds_b = [np.array(by_seed[s]["test_predictions"]) for s in seeds]
        mcnemar = [mcnemar_exact(pa == true, pb == true)[2] for pa, pb in zip(preds_a, preds_b)]
        d, lo, hi, p = pooled_bootstrap(true, preds_a, preds_b, len(labels), rng)
        tests.append({"configuration": f"{model} + {loss}", "encoder (B) vs base (A)": encoder, "seeds": len(seeds),
                      "delta macro-F1": f"{d:+.4f}", "95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "bootstrap p": round(p, 4),
                      "McNemar p per seed": ", ".join(f"{x:.3f}" for x in mcnemar)})
    tests = pd.DataFrame(tests)

    # ------------------------------------------------------------------ MLM perplexity (step 3)
    pretrain = []
    for path in sorted(Path(config.REPORTS_DIR).glob("pretrain_*.json")):
        d = json.loads(path.read_text())
        pretrain.append({"encoder": d["model"], "corpus texts": d["corpus_texts"], "corpus words": d["corpus_words"],
                         "starts from": Path(d["base_model"]).name, "best epoch": d["best_epoch"],
                         "epochs run": d.get("epochs_run", len(d["history"])),
                         "converged": {True: "yes", False: "NO"}.get(d.get("converged"), "?"),
                         "perplexity before": d["before"]["labelled_val"]["perplexity"],
                         "perplexity after": d["after"]["labelled_val"]["perplexity"]})
    pretrain = pd.DataFrame(pretrain)

    out = Path(config.REPORTS_DIR)
    out.mkdir(exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    tests.to_csv(out / "significance.csv", index=False)
    text = ("# Original BERT vs domain-adapted BERT (test set, 865 reports)\n\n"
            "## Results (mean ± standard deviation over seeds)\n\n" + markdown(table) + "\n\n"
            "## Significance: adapted encoder (B) against the original BERT (A)\n\n"
            f"Paired bootstrap of the seed-averaged macro-F1 difference ({N_BOOT} resamples); exact McNemar test per seed.\n\n"
            + (markdown(tests) if len(tests) else "No adapted encoder with the same seeds as base yet.") + "\n\n"
            "## Stage 1: MLM perplexity on the validation reports (lower = BERT knows the language better)\n\n"
            "converged = early stopping ended the run; NO = the loss was still falling at the last allowed epoch.\n"
            "Perplexities of different corpora are not comparable as a measure of quality: TAPT is trained on texts\n"
            "like the validation reports, so its perplexity is lower without the encoder being better for classification.\n\n"
            + (markdown(pretrain) if len(pretrain) else "No step-3 summary found.") + "\n")
    (out / "comparison.md").write_text(text)
    print(text)
    print(f"Saved {out / 'comparison.md'}")


if __name__ == "__main__":
    main()
