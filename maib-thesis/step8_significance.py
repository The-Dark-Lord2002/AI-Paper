"""
Step 8: significance tests on the stored test predictions. No training needed (runs in seconds).

For each pair of configurations and each seed:
  - exact McNemar test on correct/incorrect per report (Dietterich, 1998)
  - paired bootstrap of the macro-F1 difference (95% interval, 2000 resamples)
For each pair, both seeds together (the numbers the paper reports are seed means):
  - paired bootstrap of the difference of the seed-averaged macro-F1: every resample of test reports
    is applied to all seeds at once; 95% interval and two-sided bootstrap p-value

Run:  python step8_significance.py      (after step1, so data/test.csv exists)
Output: reports/significance.md, reports/significance.csv (per seed), reports/significance_pooled.csv
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config
from common import load_split

# (A, B): is B different from A?
PAIRS = [
    ("tfidf_svm", "bert__ce"),
    ("tfidf_svm", "bert_bilstm_att__wce"),
    ("tfidf_svm", "bert_bilstm_att__wce__SEED__augment"),                # baseline vs the best configuration
    ("bert__ce", "bert_bilstm__ce"),
    ("bert_bilstm__ce", "bert_bilstm_att__ce"),
    ("bert__ce", "bert_bilstm_att__ce"),
    ("bert_bilstm_att__ce", "bert_bilstm_att__wce"),
    ("bert_bilstm_att__ce", "bert_bilstm_att__cb"),
    ("bert_bilstm_att__ce", "bert_bilstm_att__focal"),
    ("bert_bilstm_att__ce", "bert_bilstm_att__ce__SEED__oversample"),
    ("bert_bilstm_att__ce", "bert_bilstm_att__ce__SEED__augment"),
    ("bert_bilstm_att__wce", "bert_bilstm_att__wce__SEED__augment"),
    ("bert_bilstm_att__ce", "bert_bilstm_att__wce__SEED__augment"),      # the best configuration
    ("bert_bilstm_att__wce", "bert_bilstm_att__wce__SEED__h384_l1"),     # bigger BiLSTM
    ("bert_bilstm_att__wce", "bert_bilstm_att__wce__SEED__h128_l2"),
    ("bert_bilstm_att__wce", "bert_bilstm_att__wce__SEED__h384_l2"),
]
SVM_PAIRS = [("tfidf_svm", "tfidf_svm_oversample"), ("tfidf_svm", "tfidf_svm_augment")]   # one run each
SEEDS = config.SEEDS
N_BOOT = 2000


def result_file(name, seed):
    """Map a configuration name and seed to its JSON file. The SVM has one run without a seed."""
    if name.startswith("tfidf_svm"):
        return Path(config.RESULTS_DIR) / f"{name}.json"
    if "__SEED__" in name:
        return Path(config.RESULTS_DIR) / (name.replace("__SEED__", f"__seed{seed}__") + ".json")
    return Path(config.RESULTS_DIR) / f"{name}__seed{seed}.json"


def load_predictions(name, seed, y_true_names):
    d = json.loads(result_file(name, seed).read_text())
    pred = np.array(d["test_predictions"])
    true = np.array([d["labels"].index(lab) for lab in y_true_names])
    # data/test.csv must be the test set these predictions were made on (same reports, same order)
    if len(true) != len(pred) or abs(macro_f1(true, pred, len(d["labels"])) - d["test"]["macro_f1"]) > 1e-9:
        raise SystemExit(f"data/test.csv does not match {result_file(name, seed).name}: the stored predictions "
                         "were made on a different test set. Use the data folder the models were trained with.")
    return true, pred


def mcnemar_exact(correct_a, correct_b):
    """Two-sided exact McNemar p-value from the discordant pairs."""
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(p, 1.0)


def macro_f1(true, pred, n_classes):
    """Macro-F1 over ALL classes, as in common.compute_metrics (a class missing from a resample counts as 0)."""
    return f1_score(true, pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def bootstrap_f1_diff(true, pred_a, pred_b, n_classes, rng):
    n = len(true)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = macro_f1(true[idx], pred_b[idx], n_classes) - macro_f1(true[idx], pred_a[idx], n_classes)
    return np.percentile(diffs, [2.5, 97.5])


def pooled_bootstrap(true, preds_a, preds_b, n_classes, rng):
    """Difference of the seed-averaged macro-F1 (B - A), its 95% interval and a two-sided bootstrap p-value."""
    def diff(idx):
        return np.mean([macro_f1(true[idx], pb[idx], n_classes) - macro_f1(true[idx], pa[idx], n_classes)
                        for pa, pb in zip(preds_a, preds_b)])
    observed = diff(np.arange(len(true)))
    diffs = np.array([diff(rng.integers(0, len(true), len(true))) for _ in range(N_BOOT)])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return observed, lo, hi, p


def main():
    _, _, test, labels = load_split()
    y_names = test["label"].tolist()
    rng = np.random.default_rng(0)
    rows = []
    jobs = [(a, b, seed) for a, b in PAIRS for seed in SEEDS] + [(a, b, 0) for a, b in SVM_PAIRS]
    for a, b, seed in jobs:
        if not (result_file(a, seed).exists() and result_file(b, seed).exists()):
            print(f"[skip] {a} -> {b} seed {seed}: result file missing")
            continue
        true, pa = load_predictions(a, seed, y_names)
        true_b, pb = load_predictions(b, seed, y_names)
        assert (true == true_b).all(), "test order differs between result files"
        n01, n10, p = mcnemar_exact(pa == true, pb == true)
        d_f1 = macro_f1(true, pb, len(labels)) - macro_f1(true, pa, len(labels))
        lo, hi = bootstrap_f1_diff(true, pa, pb, len(labels), rng)
        rows.append(dict(A=a, B=b.replace("__SEED__", "__"), seed=seed,
                         A_right_B_wrong=n01, A_wrong_B_right=n10, mcnemar_p=round(p, 4),
                         delta_macro_f1=round(d_f1, 4), ci_low=round(lo, 4), ci_high=round(hi, 4)))
        print(f"{a:28s} -> {b:40s} seed {seed}: p={p:.3f}, dF1={d_f1:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    pooled = []
    for a, b in PAIRS + SVM_PAIRS:
        seeds = SEEDS if (a, b) in PAIRS else [0]
        if not all(result_file(a, s).exists() and result_file(b, s).exists() for s in seeds):
            continue
        true = load_predictions(a, seeds[0], y_names)[0]
        preds_a = [load_predictions(a, s, y_names)[1] for s in seeds]
        preds_b = [load_predictions(b, s, y_names)[1] for s in seeds]
        d, lo, hi, p = pooled_bootstrap(true, preds_a, preds_b, len(labels), rng)
        pooled.append(dict(A=a, B=b.replace("__SEED__", "__"), seeds=len(seeds), delta_macro_f1=round(d, 4),
                           ci_low=round(lo, 4), ci_high=round(hi, 4), bootstrap_p=round(p, 4)))
        print(f"{a:28s} -> {b:40s} seeds pooled: dF1={d:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f}")

    def markdown(table):
        return ("| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
                + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)))

    df, pooled_df = pd.DataFrame(rows), pd.DataFrame(pooled)
    out = Path(config.REPORTS_DIR)
    df.to_csv(out / "significance.csv", index=False)
    pooled_df.to_csv(out / "significance_pooled.csv", index=False)
    (out / "significance.md").write_text(
        "# Significance tests (test set)\n\n"
        "## Per seed\n\n"
        "McNemar: exact two-sided test on per-report correctness. "
        "CI: 95% paired bootstrap interval of the macro-F1 difference (B - A).\n\n"
        + markdown(df) + "\n\n"
        "## Both seeds together\n\n"
        f"Difference of the seed-averaged macro-F1 (B - A); 95% paired bootstrap interval ({N_BOOT} resamples "
        f"of the test reports, each applied to all seeds) and two-sided bootstrap p-value (0.0 means < {1 / N_BOOT:g}).\n\n"
        + markdown(pooled_df) + "\n")
    print(f"\nSaved {out/'significance.md'}")


if __name__ == "__main__":
    main()
