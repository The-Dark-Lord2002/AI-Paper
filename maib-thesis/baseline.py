"""
مبنای کلاسیک: TF-IDF + Logistic Regression / Linear SVM (با و بدون وزن کلاس).
بدون این مبنا، هیچ ادعایی درباره‌ی برتری BERT قابل دفاع نیست.
اجرا روی CPU در چند ثانیه:   python baseline.py
"""
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from maib.data import load_splits
from maib.metrics import compute_metrics

p = argparse.ArgumentParser()
p.add_argument("--split_dir", default="splits")
p.add_argument("--results_dir", default="results")
args = p.parse_args()

train_df, val_df, test_df, labels = load_splits(args.split_dir)
l2i = {l: i for i, l in enumerate(labels)}
ytr = train_df["label"].map(l2i).values
yte = test_df["label"].map(l2i).values

models = {
    "tfidf_logreg": lambda cw: LogisticRegression(max_iter=3000, C=10, class_weight=cw),
    "tfidf_svm": lambda cw: LinearSVC(C=1.0, class_weight=cw),
}
out = Path(args.results_dir)
out.mkdir(parents=True, exist_ok=True)

for name, make in models.items():
    for cw, tag in [(None, "ce"), ("balanced", "wce")]:
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), make(cw))
        pipe.fit(train_df["text"], ytr)
        pred = pipe.predict(test_df["text"])
        m = compute_metrics(yte, pred, labels)
        run = f"{name}__none__{tag}__s0"
        (out / f"{run}.json").write_text(json.dumps({
            "run": run, "config": {"model": name, "head": "none", "loss": tag, "seed": 0},
            "labels": labels, "test": m, "test_predictions": pred.tolist()}, indent=2))
        print(f"{run:40s} macroF1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")
