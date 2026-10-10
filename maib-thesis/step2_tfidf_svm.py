"""
STEP 2 - Classic baseline: TF-IDF + linear SVM (proposal, step 2a)

Before any neural network: how far does a simple word-counting model get?
If BERT cannot clearly beat this, the extra cost of BERT is not justified.

  TF-IDF : turns each report into a vector with one number per word (and word pair).
           The number is high when the word is frequent in THIS report but rare in the whole collection.
  SVM    : a linear classifier that separates the classes in that vector space.

It is trained twice: on the training set as it is, and with oversampling (reports of the smaller
classes are copied at random until every class is as large as the biggest one). Only the TRAINING set
is oversampled; the test set is never touched.

Runs on the CPU in a few seconds:
  python step2_tfidf_svm.py

Output: results/tfidf_svm.json, results/tfidf_svm_oversample.json
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import config
from common import compute_metrics, load_split, save_json

train, val, test, labels = load_split()
label_to_id = {label: i for i, label in enumerate(labels)}
y_test = test["label"].map(label_to_id).values


def oversample(df, seed=0):
    """Keep every report, then add random copies of the smaller classes until all classes are equally large."""
    largest = df["label"].value_counts().max()
    parts = [pd.concat([group, group.sample(largest - len(group), replace=True, random_state=seed)])
             for _, group in df.groupby("label")]
    return pd.concat(parts)


for name, train_set in [("tfidf_svm", train), ("tfidf_svm_oversample", oversample(train))]:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),   # single words + word pairs
        LinearSVC(C=1.0),
    )
    model.fit(train_set["text"], train_set["label"].map(label_to_id).values)
    predictions = model.predict(test["text"])

    metrics = compute_metrics(y_test, predictions, labels)
    save_json({
        "run": name, "model": name, "loss": "-", "seed": 0, "tag": "",
        "labels": labels,
        "test": metrics,
        "test_predictions": predictions.tolist(),
    }, Path(config.RESULTS_DIR) / f"{name}.json")

    print(f"\n{name}  ({len(train_set)} training reports)   "
          f"test macro-F1 = {metrics['macro_f1']:.4f}   accuracy = {metrics['accuracy']:.4f}")
    print("Recall per class:")
    for label, m in sorted(metrics["per_class"].items(), key=lambda kv: kv[1]["support"]):
        print(f"  {label:30s} {m['recall']:.3f}   ({m['support']} test reports)")
