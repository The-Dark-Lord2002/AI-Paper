"""
STEP 2 - Classic baseline: TF-IDF + linear SVM (proposal, step 2a)

Before any neural network: how far does a simple word-counting model get?
If BERT cannot clearly beat this, the extra cost of BERT is not justified.

  TF-IDF : turns each report into a vector with one number per word (and word pair).
           The number is high when the word is frequent in THIS report but rare in the whole collection.
  SVM    : a linear classifier that separates the classes in that vector space.

Runs on the CPU in a few seconds:
  python step2_tfidf_svm.py

Output: results/tfidf_svm.json
"""
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import config
from common import compute_metrics, load_split, save_json

train, val, test, labels = load_split()
label_to_id = {label: i for i, label in enumerate(labels)}
y_train = train["label"].map(label_to_id).values
y_test = test["label"].map(label_to_id).values

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),   # single words + word pairs
    LinearSVC(C=1.0),
)
model.fit(train["text"], y_train)
predictions = model.predict(test["text"])

metrics = compute_metrics(y_test, predictions, labels)
save_json({
    "run": "tfidf_svm", "model": "tfidf_svm", "loss": "-", "seed": 0, "tag": "",
    "labels": labels,
    "test": metrics,
    "test_predictions": predictions.tolist(),
}, Path(config.RESULTS_DIR) / "tfidf_svm.json")

print(f"TF-IDF + SVM   test macro-F1 = {metrics['macro_f1']:.4f}   accuracy = {metrics['accuracy']:.4f}")
print("Recall per class:")
for label, m in sorted(metrics["per_class"].items(), key=lambda kv: kv[1]["support"]):
    print(f"  {label:30s} {m['recall']:.3f}   ({m['support']} test reports)")
