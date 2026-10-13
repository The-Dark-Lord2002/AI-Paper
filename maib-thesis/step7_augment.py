"""
STEP 7 - Data augmentation with a language model (supervisor's request)

Oversampling only COPIES the rare reports: the model sees the same 69 capsizing reports again and again
and learns nothing new. Here a small instruction-tuned language model (config.AUGMENT_LLM) REWRITES real
training reports of the rare classes in new words (paraphrasing). The facts stay the same, so the label
stays the same, but the wording is new.

Rules that keep the evaluation honest:
  - only TRAINING reports are rewritten. Validation and test stay 100% real and are never shown to the LLM.
  - every class with fewer than config.AUGMENT_TARGET training reports is filled up to that number.
    The real reports of a class are rewritten in turn, so each one is used about equally often.
  - a rewrite is thrown away if it is much shorter or longer than the original, uses almost the same
    words as the original (not really rewritten), shares almost no words with it (another story or
    another language), or is a duplicate.
  - label check (config.AUGMENT_LABEL_CHECK): a TF-IDF + SVM trained on the REAL training reports must
    still put the rewrite in the class of its original. Rewrites that changed a fact so much that they
    now read like another class are not used (filtering as in LAMBADA, Anaby-Tavor et al., 2020).
    Only rewrites that pass count toward the target, so rejected ones are replaced.

  python step7_augment.py

Running it again after the generation has finished only re-checks the labels and rebuilds the files.

The first run downloads the language model (about 3 GB). Stopping and starting again is safe: the rewrites
are saved after every batch and the script continues where it stopped.
If the GPU runs out of memory, use the smaller "Qwen/Qwen2.5-0.5B-Instruct" in config.AUGMENT_LLM.

Output:
  data/augmented.csv                 every rewrite (text, label, source = row of the original in train.csv),
                                     including the ones that fail the label check
  data/train_augmented.csv           real training reports + rewrites that pass the label check;
                                     used by --augment in steps 2 and 4
  reports/augmentation_examples.md   counts per class and original vs rewrite, to check by eye and for the thesis
"""
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from common import load_split

INSTRUCTION = (
    "You rewrite marine incident reports. Rewrite the report below with different words and sentence "
    "structure. Keep every fact exactly: the type of vessel, what happened, the cause, injuries and damage. "
    "Do not add new facts and do not name the incident category. Reply with the rewritten report only."
)


def load_rewriter():
    """Load the language model once and return a function that rewrites a list of reports."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No GPU found: rewriting will be very slow.")
    tokenizer = AutoTokenizer.from_pretrained(config.AUGMENT_LLM, padding_side="left")   # left: the answer follows
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.AUGMENT_LLM, dtype=dtype).to(device).eval()

    @torch.no_grad()
    def rewrite(texts):
        prompts = [tokenizer.apply_chat_template([{"role": "system", "content": INSTRUCTION},
                                                  {"role": "user", "content": text}],
                                                 tokenize=False, add_generation_prompt=True)
                   for text in texts]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        output = model.generate(**batch, max_new_tokens=256, do_sample=True, temperature=0.8, top_p=0.95,
                                pad_token_id=tokenizer.pad_token_id)
        answers = output[:, batch["input_ids"].shape[1]:]               # drop the prompt, keep only the answer
        return tokenizer.batch_decode(answers, skip_special_tokens=True)

    return rewrite


def clean(text):
    """First paragraph only, without a leading 'Rewritten report:' or quotes, single spaces."""
    text = text.strip().split("\n\n")[0]
    text = re.sub(r"^(rewritten report|report)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.strip().strip('"').split())


def acceptable(new, original, seen):
    """Keep a rewrite only if it has a similar length, shares some but not most words, and is new."""
    new_words, old_words = new.lower().split(), original.lower().split()
    if not new_words or not 0.5 <= len(new_words) / len(old_words) <= 2.0:
        return False
    overlap = len(set(new_words) & set(old_words)) / len(set(new_words) | set(old_words))
    return 0.15 <= overlap <= 0.8 and new.lower() not in seen


def label_checker(train):
    """
    Return keeps_label(texts, labels) -> array of True/False: does a TF-IDF + SVM, trained on the REAL training
    reports only, put each rewrite in its intended class? With config.AUGMENT_LABEL_CHECK = False, always True.
    """
    if not config.AUGMENT_LABEL_CHECK:
        return lambda texts, labels: np.ones(len(texts), dtype=bool)
    svm = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LinearSVC(C=1.0))
    svm.fit(train["text"], train["label"])
    return lambda texts, labels: svm.predict(list(texts)) == np.asarray(list(labels))


def augment(train, labels, out_file, keeps_label, rewrite=None):
    """
    Rewrite reports until every class has config.AUGMENT_TARGET training reports that pass the label check.
    Every rewrite that passes the other checks is appended to out_file; only those that keep their label count.
    """
    done = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame(columns=["text", "label", "source"])
    seen = set(train["text"].str.lower()) | set(done["text"].str.lower())
    counts = train["label"].value_counts()
    done_ok = done[keeps_label(done["text"], done["label"])] if len(done) else done
    need = {label: int(config.AUGMENT_TARGET - counts[label] - (done_ok["label"] == label).sum()) for label in labels}
    need = {label: n for label, n in need.items() if n > 0}
    print("New reports needed per class:", need or "none, augmentation is complete")
    if not need:
        return
    rewrite = rewrite or load_rewriter()

    random.seed(config.SPLIT_SEED)
    torch.manual_seed(config.SPLIT_SEED)
    for label, n_needed in need.items():
        sources = train.index[train["label"] == label].tolist()      # rows of the real reports of this class
        random.shuffle(sources)
        position = int((done["label"] == label).sum())               # continue the rotation where it stopped
        made, tried = 0, 0
        progress = tqdm(total=n_needed, desc=label)
        while made < n_needed and tried < 5 * n_needed:              # give up after too many rejected rewrites
            rows = [sources[(position + k) % len(sources)] for k in range(config.AUGMENT_BATCH)]
            position += len(rows)
            tried += len(rows)
            originals = train.loc[rows, "text"].tolist()
            rewrites = [clean(new) for new in rewrite(originals)]
            label_ok = keeps_label(rewrites, [label] * len(rewrites))
            new_rows, n_ok = [], 0
            for row, original, new, ok in zip(rows, originals, rewrites, label_ok):
                if made + n_ok < n_needed and acceptable(new, original, seen):
                    seen.add(new.lower())
                    new_rows.append({"text": new, "label": label, "source": row})
                    n_ok += int(ok)                                  # only rewrites that keep their label count
            if new_rows:                                             # save after every batch
                pd.DataFrame(new_rows).to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
                made += n_ok
                progress.update(n_ok)
        progress.close()
        if made < n_needed:
            print(f"[warning] {label}: only {made} of {n_needed} rewrites passed the checks")


def write_examples(train, rewrites, path, per_class=3):
    """Counts per class, then kept rewrites and (if any) one rejected rewrite next to their originals."""
    real = train["label"].value_counts()
    made = rewrites["label"].value_counts()
    kept = rewrites.loc[rewrites["kept"], "label"].value_counts()
    lines = ["# Data augmentation: original vs rewritten reports", "",
             f"Language model: `{config.AUGMENT_LLM}`. Target per class: {config.AUGMENT_TARGET} training reports. "
             f"Label check: {'on' if config.AUGMENT_LABEL_CHECK else 'off'}.", "",
             "| Class | Real | Rewritten | Passed label check | Used for training |", "|---|---:|---:|---:|---:|"]
    for label in real.index:
        lines.append(f"| {label} | {real[label]} | {made.get(label, 0)} | {kept.get(label, 0)} | "
                     f"{real[label] + kept.get(label, 0)} |")
    if len(rewrites):
        lines += ["", f"Overall {rewrites['kept'].sum()} of {len(rewrites)} rewrites "
                      f"({100 * rewrites['kept'].mean():.1f}%) passed the label check."]
    for label, group in rewrites.groupby("label"):
        lines += ["", f"## {label}"]
        shown = pd.concat([group[group["kept"]].head(per_class), group[~group["kept"]].head(1)])
        for _, row in shown.iterrows():
            title = "Rewrite" if row["kept"] else "Rewrite, REJECTED by the label check"
            lines += ["", f"**Original:** {train.loc[int(row['source']), 'text']}", "",
                      f"**{title}:** {row['text']}", "", "---"]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    train, _, _, labels = load_split()                               # validation and test are not even loaded
    out_file = Path(config.DATA_DIR) / "augmented.csv"
    keeps_label = label_checker(train)
    augment(train, labels, out_file, keeps_label)

    rewrites = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame(columns=["text", "label", "source"])
    rewrites["kept"] = keeps_label(rewrites["text"], rewrites["label"]) if len(rewrites) else []
    kept = rewrites[rewrites["kept"]]
    combined = pd.concat([train[["text", "label"]], kept[["text", "label"]]], ignore_index=True)
    combined.to_csv(Path(config.DATA_DIR) / "train_augmented.csv", index=False)
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    write_examples(train, rewrites, Path(config.REPORTS_DIR) / "augmentation_examples.md")

    print(f"\n{len(kept)} of {len(rewrites)} rewrites passed the label check")
    print(f"{len(train)} real + {len(kept)} new = {len(combined)} training reports "
          f"-> {config.DATA_DIR}/train_augmented.csv")
    print(combined["label"].value_counts().to_string())
    print(f"Check the rewrites by eye: {config.REPORTS_DIR}/augmentation_examples.md")
    print("Next: python step2_tfidf_svm.py   and   python step4_run_experiments.py --augment --seeds 1 2")
