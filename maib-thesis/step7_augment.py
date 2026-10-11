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

  python step7_augment.py

The first run downloads the language model (about 3 GB). Stopping and starting again is safe: the rewrites
are saved after every batch and the script continues where it stopped.
If the GPU runs out of memory, use the smaller "Qwen/Qwen2.5-0.5B-Instruct" in config.AUGMENT_LLM.

Output:
  data/augmented.csv                 only the new reports (text, label, source = row of the original in train.csv)
  data/train_augmented.csv           real training reports + new reports; used by --augment in steps 2 and 4
  reports/augmentation_examples.md   counts per class and original vs rewrite, to check by eye and for the thesis
"""
import random
import re
from pathlib import Path

import pandas as pd
import torch
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


def augment(train, labels, out_file, rewrite=None):
    """Rewrite reports until every class has config.AUGMENT_TARGET training reports. Appends to out_file."""
    done = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame(columns=["text", "label", "source"])
    seen = set(train["text"].str.lower()) | set(done["text"].str.lower())
    counts = train["label"].value_counts()
    need = {label: int(config.AUGMENT_TARGET - counts[label] - (done["label"] == label).sum()) for label in labels}
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
            new_rows = []
            for row, original, new in zip(rows, originals, rewrite(originals)):
                new = clean(new)
                if made + len(new_rows) < n_needed and acceptable(new, original, seen):
                    seen.add(new.lower())
                    new_rows.append({"text": new, "label": label, "source": row})
            if new_rows:                                             # save after every batch
                pd.DataFrame(new_rows).to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
                made += len(new_rows)
                progress.update(len(new_rows))
        progress.close()
        if made < n_needed:
            print(f"[warning] {label}: only {made} of {n_needed} rewrites passed the checks")


def write_examples(train, augmented, path, per_class=3):
    """Counts per class and a few originals next to their rewrites."""
    real = train["label"].value_counts()
    new = augmented["label"].value_counts()
    lines = ["# Data augmentation: original vs rewritten reports", "",
             f"Language model: `{config.AUGMENT_LLM}`. Target per class: {config.AUGMENT_TARGET} training reports.", "",
             "| Class | Real | New | Total |", "|---|---:|---:|---:|"]
    for label in real.index:
        lines.append(f"| {label} | {real[label]} | {new.get(label, 0)} | {real[label] + new.get(label, 0)} |")
    for label, group in augmented.groupby("label"):
        lines += ["", f"## {label}"]
        for _, row in group.head(per_class).iterrows():
            lines += ["", f"**Original:** {train.loc[int(row['source']), 'text']}", "",
                      f"**Rewrite:** {row['text']}", "", "---"]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    train, _, _, labels = load_split()                               # validation and test are not even loaded
    out_file = Path(config.DATA_DIR) / "augmented.csv"
    augment(train, labels, out_file)

    augmented = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame(columns=["text", "label", "source"])
    combined = pd.concat([train[["text", "label"]], augmented[["text", "label"]]], ignore_index=True)
    combined.to_csv(Path(config.DATA_DIR) / "train_augmented.csv", index=False)
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    write_examples(train, augmented, Path(config.REPORTS_DIR) / "augmentation_examples.md")

    print(f"\n{len(train)} real + {len(augmented)} new = {len(combined)} training reports "
          f"-> {config.DATA_DIR}/train_augmented.csv")
    print(combined["label"].value_counts().to_string())
    print(f"Check the rewrites by eye: {config.REPORTS_DIR}/augmentation_examples.md")
    print("Next: python step2_tfidf_svm.py   and   python step4_run_experiments.py --augment --seeds 1 2")
