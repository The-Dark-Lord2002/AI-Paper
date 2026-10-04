"""
تفسیرپذیری: کدام کلمات گزارش بیشترین وزن توجه را در تصمیم مدل داشته‌اند؟

پیش‌نیاز: یک مدل attention با --keep_ckpt آموزش داده شده باشد:
    python train.py --head bilstm_att --loss focal --seed 1 --keep_ckpt
    python explain_attention.py --ckpt checkpoints/bert-base-uncased__bilstm_att__focal__s1.pt

خروجی:
  reports/attention_examples.html : گزارش‌های test با رنگ‌آمیزی کلمات بر اساس وزن توجه
  reports/top_words_per_class.csv : پروزن‌ترین کلمات برای هر کلاس (تجمیع روی test)
"""
import argparse
import html
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from maib.data import load_splits
from maib.model import IncidentClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--model", default="bert-base-uncased")
ap.add_argument("--head", default="bilstm_att", choices=["attention", "bilstm_att"])
ap.add_argument("--split_dir", default="splits")
ap.add_argument("--max_len", type=int, default=192)
ap.add_argument("--n_examples", type=int, default=3, help="examples per class in the HTML")
ap.add_argument("--top_k", type=int, default=15)
ap.add_argument("--out", default="reports")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_df, labels = load_splits(args.split_dir)
tok = AutoTokenizer.from_pretrained(args.model)
model = IncidentClassifier(args.model, len(labels), args.head).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()

STOP = {"[CLS]", "[SEP]", "<s>", "</s>", ".", ",", "the", "a", "an", "of", "to", "and",
        "was", "were", "in", "on", "at", "with", "when", "had", "their", "they", "it", "its"}


def word_attention(text):
    """وزن توجه را از سطح زیرواژه (WordPiece) به سطح کلمه جمع می‌زند."""
    enc = tok(text, truncation=True, max_length=args.max_len, return_tensors="pt")
    word_ids = enc.word_ids()
    with torch.no_grad():
        logits, alpha = model(**{k: v.to(device) for k, v in enc.items()}, return_attention=True)
    alpha = alpha[0].float().cpu().numpy()
    words, weights = defaultdict(str), defaultdict(float)
    for t, (wid, a) in enumerate(zip(word_ids, alpha)):
        if wid is None:
            continue
        span = enc.word_to_chars(0, wid)
        words[wid] = text[span.start:span.end]
        weights[wid] += float(a)
    order = sorted(words)
    return [words[i] for i in order], [weights[i] for i in order], int(logits.argmax(-1))


per_class_words = defaultdict(Counter)
blocks = []
shown = Counter()
for text, gold in zip(test_df["text"], test_df["label"]):
    ws, ats, pred = word_attention(text)
    for w, a in zip(ws, ats):
        lw = w.lower().strip(".,;:()'\"")
        if lw and lw not in STOP and not lw.isdigit():
            per_class_words[labels[pred]][lw] += a
    if shown[gold] < args.n_examples:
        shown[gold] += 1
        mx = max(ats) or 1.0
        spans = " ".join(
            f'<span style="background: rgba(220,60,40,{a/mx:.2f})">{html.escape(w)}</span>'
            for w, a in zip(ws, ats))
        ok = "✓" if labels[pred] == gold else "✗"
        blocks.append(f"<div class='ex'><b>gold:</b> {html.escape(gold)} &nbsp; "
                      f"<b>pred:</b> {html.escape(labels[pred])} {ok}<p>{spans}</p></div>")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "attention_examples.html").write_text(
    "<!doctype html><meta charset='utf-8'><title>Attention examples</title>"
    "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;line-height:1.8}"
    ".ex{border-bottom:1px solid #ddd;padding:1em 0}</style>"
    "<h2>Attention weights over MAIB test reports</h2>" + "\n".join(blocks))

rows = []
for lab in labels:
    for rank, (w, s) in enumerate(per_class_words[lab].most_common(args.top_k), 1):
        rows.append({"class": lab, "rank": rank, "word": w, "attention_mass": round(s, 3)})
pd.DataFrame(rows).to_csv(out / "top_words_per_class.csv", index=False)
print(f"Saved {out}/attention_examples.html and {out}/top_words_per_class.csv")
