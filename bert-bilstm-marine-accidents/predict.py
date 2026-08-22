"""Classify new incident narratives with a trained BERT-BiLSTM checkpoint.

Example:
    python predict.py --checkpoint-dir checkpoints \
        "A bulk carrier ran aground after losing steering control in heavy weather."
"""
import argparse
import json

import torch
from transformers import BertTokenizerFast

from dataset import clean_text
from model import BertBiLSTMClassifier


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--bert-name", default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("text", nargs="+", help="One or more incident narratives to classify")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{args.checkpoint_dir}/label_classes.json") as f:
        classes = json.load(f)

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_name)
    model = BertBiLSTMClassifier(
        num_classes=len(classes),
        bert_name=args.bert_name,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(
        torch.load(f"{args.checkpoint_dir}/best_model.pt", map_location=device)
    )
    model.eval()

    for text in args.text:
        encoding = tokenizer(
            clean_text(text),
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(probs.argmax().item())

        print(f"\nText: {text}")
        print(f"Predicted: {classes[pred_idx]}  (confidence={probs[pred_idx]:.3f})")


if __name__ == "__main__":
    main()
