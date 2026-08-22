"""Fine-tune the BERT + BiLSTM classifier on the MAIB incident-reports dataset.

Full-text access to Xu et al. (2025) was not available in this environment
(MDPI/ResearchGate egress was blocked), so exact paper hyperparameters could
not be confirmed. This script uses standard BERT fine-tuning defaults and
exposes every hyperparameter as a CLI flag so they can be corrected to match
the paper if/when its exact settings are available. See README.md.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

from dataset import MAIBTextDataset, load_maib_splits
from model import BertBiLSTMClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bert-name", default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--freeze-bert", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bert-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="checkpoints")
    return p.parse_args()


def run_epoch(model, loader, device, optimizer=None, scheduler=None, max_grad_norm=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, all_preds, all_labels


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits, label_encoder = load_maib_splits(
        seed=args.seed, val_size=args.val_size, test_size=args.test_size
    )
    num_classes = len(label_encoder.classes_)
    print(f"Classes ({num_classes}): {list(label_encoder.classes_)}")
    for name, (texts, _) in splits.items():
        print(f"  {name}: {len(texts)} examples")

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_name)

    train_ds = MAIBTextDataset(*splits["train"], tokenizer, args.max_length)
    val_ds = MAIBTextDataset(*splits["val"], tokenizer, args.max_length)
    test_ds = MAIBTextDataset(*splits["test"], tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = BertBiLSTMClassifier(
        num_classes=num_classes,
        bert_name=args.bert_name,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
    ).to(device)

    head_params = list(model.bilstm.parameters()) + list(model.classifier.parameters())
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.bert_lr},
            {"params": head_params, "lr": args.head_lr},
        ]
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "label_classes.json"), "w") as f:
        json.dump(list(label_encoder.classes_), f)

    best_val_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, device, optimizer, scheduler, args.max_grad_norm
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved new best checkpoint (val_f1={val_f1:.4f})")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_f1, test_preds, test_labels = run_epoch(model, test_loader, device)
    print(f"\nTest results: loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_f1:.4f}\n")
    print(
        classification_report(
            test_labels, test_preds, target_names=label_encoder.classes_, digits=4
        )
    )


if __name__ == "__main__":
    main()
