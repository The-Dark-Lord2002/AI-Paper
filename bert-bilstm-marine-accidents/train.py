"""Fine-tune the BERT + BiLSTM classifier on the MAIB incident-reports dataset.

Hyperparameter defaults below are taken directly from Table 9 ("Model
parameter settings") of Zhao et al. (2025), "Causation Analysis of Marine
Traffic Accidents Using Deep Learning Approaches: A Case Study from China's
Coasts" (Systems, 13(4):284), and from Section 4.1 (train/val/test split) and
Section 4.2 / Figure 7 (epoch count) of the same paper. See README.md for the
full mapping between each flag and where it comes from in the paper, and for
the caveat that these were tuned on the paper's own 32-class, ~26k-example
dataset rather than the ~5.8k-example, single-label MAIB dataset used here.

If you hit "CUDA out of memory" on a smaller GPU than the paper's (RTX
4060 Ti, 8GB), lower --batch-size and raise --grad-accum-steps by the same
factor to keep the *effective* batch size at 32 -- e.g. --batch-size 4
--grad-accum-steps 8. Mixed-precision training is on by default on CUDA
(disable with --no-amp) and roughly halves activation memory.
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
from transformers import BertTokenizerFast

from dataset import MAIBTextDataset, load_maib_splits
from model import BertBiLSTMClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bert-name", default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--freeze-bert", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="accumulate gradients over this many batches before each optimizer "
        "step, to simulate a larger effective batch size on limited GPU memory",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--l2-weight-decay", type=float, default=0.05)
    p.add_argument("--l1-lambda", type=float, default=5e-10)
    p.add_argument("--max-grad-norm", type=float, default=2.75)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="disable automatic mixed precision (on by default on CUDA)",
    )
    return p.parse_args()


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    max_grad_norm=2.75,
    l1_lambda=0.0,
    grad_accum_steps=1,
    scaler=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    use_amp = scaler is not None and scaler.is_enabled()

    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = len(loader)

    if is_train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                if is_train and l1_lambda > 0:
                    l1_penalty = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_penalty

            if is_train:
                scaled_loss = loss / grad_accum_steps
                if use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_last_batch = (step + 1) == num_batches
                if (step + 1) % grad_accum_steps == 0 or is_last_batch:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad()

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
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Using device: {device} (mixed precision: {use_amp})")

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

    # Single learning rate across all parameters + AdamW's decoupled weight decay
    # as the L2 term, matching Table 9. The L1 term (--l1-lambda) is added
    # manually to the loss inside run_epoch since AdamW has no native L1 option.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "label_classes.json"), "w") as f:
        json.dump(list(label_encoder.classes_), f)

    best_val_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.max_grad_norm,
            args.l1_lambda,
            args.grad_accum_steps,
            scaler,
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
