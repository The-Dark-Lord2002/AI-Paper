"""Fine-tune the plain-BERT baseline on the MAIB incident-reports dataset.

This is the paper's "BERT" ablation column from Table 9 -- the same task,
same dataset splits, same training loop as train.py (BERT + BiLSTM), but
without the BiLSTM stage. Training both and running compare_models.py
reproduces, on this dataset, the comparison the paper itself makes on its
own dataset (Table 10): does BiLSTM actually help?

Hyperparameter defaults below are taken directly from the "BERT" column of
Table 9 ("Model parameter settings") of Zhao et al. (2025): hidden layer
512, learning rate 1e-6, 3 dropout layers, L1 1e-8, L2 0.05, gradient clip
2.35, AdamW, GELU activation, batch size 32. See baseline_model.py for the
one thing Table 9 doesn't fully specify (how those 3 dropout layers and the
512-dim hidden layer are laid out) and README.md for the full mapping.

Same GPU-memory notes as train.py apply: on a smaller GPU than the paper's,
use --batch-size/--grad-accum-steps to keep the same effective batch size
without running out of memory, e.g. --batch-size 4 --grad-accum-steps 8.
"""
import argparse
import json
import os

import torch
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from baseline_model import BertClassifier
from dataset import MAIBTextDataset, load_maib_splits
from training_utils import run_epoch, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bert-name", default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=512)
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
    p.add_argument("--l1-lambda", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=2.35)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="checkpoints_bert_baseline")
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="disable automatic mixed precision (on by default on CUDA)",
    )
    return p.parse_args()


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

    model = BertClassifier(
        num_classes=num_classes,
        bert_name=args.bert_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
    ).to(device)

    # Single learning rate across all parameters + AdamW's decoupled weight decay
    # as the L2 term, matching Table 9's "BERT" column. The L1 term
    # (--l1-lambda) is added manually to the loss inside run_epoch since
    # AdamW has no native L1 option.
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
