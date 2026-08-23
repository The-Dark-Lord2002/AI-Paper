"""Evaluate the plain-BERT baseline and BERT+BiLSTM model side by side on the
same held-out test split, reproducing (on this dataset) the ablation
comparison the paper makes on its own in Table 10: does BiLSTM actually
improve on BERT alone?

Run train_bert_baseline.py and train.py first (in that order or either
order -- they write to separate --output-dir paths by default:
checkpoints_bert_baseline/ and checkpoints/). Then:

    uv run compare_models.py

--seed/--val-size/--test-size must match what the two training runs used
(the defaults already do) so this reconstructs the identical test split
neither model trained or was tuned on.
"""
import argparse
import json

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from baseline_model import BertClassifier
from dataset import MAIBTextDataset, load_maib_splits
from model import BertBiLSTMClassifier
from training_utils import run_epoch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bert-name", default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--baseline-dir", default="checkpoints_bert_baseline")
    p.add_argument("--baseline-hidden-dim", type=int, default=512)
    p.add_argument("--baseline-dropout", type=float, default=0.3)

    p.add_argument("--bilstm-dir", default="checkpoints")
    p.add_argument("--bilstm-hidden", type=int, default=128)
    p.add_argument("--bilstm-layers", type=int, default=1)
    p.add_argument("--bilstm-dropout", type=float, default=0.3)

    p.add_argument("--report-path", default="comparison_report.md")
    return p.parse_args()


def load_classes(checkpoint_dir):
    with open(f"{checkpoint_dir}/label_classes.json") as f:
        return json.load(f)


def evaluate(model, loader, device, classes):
    _, acc, macro_f1, preds, labels = run_epoch(model, loader, device)
    all_labels = list(range(len(classes)))
    report = classification_report(
        labels,
        preds,
        labels=all_labels,
        target_names=classes,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        labels, preds, labels=all_labels, target_names=classes, digits=4, zero_division=0
    )
    return acc, macro_f1, report, report_text


def fmt(x):
    return f"{x:.4f}"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    baseline_classes = load_classes(args.baseline_dir)
    bilstm_classes = load_classes(args.bilstm_dir)
    if baseline_classes != bilstm_classes:
        raise ValueError(
            "The two checkpoints were trained with different label sets "
            f"({args.baseline_dir} vs {args.bilstm_dir}) -- retrain both with the "
            "same --seed so they see the same class encoding."
        )
    classes = baseline_classes
    num_classes = len(classes)

    splits, label_encoder = load_maib_splits(
        seed=args.seed, val_size=args.val_size, test_size=args.test_size
    )
    if list(label_encoder.classes_) != classes:
        raise ValueError(
            "The dataset's current class list doesn't match the checkpoints' "
            "label_classes.json -- make sure --seed matches what training used."
        )

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_name)
    test_ds = MAIBTextDataset(*splits["test"], tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test set: {len(test_ds)} examples, {num_classes} classes")

    baseline_model = BertClassifier(
        num_classes=num_classes,
        bert_name=args.bert_name,
        hidden_dim=args.baseline_hidden_dim,
        dropout=args.baseline_dropout,
    ).to(device)
    baseline_model.load_state_dict(
        torch.load(f"{args.baseline_dir}/best_model.pt", map_location=device)
    )

    bilstm_model = BertBiLSTMClassifier(
        num_classes=num_classes,
        bert_name=args.bert_name,
        lstm_hidden=args.bilstm_hidden,
        lstm_layers=args.bilstm_layers,
        dropout=args.bilstm_dropout,
    ).to(device)
    bilstm_model.load_state_dict(
        torch.load(f"{args.bilstm_dir}/best_model.pt", map_location=device)
    )

    print("\nEvaluating plain-BERT baseline...")
    base_acc, base_f1, base_report, base_text = evaluate(baseline_model, test_loader, device, classes)
    print(base_text)

    print("Evaluating BERT + BiLSTM...")
    bilstm_acc, bilstm_f1, bilstm_report, bilstm_text = evaluate(bilstm_model, test_loader, device, classes)
    print(bilstm_text)

    summary_rows = [
        ("Accuracy", base_report["accuracy"], bilstm_report["accuracy"]),
        ("Macro avg F1", base_report["macro avg"]["f1-score"], bilstm_report["macro avg"]["f1-score"]),
        ("Weighted avg F1", base_report["weighted avg"]["f1-score"], bilstm_report["weighted avg"]["f1-score"]),
    ]

    print("\n=== BERT vs. BERT+BiLSTM on the MAIB test set ===")
    print(f"{'Metric':<20}{'BERT':>10}{'BERT+BiLSTM':>15}{'Delta':>10}")
    for name, base_val, bilstm_val in summary_rows:
        print(f"{name:<20}{fmt(base_val):>10}{fmt(bilstm_val):>15}{fmt(bilstm_val - base_val):>10}")

    per_class_lines = []
    for cls in classes:
        b = base_report[cls]["f1-score"]
        s = bilstm_report[cls]["f1-score"]
        per_class_lines.append((cls, b, s, s - b))

    with open(args.report_path, "w") as f:
        f.write("# BERT vs. BERT+BiLSTM comparison\n\n")
        f.write(f"Test set: {len(test_ds)} examples, {num_classes} classes.\n\n")
        f.write("| Metric | BERT | BERT+BiLSTM | Delta |\n|---|---|---|---|\n")
        for name, base_val, bilstm_val in summary_rows:
            f.write(f"| {name} | {fmt(base_val)} | {fmt(bilstm_val)} | {fmt(bilstm_val - base_val)} |\n")
        f.write("\n## Per-class F1\n\n")
        f.write("| Class | BERT | BERT+BiLSTM | Delta |\n|---|---|---|---|\n")
        for cls, b, s, d in per_class_lines:
            f.write(f"| {cls} | {fmt(b)} | {fmt(s)} | {fmt(d)} |\n")

    print(f"\nSaved full comparison report to {args.report_path}")


if __name__ == "__main__":
    main()
