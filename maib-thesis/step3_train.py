"""
STEP 3 - Train and evaluate ONE BERT configuration (used for proposal steps 2b, 2c, 3 and 4).

  python step3_train.py --model bert_bilstm_att --loss focal --seed 1

What happens:
  1. split every report into BERT word-pieces (at most config.MAX_TOKENS)
  2. fine-tune BERT + the chosen model head with the chosen loss
  3. after every epoch, measure macro-F1 on the VALIDATION set and remember the best epoch
  4. go back to the best epoch and evaluate ONCE on the TEST set
  5. save everything to results/<model>__<loss>__seed<seed>.json

The test set never influences any decision (which epoch, which settings). That is what makes the
test score an honest estimate.
"""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config
from common import compute_metrics, load_split, save_json
from losses import LOSS_NAMES, make_loss
from models import MODEL_NAMES, IncidentClassifier


def run_name(model_name, loss_name, seed, tag=""):
    return f"{model_name}__{loss_name}__seed{seed}" + (f"__{tag}" if tag else "")


def set_seed(seed):
    """Same seed -> same initial weights, same dropout, same batch order -> same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(tokenizer, df, label_to_id, batch_size, shuffle, seed=0, oversample=False):
    """
    Tokenise the reports once and return a DataLoader that pads each batch to its longest report.
    oversample=True: draw reports with probability 1 / (size of their class), with replacement, so every
    class appears about equally often. The epoch still has len(df) reports: rare reports are repeated,
    some reports of the big classes are skipped in that epoch.
    """
    encoded = tokenizer(df["text"].tolist(), truncation=True, max_length=config.MAX_TOKENS)
    examples = [{"input_ids": ids, "label": label_to_id[label]}
                for ids, label in zip(encoded["input_ids"], df["label"])]

    def collate(batch):
        longest = max(len(ex["input_ids"]) for ex in batch)
        input_ids = torch.full((len(batch), longest), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            input_ids[i, :n] = torch.tensor(ex["input_ids"])
            attention_mask[i, :n] = 1                       # 1 = real word-piece, 0 = padding
        labels = torch.tensor([ex["label"] for ex in batch])
        return input_ids, attention_mask, labels

    generator = torch.Generator().manual_seed(seed)         # makes the shuffling order reproducible
    if oversample:
        class_size = df["label"].map(df["label"].value_counts()).values
        sampler = WeightedRandomSampler(weights=1.0 / class_size, num_samples=len(examples),
                                        replacement=True, generator=generator)
        return DataLoader(examples, batch_size=batch_size, sampler=sampler, collate_fn=collate)
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator)


@torch.no_grad()
def predict(model, loader, device, use_amp, tokenizer=None):
    """Predicted and true class ids for a whole set. With a tokenizer, also returns attention per report."""
    model.eval()
    predictions, truth, attention = [], [], []
    for input_ids, attention_mask, labels in loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, alpha = model(input_ids, attention_mask)
        predictions += logits.argmax(dim=-1).tolist()
        truth += labels.tolist()
        if tokenizer is not None and alpha is not None:
            for ids, mask, weights in zip(input_ids, attention_mask, alpha):
                n = int(mask.sum())
                attention.append({"tokens": tokenizer.convert_ids_to_tokens(ids[:n].tolist()),
                                  "weights": [round(w, 5) for w in weights[:n].tolist()]})
    return np.array(truth), np.array(predictions), attention


def train(model_name, loss_name, seed, batch_size=config.BATCH_SIZE, lr_bert=config.LR_BERT,
          lr_head=config.LR_HEAD, epochs=config.EPOCHS, patience=config.PATIENCE, oversample=False, tag=""):
    name = run_name(model_name, loss_name, seed, tag)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"            # mixed precision (float16) on the GPU: faster, half the memory
    grad_accum = max(1, config.EFFECTIVE_BATCH // batch_size)
    print(f"\n===== {name} =====  device={device}  batch={batch_size} x {grad_accum} accumulation steps")

    # ------------------------------------------------------------------ data
    train_df, val_df, test_df, labels = load_split()
    label_to_id = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_NAME)
    train_loader = make_loader(tokenizer, train_df, label_to_id, batch_size, shuffle=True, seed=seed,
                               oversample=oversample)
    val_loader = make_loader(tokenizer, val_df, label_to_id, config.EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_loader(tokenizer, test_df, label_to_id, config.EVAL_BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------ model, loss, optimiser
    model = IncidentClassifier(model_name, len(labels)).to(device)
    class_counts = np.bincount(train_df["label"].map(label_to_id), minlength=len(labels))
    loss_fn = make_loss(loss_name, class_counts).to(device)

    # BERT is already trained -> small learning rate. The new layers start from random -> larger one.
    bert_params = list(model.bert.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    optimizer = torch.optim.AdamW([{"params": bert_params, "lr": lr_bert},
                                   {"params": head_params, "lr": lr_head}],
                                  weight_decay=config.WEIGHT_DECAY)
    updates_per_epoch = -(-len(train_loader) // grad_accum)          # ceiling division
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.WARMUP_FRACTION * total_updates),
                                                total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)          # protects tiny float16 gradients

    # ------------------------------------------------------------------ training loop
    best_f1, best_epoch, best_weights, history = -1.0, 0, None, []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, mininterval=2)
        for step, (input_ids, attention_mask, labels_batch) in enumerate(progress, start=1):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels_batch = labels_batch.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels_batch)
            scaler.scale(loss / grad_accum).backward()               # gradients add up over grad_accum batches
            running_loss += loss.item()

            if step % grad_accum == 0 or step == len(train_loader):  # one weight update per EFFECTIVE_BATCH
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        y_val, p_val, _ = predict(model, val_loader, device, use_amp)
        val_f1 = compute_metrics(y_val, p_val, labels)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader), "val_macro_f1": val_f1})
        print(f"[epoch {epoch}] train loss={running_loss / len(train_loader):.4f}  "
              f"val macro-F1={val_f1:.4f}  ({(time.time() - start) / 60:.1f} min)")

        if val_f1 > best_f1:                                         # new best: keep a copy of the weights
            best_f1, best_epoch = val_f1, epoch
            best_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        elif epoch - best_epoch >= patience:
            print(f"[early stop] no improvement for {patience} epochs")
            break

    # ------------------------------------------------------------------ final evaluation (once)
    model.load_state_dict(best_weights)
    y_val, p_val, _ = predict(model, val_loader, device, use_amp)
    keep_attention = tokenizer if model_name == "bert_bilstm_att" else None
    y_test, p_test, attention = predict(model, test_loader, device, use_amp, tokenizer=keep_attention)
    test_metrics = compute_metrics(y_test, p_test, labels)
    print(f"[test] macro-F1={test_metrics['macro_f1']:.4f}  accuracy={test_metrics['accuracy']:.4f}  "
          f"(best epoch {best_epoch})")

    result = {
        "run": name, "model": model_name, "loss": loss_name, "seed": seed, "tag": tag,
        "settings": {"bert": config.BERT_NAME, "batch_size": batch_size, "grad_accum": grad_accum,
                     "lr_bert": lr_bert, "lr_head": lr_head, "epochs": epochs, "patience": patience,
                     "oversample": oversample, "max_tokens": config.MAX_TOKENS, "device": str(device)},
        "labels": labels,
        "best_epoch": best_epoch,
        "train_minutes": round((time.time() - start) / 60, 1),
        "history": history,
        "val": compute_metrics(y_val, p_val, labels),
        "test": test_metrics,
        "test_predictions": p_test.tolist(),
        "test_attention": attention,                  # filled only for bert_bilstm_att, used in step 6
    }
    save_json(result, Path(config.RESULTS_DIR) / f"{name}.json")

    del model, optimizer, best_weights                # free GPU memory before the next run
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--loss", choices=LOSS_NAMES, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--oversample", action="store_true", help="show rare classes more often (see config.py)")
    args = parser.parse_args()
    train(args.model, args.loss, args.seed, batch_size=args.batch_size,
          oversample=args.oversample, tag="oversample" if args.oversample else "")
