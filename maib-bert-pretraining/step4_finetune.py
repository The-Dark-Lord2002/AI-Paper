"""
STEP 4 - Stage 2, supervised: fine-tune ONE configuration on the labelled dataset.

Exactly the training of the paper (maib-thesis/step3_train.py); only the encoder can change:

  python step4_finetune.py --model bert_bilstm_att --loss wce --seed 1                                 # original BERT
  python step4_finetune.py --model bert_bilstm_att --loss wce --seed 1 --encoder models/bert-maib-external

  1. split every report into word-pieces (at most config.MAX_TOKENS)
  2. fine-tune the encoder + the model head with the chosen loss
  3. after every epoch, measure macro-F1 on the VALIDATION set and remember the best epoch
  4. go back to the best epoch and evaluate ONCE on the TEST set
  5. save to results/<encoder>__<model>__<loss>__seed<seed>.json   (encoder: "base" = original BERT)
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config
from common import compute_metrics, load_split, save_json, set_seed
from losses import LOSS_NAMES, make_loss
from models import MODEL_NAMES, IncidentClassifier


def encoder_name(encoder):
    """Short name used in file names and tables: "base" for the original BERT, else the model folder's name."""
    return "base" if encoder == config.BASE_MODEL else Path(encoder).name


def run_name(encoder, model_name, loss_name, seed):
    return f"{encoder_name(encoder)}__{model_name}__{loss_name}__seed{seed}"


def make_loader(tokenizer, df, label_to_id, batch_size, shuffle, seed=0):
    """Tokenise the reports once; each batch is padded to its longest report."""
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
            attention_mask[i, :n] = 1
        labels = torch.tensor([ex["label"] for ex in batch])
        return input_ids, attention_mask, labels

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator)


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    predictions, truth = [], []
    for input_ids, attention_mask, labels in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(input_ids.to(device), attention_mask.to(device))
        predictions += logits.argmax(dim=-1).tolist()
        truth += labels.tolist()
    return np.array(truth), np.array(predictions)


def train(model_name, loss_name, seed, encoder=config.BASE_MODEL, batch_size=config.BATCH_SIZE):
    name = run_name(encoder, model_name, loss_name, seed)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    grad_accum = max(1, config.EFFECTIVE_BATCH // batch_size)
    print(f"\n===== {name} =====  encoder={encoder}  device={device}  batch={batch_size} x {grad_accum}")

    # ------------------------------------------------------------------ data
    train_df, val_df, test_df, labels = load_split()
    label_to_id = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    train_loader = make_loader(tokenizer, train_df, label_to_id, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(tokenizer, val_df, label_to_id, config.EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_loader(tokenizer, test_df, label_to_id, config.EVAL_BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------ model, loss, optimiser
    model = IncidentClassifier(model_name, len(labels), encoder=encoder).to(device)
    class_counts = np.bincount(train_df["label"].map(label_to_id), minlength=len(labels))
    loss_fn = make_loss(loss_name, class_counts).to(device)
    bert_params = list(model.bert.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    optimizer = torch.optim.AdamW([{"params": bert_params, "lr": config.LR_BERT},
                                   {"params": head_params, "lr": config.LR_HEAD}],
                                  weight_decay=config.WEIGHT_DECAY)
    total_updates = -(-len(train_loader) // grad_accum) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.WARMUP_FRACTION * total_updates),
                                                total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------ training loop
    best_f1, best_epoch, best_weights, history = -1.0, 0, None, []
    start = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.EPOCHS}", leave=False, mininterval=2)
        for step, (input_ids, attention_mask, labels_batch) in enumerate(progress, start=1):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels_batch = labels_batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels_batch)
            scaler.scale(loss / grad_accum).backward()
            running_loss += loss.item()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        y_val, p_val = predict(model, val_loader, device, use_amp)
        val_f1 = compute_metrics(y_val, p_val, labels)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader), "val_macro_f1": val_f1})
        print(f"[epoch {epoch}] train loss={running_loss / len(train_loader):.4f}  "
              f"val macro-F1={val_f1:.4f}  ({(time.time() - start) / 60:.1f} min)")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        elif epoch - best_epoch >= config.PATIENCE:
            print(f"[early stop] no improvement for {config.PATIENCE} epochs")
            break

    # ------------------------------------------------------------------ final evaluation (once)
    model.load_state_dict(best_weights)
    y_val, p_val = predict(model, val_loader, device, use_amp)
    y_test, p_test = predict(model, test_loader, device, use_amp)
    test_metrics = compute_metrics(y_test, p_test, labels)
    print(f"[test] macro-F1={test_metrics['macro_f1']:.4f}  accuracy={test_metrics['accuracy']:.4f}  "
          f"(best epoch {best_epoch})")

    result = {
        "run": name, "encoder": encoder_name(encoder), "model": model_name, "loss": loss_name, "seed": seed,
        "tag": "",
        "settings": {"bert": encoder, "batch_size": batch_size, "grad_accum": grad_accum,
                     "lr_bert": config.LR_BERT, "lr_head": config.LR_HEAD, "epochs": config.EPOCHS,
                     "patience": config.PATIENCE, "train_reports": len(train_df),
                     "max_tokens": config.MAX_TOKENS, "device": str(device)},
        "labels": labels,
        "best_epoch": best_epoch,
        "train_minutes": round((time.time() - start) / 60, 1),
        "history": history,
        "val": compute_metrics(y_val, p_val, labels),
        "test": test_metrics,
        "test_predictions": p_test.tolist(),
    }
    save_json(result, Path(config.RESULTS_DIR) / f"{name}.json")

    del model, optimizer, best_weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--loss", choices=LOSS_NAMES, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--encoder", default=config.BASE_MODEL,
                        help="bert-base-uncased (default) or a model folder of step 3, e.g. models/bert-maib-external")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()
    train(args.model, args.loss, args.seed, encoder=args.encoder, batch_size=args.batch_size)
