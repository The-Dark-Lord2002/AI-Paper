"""Training-loop utilities shared by the plain-BERT and BERT+BiLSTM training
cells in Thesis_Progress_Full_Pipeline.ipynb, so both models run the exact
same training/evaluation implementation and only differ in which model is built.
"""
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    """Run one training or evaluation pass over `loader`.

    Pass `optimizer` (and optionally `scaler`) to train; omit both to evaluate.
    Supports gradient accumulation (simulate a larger effective batch on a
    smaller GPU) and automatic mixed precision.
    """
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
