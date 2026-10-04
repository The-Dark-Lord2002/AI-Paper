"""
آموزش یک پیکربندی: (مدل، head، loss، seed).

مثال (لپ‌تاپ با GPU 4GB):
    python train.py --head bilstm_att --loss focal --seed 1     # روش پیشنهادی
    python train.py --head bilstm --loss ce --seed 1            # بازتولید مقاله‌ی مرجع
    python train.py --head cls --loss ce --seed 1               # مبنای BERT ساده

خروجی: results/<run_name>.json  (معیارهای val و test، ماتریس درهم‌ریختگی، تنظیمات)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from maib.data import load_splits
from maib.losses import LOSSES, build_loss
from maib.metrics import compute_metrics
from maib.model import HEADS, IncidentClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_name(cfg) -> str:
    m = cfg["model"].rstrip("/").split("/")[-1]
    return f"{m}__{cfg['head']}__{cfg['loss']}__s{cfg['seed']}"


class TextDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["labels"] = self.labels[i]
        return item


def make_loader(tok, df, label2id, max_len, batch, shuffle, seed):
    enc = tok(df["text"].tolist(), truncation=True, max_length=max_len)
    labels = [label2id[l] for l in df["label"]]

    def collate(items):
        feats = [{k: v for k, v in it.items() if k != "labels"} for it in items]
        out = tok.pad(feats, return_tensors="pt")
        out["labels"] = torch.tensor([it["labels"] for it in items])
        return out

    g = torch.Generator().manual_seed(seed)
    return DataLoader(TextDS(enc, labels), batch_size=batch, shuffle=shuffle,
                      collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    preds, gold = [], []
    for b in loader:
        y = b.pop("labels")
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(**b)
        preds.append(logits.argmax(-1).cpu())
        gold.append(y)
    return torch.cat(gold).numpy(), torch.cat(preds).numpy()


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not cfg.get("no_amp", False)

    train_df, val_df, test_df, labels = load_splits(cfg["split_dir"])
    label2id = {l: i for i, l in enumerate(labels)}
    counts = np.bincount([label2id[l] for l in train_df["label"]], minlength=len(labels))

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    n_trunc = sum(len(x) > cfg["max_len"] for x in tok(train_df["text"].tolist())["input_ids"])
    print(f"[data] train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
          f"truncated at {cfg['max_len']} tokens: {100*n_trunc/len(train_df):.1f}%")

    tr = make_loader(tok, train_df, label2id, cfg["max_len"], cfg["batch"], True, cfg["seed"])
    va = make_loader(tok, val_df, label2id, cfg["max_len"], cfg["eval_batch"], False, 0)
    te = make_loader(tok, test_df, label2id, cfg["max_len"], cfg["eval_batch"], False, 0)

    model = IncidentClassifier(cfg["model"], len(labels), cfg["head"],
                               lstm_hidden=cfg["lstm_hidden"],
                               grad_checkpointing=cfg["grad_ckpt"]).to(device)
    criterion = build_loss(cfg["loss"], counts, gamma=cfg["gamma"],
                           beta=cfg["beta"], tau=cfg["tau"]).to(device)

    head_ids = {id(p) for p in model.head_parameters()}
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    enc_named = [(n, p) for n, p in model.named_parameters() if id(p) not in head_ids]
    groups = [
        {"params": [p for n, p in enc_named if not any(k in n for k in no_decay)],
         "lr": cfg["lr"], "weight_decay": 0.01},
        {"params": [p for n, p in enc_named if any(k in n for k in no_decay)],
         "lr": cfg["lr"], "weight_decay": 0.0},
        {"params": model.head_parameters(), "lr": cfg["head_lr"], "weight_decay": 0.01},
    ]
    opt = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(tr) / cfg["grad_accum"])
    total = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{run_name(cfg)}.pt"

    best_f1, best_epoch, bad, history = -1.0, -1, 0, []
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(tr, desc=f"epoch {epoch}/{cfg['epochs']}", leave=False, mininterval=2)
        for step, b in enumerate(bar, 1):
            y = b.pop("labels").to(device)
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(**b)
            loss = criterion(logits, y) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            running += loss.item() * cfg["grad_accum"]
            if step % 20 == 0:
                bar.set_postfix(loss=f"{running / step:.4f}")
            if step % cfg["grad_accum"] == 0 or step == len(tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        yv, pv = predict(model, va, device, amp)
        vm = compute_metrics(yv, pv, labels)
        history.append({"epoch": epoch, "train_loss": running / len(tr),
                        "val_macro_f1": vm["macro_f1"], "val_acc": vm["accuracy"]})
        print(f"[epoch {epoch}] loss={running/len(tr):.4f} val_macroF1={vm['macro_f1']:.4f} "
              f"val_acc={vm['accuracy']:.4f} ({time.time()-t0:.0f}s)")

        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = vm["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"[early stop] no val improvement for {bad} epochs")
                break

    # ارزیابی نهایی روی test فقط یک‌بار و با بهترین checkpoint (انتخاب‌شده با val)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    yv, pv = predict(model, va, device, amp)
    yt, pt = predict(model, te, device, amp)
    result = {
        "run": run_name(cfg), "config": cfg, "labels": labels,
        "best_epoch": best_epoch, "train_seconds": round(time.time() - t0, 1),
        "history": history,
        "val": compute_metrics(yv, pv, labels),
        "test": compute_metrics(yt, pt, labels),
        "test_predictions": pt.tolist(),
    }
    res_dir = Path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{run_name(cfg)}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[test] macroF1={result['test']['macro_f1']:.4f} acc={result['test']['accuracy']:.4f}")

    if not cfg["keep_ckpt"]:
        ckpt.unlink(missing_ok=True)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--head", default="bilstm_att", choices=HEADS)
    p.add_argument("--lstm_hidden", type=int, default=128, help="BiLSTM units per direction (paper: 128)")
    p.add_argument("--loss", default="ce", choices=LOSSES)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--split_dir", default="splits")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-3, help="lr for BiLSTM/attention/classifier")
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=2.0, help="focal gamma")
    p.add_argument("--beta", type=float, default=0.999, help="class-balanced beta")
    p.add_argument("--tau", type=float, default=1.0, help="logit-adjustment tau")
    p.add_argument("--grad_ckpt", action="store_true", help="save VRAM, ~30%% slower")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--keep_ckpt", action="store_true")
    return p


if __name__ == "__main__":
    run(vars(get_parser().parse_args()))
