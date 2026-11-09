"""
STEP 3 - Stage 1, unsupervised: continue the pre-training of BERT with masked language modelling (MLM).

How MLM works (the task BERT was originally pre-trained on, Devlin et al., 2019):
  1. take a text                                  "the vessel ran aground on a sandbank"
  2. hide 15% of its word-pieces at random        "the vessel ran [MASK] on a sandbank"
  3. BERT predicts each hidden word-piece from its context; the loss is the cross-entropy of these
     predictions only. No label is needed: the text is its own answer ("self-supervised").
  Of the chosen 15%, 80% become [MASK], 10% a random word-piece and 10% stay unchanged, so BERT also
  learns to check the words it can see.

On maritime texts BERT learns words and phrases that are rare in its original data (Wikipedia and books):
"allided", "girting", "FRC", "ingress", "pilot ladder", ... This is domain-adaptive pre-training (DAPT);
on the task's own training texts it is task-adaptive pre-training (TAPT) (Gururangan et al., 2020).

  python step3_pretrain_mlm.py --corpus external         # DAPT: the other dataset (step 2)
  python step3_pretrain_mlm.py --corpus task             # TAPT: only the TRAINING reports of our dataset
  python step3_pretrain_mlm.py --corpus external-then-task   # TAPT starting from the DAPT model (run external first)
  python step3_pretrain_mlm.py --corpus external+task    # both corpora mixed in one run
  python step3_pretrain_mlm.py --corpus external --max-texts 20000 --epochs 3   # a shorter run on a big corpus

How long it trains (the principled way): the number of epochs is not chosen by hand. After every epoch, the
MLM loss is measured on the VALIDATION reports of the labelled dataset (their text only) with the same fixed
masks every time. Training stops when this loss has not improved for config.MLM_PATIENCE epochs, and the epoch
with the lowest loss is kept. config.MLM_EPOCHS is only a ceiling; if the best epoch is the last allowed one,
the loss was still falling, the run is reported as "not converged" and the ceiling should be raised.
The test reports are never read.

If the run stops (power cut, Ctrl+C), the same command continues after the last finished epoch: the state is
saved after every epoch in models/bert-maib-<corpus>.resume.pt (about 2 GB, deleted when the run ends).

Output: models/bert-maib-<corpus>/   the adapted BERT (weights + tokenizer), used by step 4
        reports/pretrain_<corpus>.json  MLM loss and perplexity before and after, per epoch
"""
import argparse
import copy
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

import config
from common import (load_checkpoint, load_split, read_lines, rng_state, save_checkpoint, save_json, set_rng_state,
                    set_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", choices=list(config.MLM_EPOCHS), required=True)
parser.add_argument("--base-model", help="the BERT to start from (default: the original BERT; for "
                                         "external-then-task: models/bert-maib-external)")
parser.add_argument("--epochs", type=int, help="maximum epochs (default: config.MLM_EPOCHS of the corpus)")
parser.add_argument("--patience", type=int, help="early-stopping patience (default: config.MLM_PATIENCE of the corpus)")
parser.add_argument("--batch-size", type=int, default=config.MLM_BATCH_SIZE)
parser.add_argument("--lr", type=float, default=config.MLM_LR)
parser.add_argument("--max-texts", type=int,
                    help="use at most this many texts of the other dataset (random, fixed seed): a shorter run")
args = parser.parse_args()
args.epochs = args.epochs or config.MLM_EPOCHS[args.corpus]
args.patience = args.patience or config.MLM_PATIENCE[args.corpus]

name = "bert-maib-" + args.corpus.replace("+", "-")
out_dir = Path(config.MODELS_DIR) / name
corpus_dir = Path(config.CORPUS_DIR)

# external-then-task = TAPT on top of DAPT: the texts are the training reports ("task"), but training starts
# from the BERT already adapted to the other dataset instead of the original BERT.
if args.corpus == "external-then-task":
    CORPUS_PARTS = ["task"]
    args.base_model = args.base_model or str(Path(config.MODELS_DIR) / "bert-maib-external")
    if not (Path(args.base_model) / "config.json").exists():
        raise SystemExit(f"{args.base_model} not found: run stage 1 on 'external' first "
                         "(python step3_pretrain_mlm.py --corpus external).")
else:
    CORPUS_PARTS = args.corpus.split("+")
    args.base_model = args.base_model or config.BASE_MODEL


def mask_tokens(input_ids, special, tokenizer, generator):
    """
    Hide MLM_PROBABILITY of the word-pieces. Returns the masked input and the labels: the original id at
    the hidden positions and -100 everywhere else (-100 = ignored by the loss).
    special: True for [CLS], [SEP] and padding, which are never hidden.
    """
    labels = input_ids.clone()
    probability = torch.full(input_ids.shape, config.MLM_PROBABILITY).masked_fill(special, 0.0)
    chosen = torch.bernoulli(probability, generator=generator).bool()
    for row in range(len(chosen)):                     # every text hides at least one word-piece
        if not chosen[row].any():
            chosen[row, (~special[row]).nonzero()[0]] = True
    labels[~chosen] = -100

    inputs = input_ids.clone()
    to_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8), generator=generator).bool() & chosen
    inputs[to_mask] = tokenizer.mask_token_id                                     # 80%: [MASK]
    to_random = torch.bernoulli(torch.full(input_ids.shape, 0.5), generator=generator).bool() & chosen & ~to_mask
    random_ids = torch.randint(len(tokenizer), input_ids.shape, generator=generator)
    inputs[to_random] = random_ids[to_random]                                     # 10%: a random word-piece
    return inputs, labels                                                         # 10%: unchanged


def make_loader(tokenizer, texts, batch_size, shuffle, seed):
    encoded = tokenizer(texts, truncation=True, max_length=config.MLM_MAX_TOKENS, return_special_tokens_mask=True)
    examples = list(zip(encoded["input_ids"], encoded["special_tokens_mask"]))

    def collate(batch):
        longest = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), longest), tokenizer.pad_token_id, dtype=torch.long)
        special = torch.ones((len(batch), longest), dtype=torch.bool)             # padding counts as special
        attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, (ids, spec) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            special[i, :len(ids)] = torch.tensor(spec, dtype=torch.bool)
            attention_mask[i, :len(ids)] = 1
        return input_ids, special, attention_mask

    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


def fixed_masks(tokenizer, texts):
    """Mask an evaluation set ONCE, so every epoch and the original BERT are scored on the same task."""
    generator = torch.Generator().manual_seed(config.MLM_SEED)
    batches = []
    # small batches: the MLM output has one score per vocabulary word (30,522) for every word-piece
    for input_ids, special, attention_mask in make_loader(tokenizer, texts, config.MLM_EVAL_BATCH_SIZE, False, 0):
        inputs, labels = mask_tokens(input_ids, special, tokenizer, generator)
        batches.append((inputs, attention_mask, labels))
    return batches


@torch.no_grad()
def mlm_loss(model, batches, device, use_amp):
    """Mean cross-entropy over all hidden word-pieces of an evaluation set. Perplexity = exp(loss)."""
    model.eval()
    total, count = 0.0, 0
    for inputs, attention_mask, labels in batches:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=inputs.to(device), attention_mask=attention_mask.to(device)).logits
        hidden = labels.to(device) != -100                     # score only the hidden word-pieces
        loss = torch.nn.functional.cross_entropy(logits[hidden].float(), labels.to(device)[hidden], reduction="sum")
        total, count = total + loss.item(), count + int(hidden.sum())
    return total / count


def scores(model, eval_sets, device, use_amp):
    out = {}
    for key, batches in eval_sets.items():
        loss = mlm_loss(model, batches, device, use_amp)
        out[key] = {"loss": round(loss, 4), "perplexity": round(math.exp(loss), 2)}
    return out


def main():
    set_seed(config.MLM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    grad_accum = max(1, config.MLM_EFFECTIVE_BATCH // args.batch_size)

    # ------------------------------------------------------------------ texts
    texts = []
    for part in CORPUS_PARTS:
        path = corpus_dir / f"{part}_train.txt"
        if not path.exists():
            raise SystemExit(f"{path} not found. Run step2_build_corpus.py first.")
        lines = read_lines(path)
        if not lines:
            raise SystemExit(f"{path} is empty: there is no text of the other dataset. Download it with "
                             "step0_download_sources.py, then run step2_build_corpus.py again.")
        if part == "external" and args.max_texts and len(lines) > args.max_texts:
            lines = random.Random(config.MLM_SEED).sample(lines, args.max_texts)
        texts += lines
    _, val, _, _ = load_split()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    eval_sets = {"labelled_val": fixed_masks(tokenizer, val["text"].tolist())}    # selects the best epoch
    external_val = corpus_dir / "external_val.txt"
    if external_val.exists() and read_lines(external_val):
        eval_sets["external_val"] = fixed_masks(tokenizer, read_lines(external_val))
    print(f"===== {name} =====  {len(texts)} texts, {sum(len(t.split()) for t in texts):,} words, "
          f"starting from {args.base_model}, device={device}, batch={args.batch_size} x {grad_accum}, "
          f"at most {args.epochs} epochs, patience {args.patience}")

    # ------------------------------------------------------------------ model and optimiser
    # The learning rate rises linearly during the first MLM_WARMUP_FRACTION of the updates (warm-up: no large
    # steps while the optimiser state is still empty), then falls linearly towards 0 at the ceiling of epochs.
    # Early stopping usually ends the run before the ceiling, which is normal.
    model = AutoModelForMaskedLM.from_pretrained(args.base_model).to(device)
    loader = make_loader(tokenizer, texts, args.batch_size, shuffle=True, seed=config.MLM_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.MLM_WEIGHT_DECAY)
    total_updates = -(-len(loader) // grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.MLM_WARMUP_FRACTION * total_updates),
                                                total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mask_generator = torch.Generator().manual_seed(config.MLM_SEED)               # new masks every epoch

    # ------------------------------------------------------------------ start, or continue a stopped run
    resume_path = Path(config.MODELS_DIR) / f"{name}.resume.pt"
    # A checkpoint is only continued by a run with exactly these settings (otherwise the result would be a mix).
    settings = {"corpus": args.corpus, "base_model": args.base_model, "texts": len(texts), "epochs": args.epochs,
                "patience": args.patience, "batch_size": args.batch_size, "lr": args.lr, "max_texts": args.max_texts}
    state = load_checkpoint(resume_path, settings)
    if state is None:
        before = scores(model, eval_sets, device, use_amp)
        print(f"[before] {before}")
        best_loss, best_epoch, best_weights, history, first_epoch, minutes_before = \
            before["labelled_val"]["loss"], 0, None, [], 1, 0.0
    else:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        mask_generator.set_state(state["mask_rng"])
        loader.generator.set_state(state["loader_rng"])
        before, best_loss, best_epoch, best_weights, history = (state[k] for k in
                                                                ("before", "best_loss", "best_epoch", "best_weights", "history"))
        first_epoch = args.epochs + 1 if state["stopped"] else state["epoch"] + 1
        minutes_before = state["minutes"]

    # ------------------------------------------------------------------ training loop
    start = time.time()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, mininterval=2)
        for step, (input_ids, special, attention_mask) in enumerate(progress, start=1):
            inputs, labels = mask_tokens(input_ids, special, tokenizer, mask_generator)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=inputs.to(device), attention_mask=attention_mask.to(device),
                             labels=labels.to(device)).loss
            scaler.scale(loss / grad_accum).backward()
            running += loss.item()
            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            progress.set_postfix(loss=f"{running / step:.4f}")

        epoch_scores = scores(model, eval_sets, device, use_amp)
        history.append({"epoch": epoch, "train_loss": round(running / len(loader), 4), **epoch_scores})
        val_loss = epoch_scores["labelled_val"]["loss"]
        minutes = minutes_before + (time.time() - start) / 60
        print(f"[epoch {epoch}] train loss={running / len(loader):.4f}  {epoch_scores}  "
              f"({minutes:.1f} min; at most {minutes / epoch * (args.epochs - epoch):.0f} min left)")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        stop = epoch - best_epoch >= args.patience      # early stopping
        save_checkpoint({"settings": settings, "epoch": epoch, "stopped": stop or epoch == args.epochs,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "rng": rng_state(), "mask_rng": mask_generator.get_state(),
                         "loader_rng": loader.generator.get_state(), "before": before, "best_loss": best_loss,
                         "best_epoch": best_epoch, "best_weights": best_weights, "history": history,
                         "minutes": minutes}, resume_path)
        if stop:
            print(f"[early stop] no improvement for {args.patience} epochs; the best was epoch {best_epoch}")
            break

    # ------------------------------------------------------------------ save the best epoch
    if best_weights is None:
        print("[note] no epoch beat the original BERT on the validation reports; saving the last epoch anyway")
        best_epoch = history[-1]["epoch"]
    else:
        model.load_state_dict(best_weights)
    after = scores(model, eval_sets, device, use_amp)
    # Converged = early stopping ended the run. If the best epoch is the last one that was run and no early stop
    # happened, the validation loss was still falling: more epochs would probably help.
    converged = len(history) - (best_epoch if best_weights is not None else 0) >= args.patience
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    summary = {
        "model": name, "base_model": args.base_model, "corpus": args.corpus,
        "corpus_texts": len(texts), "corpus_words": sum(len(t.split()) for t in texts),
        "settings": {"mlm_probability": config.MLM_PROBABILITY, "max_tokens": config.MLM_MAX_TOKENS,
                     "batch_size": args.batch_size, "grad_accum": grad_accum, "lr": args.lr,
                     "epochs": args.epochs, "patience": args.patience, "max_texts": args.max_texts,
                     "device": str(device),
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "best_epoch": best_epoch, "epochs_run": len(history), "converged": converged, "train_minutes": round(minutes_before + (time.time() - start) / 60, 1),
        "before": before, "after": after, "history": history,
    }
    save_json(summary, Path(config.REPORTS_DIR) / f"pretrain_{args.corpus.replace('+', '-')}.json")
    resume_path.unlink(missing_ok=True)                  # finished: the checkpoint is no longer needed
    print(f"[after] {after}\nSaved {out_dir}/ (best epoch {best_epoch} of {len(history)} run)")
    if not converged:
        print(f"[warning] NOT CONVERGED: the validation loss was still falling at the last epoch ({args.epochs}). "
              f"Raise the ceiling (--epochs, or config.MLM_EPOCHS) and run again for a complete pre-training.")


if __name__ == "__main__":
    main()
