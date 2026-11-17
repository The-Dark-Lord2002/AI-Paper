"""
Check the machine BEFORE the long run, and measure how long the experiment will take on it.

  1. PyTorch sees the GPU and can really compute on it
  2. BERT can be loaded (downloaded from Hugging Face the first time)
  3. a few real training steps on the real texts: seconds per step and GPU memory
  4. from that, an estimate of the hours for stage 1 (MLM) and stage 2 (fine-tuning)

  python check_setup.py                  # for a 4 GB GPU: batch 8 (the default here)
  python check_setup.py --batch-size 32  # for a 16 GB GPU
Nothing is saved; it takes a few minutes.
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

import config
from common import read_lines

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--steps", type=int, default=12, help="training steps to time (after 3 warm-up steps)")
parser.add_argument("--model", default=config.BASE_MODEL)
args = parser.parse_args()

# ------------------------------------------------------------------ 1. GPU
print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("[warning] PyTorch does not see a GPU: everything would run on the CPU (very slow for stage 1).\n"
          "          Check the NVIDIA driver, and that the CUDA build of PyTorch is installed (install script).")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    major, minor = torch.cuda.get_device_capability(0)
    print(f"[ok] GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB, compute capability {major}.{minor}")
    try:
        x = torch.randn(512, 512, device=device, dtype=torch.float16)
        (x @ x).sum().item()
    except RuntimeError as error:
        raise SystemExit(f"[error] The GPU is visible but cannot compute: {error}\n"
                         "This PyTorch build does not support this (older) GPU. Install the build the install "
                         "script chooses (torch 2.7.1 for CUDA 12.6 supports GTX 900 / 1000 cards).")
    print("[ok] the GPU computes in float16")

# ------------------------------------------------------------------ 2. BERT
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
print(f"[ok] {args.model} loaded")

# ------------------------------------------------------------------ 3. time real training steps
corpus = Path(config.CORPUS_DIR)
sets = {name: read_lines(corpus / f"{name}_train.txt") for name in ("task", "external")
        if (corpus / f"{name}_train.txt").exists()}
if not sets:
    raise SystemExit("[missing] corpus/: copy it from the laptop (make_upload_zip.sh does) or run the pipeline, "
                     "which builds it.")
use_amp = device.type == "cuda"
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
seconds = {}
for name, texts in sets.items():
    model.train()
    times = []
    for step in range(args.steps + 3):
        batch = [texts[(step * args.batch_size + i) % len(texts)] for i in range(args.batch_size)]
        enc = tokenizer(batch, truncation=True, max_length=config.MLM_MAX_TOKENS, padding=True, return_tensors="pt")
        start = time.time()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                         labels=enc["input_ids"].to(device)).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= 3:                                   # the first steps are slower (warm-up)
            times.append(time.time() - start)
    seconds[name] = sum(times) / len(times)
    print(f"[ok] {name}: {len(texts):,} texts, {seconds[name]:.2f} s per step of {args.batch_size} texts")
if device.type == "cuda":
    used, total = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[{'ok' if used < 0.9 * total else 'warning'}] GPU memory used: {used:.1f} of {total:.1f} GB"
          + ("" if used < 0.9 * total else "  -> use a smaller --batch-size / --mlm-batch-size"))

# ------------------------------------------------------------------ 4. estimate
print("\nEstimated time on this machine (rough; the real number of stage-1 epochs is decided by early stopping):")
hours = lambda n_texts, s, epochs=1: n_texts / args.batch_size * s * epochs / 3600
total = 0.0
if "task" in sets:
    tapt = hours(len(sets["task"]), seconds["task"])
    finetune = hours(len(sets["task"]), seconds["task"], config.EPOCHS)       # same texts, 5 epochs
    print(f"  TAPT (stage 1 on your training reports): {tapt * 60:.0f} min per epoch, about {tapt * 20:.1f} h for ~20 epochs")
    print(f"  one fine-tuning run (stage 2):           about {finetune:.1f} h")
    print(f"  14 fine-tuning runs (base seed 3, TAPT, DAPT; 2 models x 3 seeds): about {14 * finetune:.0f} h")
    total += tapt * 20 + 14 * finetune
if "external" in sets:
    dapt = hours(len(sets["external"]), seconds["external"])
    print(f"  DAPT (stage 1 on the other MAIB texts):  {dapt:.1f} h per epoch, about {3 * dapt:.0f} h for 3 epochs")
    total += 3 * dapt
print(f"  TOTAL: about {total:.0f} h = {total / 24:.1f} days of continuous running "
      f"(it can be stopped and continued at any time)")
