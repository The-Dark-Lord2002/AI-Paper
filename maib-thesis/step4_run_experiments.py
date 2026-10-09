"""
STEP 4 - Run the whole experiment table (proposal steps 2b, 2c, 3 and 4).

config.EXPERIMENTS has 6 (model, loss) pairs, each trained with every seed in config.SEEDS:
6 x 3 = 18 training runs.

A run whose result file already exists is skipped. If the computer stops half way,
run the same command again and it continues where it stopped.

  python step4_run_experiments.py                     # laptop GPU (batch 8, 4 accumulation steps)
  python step4_run_experiments.py --batch-size 32     # bigger GPU (Kaggle / Colab): same result, faster
  python step4_run_experiments.py --seeds 1 2         # fewer seeds if short on time
  python step4_run_experiments.py --paper-settings    # optional: reference model with the paper's lr = 1e-6
"""
import argparse
from pathlib import Path

import config
from step3_train import run_name, train

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
parser.add_argument("--paper-settings", action="store_true",
                    help="train only bert_bilstm + ce with the reference paper's settings (config.PAPER_SETTINGS)")
args = parser.parse_args()

if args.paper_settings:
    experiments, extra = [("bert_bilstm", "ce")], config.PAPER_SETTINGS
else:
    experiments, extra = config.EXPERIMENTS, {}

plan = [(model, loss, seed) for model, loss in experiments for seed in args.seeds]
print(f"{len(plan)} runs planned")

for i, (model, loss, seed) in enumerate(plan, start=1):
    name = run_name(model, loss, seed, extra.get("tag", ""))
    if (Path(config.RESULTS_DIR) / f"{name}.json").exists():
        print(f"[{i}/{len(plan)}] already done: {name}")
        continue
    print(f"[{i}/{len(plan)}] training: {name}")
    train(model, loss, seed, batch_size=args.batch_size, **extra)

print("\nAll runs finished. Next: python step5_report.py")
