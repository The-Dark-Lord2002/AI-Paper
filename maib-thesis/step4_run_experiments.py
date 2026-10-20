"""
STEP 4 - Run the whole experiment table (proposal steps 2b, 2c, 3 and 4).

config.EXPERIMENTS has 6 (model, loss) pairs, each trained with every seed in config.SEEDS:
6 x 2 = 12 training runs.

A run whose result file already exists is skipped. If the computer stops half way,
run the same command again and it continues where it stopped.

  python step4_run_experiments.py                     # laptop GPU (batch 8, 4 accumulation steps)
  python step4_run_experiments.py --batch-size 32     # bigger GPU: same result, faster
  python step4_run_experiments.py --seeds 1 2 3       # a third seed
  python step4_run_experiments.py --paper-settings    # optional: reference model with the paper's lr = 1e-6
  python step4_run_experiments.py --oversample        # oversampling runs (config.OVERSAMPLE_EXPERIMENTS)
  python step4_run_experiments.py --augment           # LLM data augmentation (config.AUGMENT_EXPERIMENTS), after step 7
  python step4_run_experiments.py --bilstm-sizes      # bigger BiLSTM (config.BILSTM_SIZES)
"""
import argparse
from pathlib import Path

import config
from step3_train import default_tag, run_name, train

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
group = parser.add_mutually_exclusive_group()
group.add_argument("--paper-settings", action="store_true",
                   help="train only bert_bilstm + ce with the reference paper's settings (config.PAPER_SETTINGS)")
group.add_argument("--oversample", action="store_true",
                   help="train config.OVERSAMPLE_EXPERIMENTS with rare classes shown more often")
group.add_argument("--augment", action="store_true",
                   help="train config.AUGMENT_EXPERIMENTS on real + rewritten reports (run step7_augment.py first)")
group.add_argument("--bilstm-sizes", action="store_true",
                   help="train bert_bilstm_att + config.BILSTM_SIZE_LOSS with every size in config.BILSTM_SIZES")
args = parser.parse_args()

# every entry: (model, loss, extra settings for train())
if args.paper_settings:
    configs = [("bert_bilstm", "ce", config.PAPER_SETTINGS)]
elif args.oversample:
    configs = [(model, loss, dict(oversample=True)) for model, loss in config.OVERSAMPLE_EXPERIMENTS]
elif args.augment:
    configs = [(model, loss, dict(augment=True)) for model, loss in config.AUGMENT_EXPERIMENTS]
elif args.bilstm_sizes:
    configs = [("bert_bilstm_att", config.BILSTM_SIZE_LOSS, dict(lstm_hidden=hidden, lstm_layers=layers))
               for hidden, layers in config.BILSTM_SIZES]
else:
    configs = [(model, loss, {}) for model, loss in config.EXPERIMENTS]

plan = [(model, loss, seed, extra) for model, loss, extra in configs for seed in args.seeds]
print(f"{len(plan)} runs planned")

for i, (model, loss, seed, extra) in enumerate(plan, start=1):
    tag = extra["tag"] if "tag" in extra else default_tag(**extra)
    name = run_name(model, loss, seed, tag)
    if (Path(config.RESULTS_DIR) / f"{name}.json").exists():
        print(f"[{i}/{len(plan)}] already done: {name}")
        continue
    print(f"[{i}/{len(plan)}] training: {name}")
    train(model, loss, seed, batch_size=args.batch_size, **extra)

print("\nAll runs finished. Next: python step5_report.py")
