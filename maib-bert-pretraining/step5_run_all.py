"""
STEP 5 - Fine-tune every configuration of config.FINETUNE_CONFIGS with every encoder and every seed.

Encoders: the original BERT ("base") and every adapted model in models/ (step 3).
A run whose result file already exists is skipped, so the script can be stopped and started again.

  python step5_run_all.py                                     # base + all models in models/
  python step5_run_all.py --encoders models/bert-maib-external
  python step5_run_all.py --skip-base                         # base results copied from the paper (see README)
"""
import argparse
from pathlib import Path

import config
from step4_finetune import run_name, train

parser = argparse.ArgumentParser()
parser.add_argument("--encoders", nargs="+", help="encoders to fine-tune (default: base + every folder in models/)")
parser.add_argument("--skip-base", action="store_true", help="do not fine-tune the original BERT")
args = parser.parse_args()

encoders = args.encoders or sorted(str(p) for p in Path(config.MODELS_DIR).glob("*") if (p / "config.json").exists())
if not args.skip_base and config.BASE_MODEL not in encoders:
    encoders = [config.BASE_MODEL] + encoders
jobs = [(e, m, l, s) for e in encoders for m, l in config.FINETUNE_CONFIGS for s in config.SEEDS]
print(f"{len(jobs)} runs: {len(encoders)} encoders x {len(config.FINETUNE_CONFIGS)} configurations "
      f"x {len(config.SEEDS)} seeds")
for i, (encoder, model, loss, seed) in enumerate(jobs, start=1):
    if (Path(config.RESULTS_DIR) / f"{run_name(encoder, model, loss, seed)}.json").exists():
        print(f"[{i}/{len(jobs)}] {run_name(encoder, model, loss, seed)}: already done, skipped")
        continue
    print(f"[{i}/{len(jobs)}]", end="")
    train(model, loss, seed, encoder=encoder)
