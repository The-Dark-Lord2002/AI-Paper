"""
Move the results of an earlier stage-1 setting out of the way, so the pipeline trains the adapted BERTs again
with the current settings. Nothing is deleted: everything is MOVED to archive/<name>/ and can be moved back.

Why it is needed: run_pipeline.py skips a stage-1 model that already exists in models/ and a fine-tuning run
whose result file already exists in results/. After changing the stage-1 settings (more texts, more epochs),
those old files would be reused, so they have to go.

What is moved:
  models/bert-maib-*/                     the adapted BERTs (and any unfinished *.resume.pt)
  results/bert-maib-*                     the fine-tuning results obtained with them
  reports/pretrain_*.json                 their stage-1 summaries
  reports/comparison.md, .csv, significance.csv
What stays: the original-BERT results (results/bert__*.json, results/base__*.json), data/ and corpus/.

  python archive_previous_run.py light          # shows what would be moved
  python archive_previous_run.py light --yes    # moves it to archive/light/
"""
import argparse
import shutil
from pathlib import Path

import config

parser = argparse.ArgumentParser()
parser.add_argument("name", help="name of the archive folder, e.g. 'light'")
parser.add_argument("--yes", action="store_true", help="really move the files (without it: only show them)")
args = parser.parse_args()

target = Path("archive") / args.name
if target.exists():
    raise SystemExit(f"{target} already exists: choose another name.")

models, results, reports = Path(config.MODELS_DIR), Path(config.RESULTS_DIR), Path(config.REPORTS_DIR)
items = (sorted(models.glob("bert-maib-*")) + sorted(results.glob("bert-maib-*"))
         + sorted(reports.glob("pretrain_*.json"))
         + [p for p in (reports / "comparison.md", reports / "comparison.csv", reports / "significance.csv") if p.exists()])
if not items:
    raise SystemExit("Nothing to archive.")

for path in items:
    print(f"{'move' if args.yes else 'would move'}  {path}  ->  {target / path}")
    if args.yes:
        (target / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target / path))
print(f"\n{len(items)} items " + (f"moved to {target}/." if args.yes else "would be moved. Add --yes to move them."))
